#include <stdio.h>
#include <string.h>

#include "py/obj.h"
#include "py/runtime.h"
#include "py/mphal.h"
#include "py/objarray.h"
#include "py/binary.h"
#include "py_assert.h"
#include "mperrno.h"
#include "mphalport.h"
#include "modMaix.h"
#include "imlib.h"

#include "sleep.h"
#include "lcd.h"
#include "sysctl.h"
#include "fpioa.h"
#include "sipeed_sk9822.h"
#include "py_image.h"

// --- 新增: 引入底层 I2S 和 DMA 驱动，以及我们的 DOA 算法头文件 ---
#include "i2s.h"
#include "dmac.h"
#include "m_estimation.h"

#define PLL2_OUTPUT_FREQ 45158400UL

// --- 全新的 DMA 数据接收池 ---
STATIC volatile uint8_t rx_flag = 0;
// 1024点 * 4通道 * 左右2声道 = 8192个 int32_t 数据
int32_t i2s_rx_buf[FFT_N * 8] __attribute__((aligned(128)));
// 用于传递给 M-估计管线的 7 个浮点通道裸流
float mic_raw_float[NUM_MICS][FFT_N];

// DMA 接收满 1024 点后的中断回调
static int i2s_dma_cb(void *ctx)
{
    rx_flag = 1;
    return 0;
}

STATIC mp_obj_t Maix_mic_array_init(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args)
{
    enum {
        ARG_i2s_d0,
        ARG_i2s_d1,
        ARG_i2s_d2,
        ARG_i2s_d3,
        ARG_i2s_ws,
        ARG_i2s_sclk,

        ARG_sk9822_dat,
        ARG_sk9822_clk,
    };

    static const mp_arg_t allowed_args[]={
        {MP_QSTR_i2s_d0,    MP_ARG_INT, {.u_int = 23}},
        {MP_QSTR_i2s_d1,    MP_ARG_INT, {.u_int = 22}},
        {MP_QSTR_i2s_d2,    MP_ARG_INT, {.u_int = 21}},
        {MP_QSTR_i2s_d3,    MP_ARG_INT, {.u_int = 20}},
        {MP_QSTR_i2s_ws,    MP_ARG_INT, {.u_int = 19}},
        {MP_QSTR_i2s_sclk,  MP_ARG_INT, {.u_int = 18}},

        {MP_QSTR_sk9822_dat, MP_ARG_INT, {.u_int = 24}},
        {MP_QSTR_sk9822_clk, MP_ARG_INT, {.u_int = 25}},
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    // 引脚映射 (FPIOA) 保持不变
    fpioa_set_function(args[ARG_i2s_d0].u_int, FUNC_I2S0_IN_D0);
    fpioa_set_function(args[ARG_i2s_d1].u_int, FUNC_I2S0_IN_D1);
    fpioa_set_function(args[ARG_i2s_d2].u_int, FUNC_I2S0_IN_D2);
    fpioa_set_function(args[ARG_i2s_d3].u_int, FUNC_I2S0_IN_D3);
    fpioa_set_function(args[ARG_i2s_ws].u_int, FUNC_I2S0_WS);
    fpioa_set_function(args[ARG_i2s_sclk].u_int, FUNC_I2S0_SCLK);
    fpioa_set_function(args[ARG_sk9822_dat].u_int, FUNC_GPIOHS0 + SK9822_DAT_GPIONUM);
    fpioa_set_function(args[ARG_sk9822_clk].u_int, FUNC_GPIOHS0 + SK9822_CLK_GPIONUM);

    sipeed_init_mic_array_led();

    // --- 抛弃旧的 lib_mic_init，改用纯硬件级 I2S 初始化 ---
    i2s_init(I2S_DEVICE_0, I2S_RECEIVER, 0x0F);
    for(int i = 0; i < 4; i++){
        i2s_rx_channel_config(I2S_DEVICE_0, I2S_CHANNEL_0 + i,
                              RESOLUTION_32_BIT, SCLK_CYCLES_32,
                              TRIGGER_LEVEL_4, STANDARD_MODE);
    }
    i2s_set_sample_rate(I2S_DEVICE_0, SAMPLE_RATE);

    // 初始化你的算法阵列物理矩阵
    init_array_geometry();

    // 开启第一次底层 DMA 录音 (非阻塞)
    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4, i2s_dma_cb, NULL);

    return mp_const_true;
}

MP_DEFINE_CONST_FUN_OBJ_KW(Maix_mic_array_init_obj, 0, Maix_mic_array_init);

STATIC mp_obj_t Maix_mic_array_deinit(void)
{
    // 这里可以保持简单，无需复杂卸载
    return mp_const_true;
}

MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_deinit_obj, Maix_mic_array_deinit);

// 因为我们抛弃了自带的热力图算法，调用 get_map 时直接抛出错误提示用户
STATIC mp_obj_t Maix_mic_array_get_map(void)
{
    mp_raise_ValueError("Thermal map is disabled. Using high-precision DOA (get_dir) instead.");
    return mp_const_none;
}

MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_map_obj, Maix_mic_array_get_map);

// --- 全新重构的高精度角度获取函数 ---
STATIC mp_obj_t Maix_mic_array_get_dir(void)
{
    // 1. 等待 DMA 接收完毕
    uint32_t timeout = 2000000;
    while(rx_flag == 0 && timeout > 0) { timeout--; }
    if(timeout == 0)
    {
        mp_raise_OSError(MP_ETIMEDOUT);
        return mp_const_false;
    }
    rx_flag = 0;

    // 2. 将 I2S 硬件交织的数据提取出来，转换为 Float 格式交给算法
    // I2S 接收为 32位对齐，有效数据通常在高 16 位
    for (int i = 0; i < FFT_N; i++) {
        mic_raw_float[0][i] = (float)(i2s_rx_buf[i*8 + 0] >> 16);
        mic_raw_float[1][i] = (float)(i2s_rx_buf[i*8 + 1] >> 16);
        mic_raw_float[2][i] = (float)(i2s_rx_buf[i*8 + 2] >> 16);
        mic_raw_float[3][i] = (float)(i2s_rx_buf[i*8 + 3] >> 16);
        mic_raw_float[4][i] = (float)(i2s_rx_buf[i*8 + 4] >> 16);
        mic_raw_float[5][i] = (float)(i2s_rx_buf[i*8 + 5] >> 16);
        mic_raw_float[6][i] = (float)(i2s_rx_buf[i*8 + 6] >> 16);
    }

    // 3. 立即重启 DMA 接收下一帧，保证录音连续性
    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4, i2s_dma_cb, NULL);

    // 4. 调用硬核 DOA 管线进行高斯-牛顿微调解算
    float final_angle = run_doa_pipeline(mic_raw_float);

    // 5. 返回计算出的精确角度
    return mp_obj_new_float(final_angle);
}

// 宏定义更新：原来有参数，现在无参数，所以使用 _0
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_dir_obj, Maix_mic_array_get_dir);

// --- 保留原有的 LED 控制逻辑不变 ---
STATIC mp_obj_t Maix_mic_array_set_led(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args)
{
    int index, brightness[12] = {0}, led_color[12] = {0}, color[3] = {0};

    mp_obj_t *items;
    mp_obj_get_array_fixed_n(pos_args[0], 12, &items);

    for(index= 0; index < 12; index++)
        brightness[index] = mp_obj_get_int(items[index]);

    mp_obj_get_array_fixed_n(pos_args[1], 3, &items);
    for(index = 0; index < 3; index++)
        color[index] = mp_obj_get_int(items[index]);

    uint32_t set_color = (color[2] << 16) | (color[1] << 8) | (color[0]);

    for (index = 0; index < 12; index++)
    {
        led_color[index] = (brightness[index] / 2) > 1 ? (((0xe0 | (brightness[index] * 2)) << 24) | set_color) : 0xe0000000;
    }

    sysctl_disable_irq();
    sk9822_start_frame();
    for (index = 0; index < 12; index++)
    {
        sk9822_send_data(led_color[index]);
    }
    sk9822_stop_frame();
    sysctl_enable_irq();

    return mp_const_true;
}

MP_DEFINE_CONST_FUN_OBJ_KW(Maix_mic_array_set_led_obj, 2, Maix_mic_array_set_led);

STATIC const mp_rom_map_elem_t Maix_mic_array_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_init), MP_ROM_PTR(&Maix_mic_array_init_obj) },
    { MP_ROM_QSTR(MP_QSTR_deinit), MP_ROM_PTR(&Maix_mic_array_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_dir), MP_ROM_PTR(&Maix_mic_array_get_dir_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_led), MP_ROM_PTR(&Maix_mic_array_set_led_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_map), MP_ROM_PTR(&Maix_mic_array_get_map_obj) },
};

STATIC MP_DEFINE_CONST_DICT(Maix_mic_array_dict, Maix_mic_array_locals_dict_table);

const mp_obj_type_t Maix_mic_array_type = {
    { &mp_type_type },
    .name = MP_QSTR_MIC_ARRAY,
    .locals_dict = (mp_obj_dict_t*)&Maix_mic_array_dict,
};

// =========================================================================
// 核心黑科技：单编译单元包含！
// 将你的核心算法文件直接吸纳到这里，彻底免去修改 CMakeLists.txt 的烦恼！
// =========================================================================
#include "m_estimation.c"
