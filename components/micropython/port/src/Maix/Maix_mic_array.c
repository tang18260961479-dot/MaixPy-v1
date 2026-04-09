#include <stdio.h>
#include <string.h>
#include "py/obj.h"
#include "py/runtime.h"
#include "py/mphal.h"
#include "py/objarray.h"
#include "py/binary.h"
#include "mperrno.h"
#include "modMaix.h"
#include "sleep.h"
#include "sysctl.h"
#include "fpioa.h"
#include "i2s.h"
#include "dmac.h"

#define PLL2_OUTPUT_FREQ 45158400UL
#define FFT_N 1024
#define SAMPLE_RATE 48000

STATIC volatile uint8_t rx_flag = 0;
int32_t i2s_rx_buf[FFT_N * 8] __attribute__((aligned(128)));

static int i2s_dma_cb(void *ctx) { 
    rx_flag = 1; 
    return 0; 
}

STATIC mp_obj_t Maix_mic_array_init(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    enum { ARG_i2s_d0, ARG_i2s_d1, ARG_i2s_d2, ARG_i2s_d3, ARG_i2s_ws, ARG_i2s_sclk, };
    static const mp_arg_t allowed_args[]={
        {MP_QSTR_i2s_d0, MP_ARG_INT, {.u_int = 23}}, {MP_QSTR_i2s_d1, MP_ARG_INT, {.u_int = 22}}, 
        {MP_QSTR_i2s_d2, MP_ARG_INT, {.u_int = 21}}, {MP_QSTR_i2s_d3, MP_ARG_INT, {.u_int = 20}}, 
        {MP_QSTR_i2s_ws, MP_ARG_INT, {.u_int = 19}}, {MP_QSTR_i2s_sclk, MP_ARG_INT, {.u_int = 18}}, 
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)]; 
    int i;
    
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    
    fpioa_set_function(args[ARG_i2s_d0].u_int, FUNC_I2S0_IN_D0); 
    fpioa_set_function(args[ARG_i2s_d1].u_int, FUNC_I2S0_IN_D1); 
    fpioa_set_function(args[ARG_i2s_d2].u_int, FUNC_I2S0_IN_D2); 
    fpioa_set_function(args[ARG_i2s_d3].u_int, FUNC_I2S0_IN_D3); 
    fpioa_set_function(args[ARG_i2s_ws].u_int, FUNC_I2S0_WS); 
    fpioa_set_function(args[ARG_i2s_sclk].u_int, FUNC_I2S0_SCLK); 
    
    sysctl_pll_set_freq(SYSCTL_PLL2, PLL2_OUTPUT_FREQ); 
    sysctl_clock_enable(SYSCTL_CLOCK_I2S0);
    i2s_init(I2S_DEVICE_0, I2S_RECEIVER, 0x0F);
    
    for(i = 0; i < 4; i++) {
        i2s_rx_channel_config(I2S_DEVICE_0, I2S_CHANNEL_0 + i, RESOLUTION_32_BIT, SCLK_CYCLES_32, TRIGGER_LEVEL_4, STANDARD_MODE);
    }
    i2s_set_sample_rate(I2S_DEVICE_0, SAMPLE_RATE);
    
    dmac_set_irq(DMAC_CHANNEL4, i2s_dma_cb, NULL, 3);
    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);
    return mp_const_true;
}
MP_DEFINE_CONST_FUN_OBJ_KW(Maix_mic_array_init_obj, 0, Maix_mic_array_init);

STATIC mp_obj_t Maix_mic_array_deinit(void) { return mp_const_true; }
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_deinit_obj, Maix_mic_array_deinit);

/* ============================================================================ */
/* 极速获取单通道 PCM 音频 (完美对齐 24 位精度与防量化噪声截断) */
/* ============================================================================ */
STATIC mp_obj_t Maix_mic_array_get_raw_1ch(void) {
    volatile uint8_t retry = 100;
    static float prev_x = 0.0f;
    static float prev_y = 0.0f;
    static int16_t pcm_out[FFT_N]; // 单通道的 1024 个点
    int i;
    float x, y, amplified;
    
    while(rx_flag == 0) { retry--; msleep(1); if(retry == 0) break; }
    if(rx_flag == 0 && retry == 0) { mp_raise_OSError(MP_ETIMEDOUT); return mp_const_false; }
    rx_flag = 0;

    for (i = 0; i < FFT_N; i++) {
        /* ★ 关键修复 1：>> 8 提取 32 位槽内高 24 位有效数据，保留极限微音精度！
           注: [i*8 + 0] 代表提取第 0 个麦克风。如果想提取中心的第 6 个麦克风，可改为 [i*8 + 6] */
        x = (float)(i2s_rx_buf[i*8 + 0] >> 8);
        
        /* ★ 关键修复 2：IIR 滤波器消除直流偏置 (DC Blocker) */
        y = x - prev_x + 0.995f * prev_y;
        prev_x = x;
        prev_y = y;

        /* * ★ 数据重映射与放大防抖：
         * y 现在的幅度是 24-bit 级 (-8388608 ~ 8388607)。
         * 要塞进标准 16-bit 的 pcm_out 中 (-32768 ~ 32767)，理论上需要除以 256.0f。
         * 为了保留安静环境的极弱信号，让它连续平滑而非变成方波杂音，
         * 我们让它除以 16.0f，这等效于 (y / 256.0f) * 16.0f，即自带 16 倍的高保真数字增益。
         */
        amplified = y / 16.0f; 
        
        /* 强音硬限幅，保证不溢出爆音 */
        if(amplified > 32767.0f) amplified = 32767.0f;
        if(amplified < -32768.0f) amplified = -32768.0f;
        
        pcm_out[i] = (int16_t)amplified;
    }

    /* 立即重启 DMA */
    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);

    /* 抛出一个 2KB (1024 * 2 byte) 的干净单通道字节流给 Python 层 */
    return mp_obj_new_bytes((const byte*)pcm_out, FFT_N * sizeof(int16_t));
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_raw_1ch_obj, Maix_mic_array_get_raw_1ch);

STATIC const mp_rom_map_elem_t Maix_mic_array_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_init), MP_ROM_PTR(&Maix_mic_array_init_obj) },
    { MP_ROM_QSTR(MP_QSTR_deinit), MP_ROM_PTR(&Maix_mic_array_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_raw_1ch), MP_ROM_PTR(&Maix_mic_array_get_raw_1ch_obj) }, // 接口名已改
};
STATIC MP_DEFINE_CONST_DICT(Maix_mic_array_dict, Maix_mic_array_locals_dict_table);

const mp_obj_type_t Maix_mic_array_type = {
    { &mp_type_type }, .name = MP_QSTR_MIC_ARRAY, .locals_dict = (mp_obj_dict_t*)&Maix_mic_array_dict,
};
