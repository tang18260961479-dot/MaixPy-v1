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
#define NUM_MICS 6       // 只提取 6 颗外围纯净麦克风
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
    
    for(int i = 0; i < 4; i++) {
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

// ============================================================================
// 极速获取 6 通道交织 PCM 音频 (专为 MATLAB 离线分析打造)
// ============================================================================
STATIC mp_obj_t Maix_mic_array_get_raw_6ch(void) {
    volatile uint8_t retry = 100;
    while(rx_flag == 0) { retry--; msleep(1); if(retry == 0) break; }
    if(rx_flag == 0 && retry == 0) { mp_raise_OSError(MP_ETIMEDOUT); return mp_const_false; }
    rx_flag = 0;

    static float prev_x[NUM_MICS] = {0};
    static float prev_y[NUM_MICS] = {0};
    
    // 生成 6 通道交织数据：[m0, m1, m2, m3, m4, m5, m0, m1...]
    static int16_t pcm_out[FFT_N * NUM_MICS];

    for (int i = 0; i < FFT_N; i++) {
        for (int m = 0; m < NUM_MICS; m++) {
            // 提取高 16 位有效数据
            float x = (float)(i2s_rx_buf[i*8 + m] >> 16);
            
            // 独立的 IIR 滤波器消除各通道直流偏置
            float y = x - prev_x[m] + 0.995f * prev_y[m];
            prev_x[m] = x;
            prev_y[m] = y;

            float amplified = y * 4.0f; // 数字增益
            if(amplified > 32767.0f) amplified = 32767.0f;
            if(amplified < -32768.0f) amplified = -32768.0f;
            
            // 写入交织数组
            pcm_out[i * NUM_MICS + m] = (int16_t)amplified;
        }
    }

    // 立即重启 DMA
    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);

    // 返回一整块 12KB 的字节流给 Python
    return mp_obj_new_bytes((const byte*)pcm_out, FFT_N * NUM_MICS * sizeof(int16_t));
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_raw_6ch_obj, Maix_mic_array_get_raw_6ch);

STATIC const mp_rom_map_elem_t Maix_mic_array_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_init), MP_ROM_PTR(&Maix_mic_array_init_obj) },
    { MP_ROM_QSTR(MP_QSTR_deinit), MP_ROM_PTR(&Maix_mic_array_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_raw_6ch), MP_ROM_PTR(&Maix_mic_array_get_raw_6ch_obj) },
};
STATIC MP_DEFINE_CONST_DICT(Maix_mic_array_dict, Maix_mic_array_locals_dict_table);

const mp_obj_type_t Maix_mic_array_type = {
    { &mp_type_type }, .name = MP_QSTR_MIC_ARRAY, .locals_dict = (mp_obj_dict_t*)&Maix_mic_array_dict,
};
