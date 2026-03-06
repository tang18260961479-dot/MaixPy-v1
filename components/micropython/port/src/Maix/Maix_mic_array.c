#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

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

// --- 底层硬件驱动 ---
#include "i2s.h"
#include "dmac.h"

#define PLL2_OUTPUT_FREQ 45158400UL

// ============================================================================
// 第一部分：核心宏定义、结构体与全局变量 (严格保持一致)
// ============================================================================

#define FFT_N 1024
#define NUM_MICS 7
#define NUM_PAIRS 21
#define SAMPLE_RATE 48000.0f
#define SOUND_SPEED 343.0f
#undef PI
#define PI 3.14159265358979323846f
#define DEG2RAD (PI / 180.0f)

typedef struct { 
    int u; 
    int v; 
    float dist; 
    float dx; 
    float dy; 
} MicPairConf;

MicPairConf pair_conf[NUM_PAIRS];

// DMA 接收专用内存池与标志位
STATIC volatile uint8_t rx_flag = 0;
int32_t i2s_rx_buf[FFT_N * 8] __attribute__((aligned(128)));
float mic_raw_float[NUM_MICS][FFT_N];

// ============================================================================
// 第二部分：纯 C 语言硬核数学与信号处理管线
// ============================================================================

static void init_array_geometry(void) {
    float R = 0.04f;
    float theta_mic[6] = {0, 60, 120, 180, 240, 300};
    float mic_pos_2d[NUM_MICS][2];
    for(int i = 0; i < 6; i++) {
        mic_pos_2d[i][0] = R * cosf(theta_mic[i] * DEG2RAD);
        mic_pos_2d[i][1] = R * sinf(theta_mic[i] * DEG2RAD);
    }
    mic_pos_2d[6][0] = 0.0f; 
    mic_pos_2d[6][1] = 0.0f;

    int k = 0;
    for (int i = 0; i < NUM_MICS; i++) {
        for (int j = i + 1; j < NUM_MICS; j++) {
            pair_conf[k].u = i; 
            pair_conf[k].v = j;
            pair_conf[k].dx = mic_pos_2d[j][0] - mic_pos_2d[i][0];
            pair_conf[k].dy = mic_pos_2d[j][1] - mic_pos_2d[i][1];
            pair_conf[k].dist = sqrtf(pair_conf[k].dx * pair_conf[k].dx + pair_conf[k].dy * pair_conf[k].dy);
            k++;
        }
    }
}

static void fft_radix2(float* real, float* imag, int n, int is_inverse) {
    int i, j, k, l;
    float tx, ty, u1, u2, z;
    j = 0;
    for (i = 0; i < n - 1; i++) {
        if (i < j) {
            tx = real[i]; real[i] = real[j]; real[j] = tx;
            ty = imag[i]; imag[i] = imag[j]; imag[j] = ty;
        }
        k = n / 2;
        while (k <= j) { j -= k; k /= 2; }
        j += k;
    }
    for (l = 1; l < n; l *= 2) {
        z = PI / l;
        if (is_inverse) z = -z;
        u1 = 1.0f; u2 = 0.0f;
        float cos_z = cosf(z), sin_z = sinf(z);
        for (j = 0; j < l; j++) {
            for (i = j; i < n; i += 2 * l) {
                int p = i + l;
                tx = real[p] * u1 - imag[p] * u2;
                ty = real[p] * u2 + imag[p] * u1;
                real[p] = real[i] - tx; 
                imag[p] = imag[i] - ty;
                real[i] += tx;          
                imag[i] += ty;
            }
            float t = u1;
            u1 = t * cos_z - u2 * sin_z;
            u2 = t * sin_z + u2 * cos_z;
        }
    }
    if (is_inverse) {
        for (i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; }
    }
}

// ============================================================================
// 算法 2：Baseline - SRP-PHAT (修复底层 FFT 共轭反向问题)
// ============================================================================
static float run_doa_pipeline(float mic_data[NUM_MICS][FFT_N]) {
    static float mic_real[NUM_MICS][FFT_N]; 
    static float mic_imag[NUM_MICS][FFT_N]; 
    
    for (int m = 0; m < NUM_MICS; m++) {
        for (int i = 0; i < FFT_N; i++) {
            float window = 0.54f - 0.46f * cosf(2.0f * PI * i / (FFT_N - 1));
            mic_real[m][i] = mic_data[m][i] * window;
            mic_imag[m][i] = 0.0f;
        }
        fft_radix2(mic_real[m], mic_imag[m], FFT_N, 0);
    }

    float df = SAMPLE_RATE / FFT_N;  
    int start_bin = (int)ceilf(50.0f / df);    
    int end_bin = (int)floorf(8000.0f / df);   

    static float R_phat_r[NUM_PAIRS][FFT_N / 2];
    static float R_phat_i[NUM_PAIRS][FFT_N / 2];

    int k = 0;
    for (int u = 0; u < NUM_MICS; u++) {
        for (int v = u + 1; v < NUM_MICS; v++) {
            for (int bin = start_bin; bin <= end_bin; bin++) {
                // 注意：这里的 cross_i 取出来后，符号实际上是 MATLAB 的相反数
                float cross_r = mic_real[u][bin] * mic_real[v][bin] + mic_imag[u][bin] * mic_imag[v][bin];
                float cross_i = mic_imag[u][bin] * mic_real[v][bin] - mic_real[u][bin] * mic_imag[v][bin];
                
                float mag = sqrtf(cross_r * cross_r + cross_i * cross_i) + 1e-9f;
                R_phat_r[k][bin] = cross_r / mag;
                R_phat_i[k][bin] = cross_i / mag;
            }
            k++;
        }
    }

    float max_srp_energy = -FLT_MAX;
    float best_ang = 0.0f;

    for (int theta = -180; theta < 180; theta++) {
        float e_sum = 0.0f;
        float cos_t = cosf(theta * DEG2RAD);
        float sin_t = sinf(theta * DEG2RAD);

        for (int p = 0; p < NUM_PAIRS; p++) {
            float theo_tdoa = (pair_conf[p].dx * cos_t + pair_conf[p].dy * sin_t) / SOUND_SPEED;
            
            for (int bin = start_bin; bin <= end_bin; bin++) {
                float freq = bin * df;
                float omega = 2.0f * PI * freq;
                
                float phase = omega * theo_tdoa;
                float steer_r = cosf(phase);
                float steer_i = sinf(phase);
                
                // -----------------------------------------------------------------
                // 【核心修复】: 
                // 标准公式应为 R_r * steer_r - R_i * steer_i。
                // 因为底层 FFT 导致 R_phat_i 的符号被反转，这里必须改为加号 (+)，
                // 从而抵消共轭误差，与 MATLAB 的 real(R .* exp(j*omega*tau)) 绝对对齐！
                // -----------------------------------------------------------------
                e_sum += (R_phat_r[p][bin] * steer_r + R_phat_i[p][bin] * steer_i);
            }
        }

        if (e_sum > max_srp_energy) {
            max_srp_energy = e_sum;
            best_ang = (float)theta;
        }
    }

    float final_ang = fmodf(best_ang + 180.0f, 360.0f);
    if (final_ang < 0) final_ang += 360.0f;
    return final_ang - 180.0f;
}
// ============================================================================
// 第三部分：MicroPython 底层接口绑定 (与系统保持一致)
// ============================================================================

static int i2s_dma_cb(void *ctx) {
    rx_flag = 1;
    return 0;
}

STATIC mp_obj_t Maix_mic_array_init(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args)
{
    enum {
        ARG_i2s_d0, ARG_i2s_d1, ARG_i2s_d2, ARG_i2s_d3, ARG_i2s_ws, ARG_i2s_sclk, ARG_sk9822_dat, ARG_sk9822_clk,
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

    fpioa_set_function(args[ARG_i2s_d0].u_int, FUNC_I2S0_IN_D0);
    fpioa_set_function(args[ARG_i2s_d1].u_int, FUNC_I2S0_IN_D1);
    fpioa_set_function(args[ARG_i2s_d2].u_int, FUNC_I2S0_IN_D2);
    fpioa_set_function(args[ARG_i2s_d3].u_int, FUNC_I2S0_IN_D3);
    fpioa_set_function(args[ARG_i2s_ws].u_int, FUNC_I2S0_WS);
    fpioa_set_function(args[ARG_i2s_sclk].u_int, FUNC_I2S0_SCLK);
    fpioa_set_function(args[ARG_sk9822_dat].u_int, FUNC_GPIOHS0 + SK9822_DAT_GPIONUM);
    fpioa_set_function(args[ARG_sk9822_clk].u_int, FUNC_GPIOHS0 + SK9822_CLK_GPIONUM);

    sipeed_init_mic_array_led();

    sysctl_pll_set_freq(SYSCTL_PLL2, PLL2_OUTPUT_FREQ); 
    sysctl_clock_enable(SYSCTL_CLOCK_I2S0);

    i2s_init(I2S_DEVICE_0, I2S_RECEIVER, 0x0F);
    for(int i = 0; i < 4; i++){
        i2s_rx_channel_config(I2S_DEVICE_0, I2S_CHANNEL_0 + i,
                              RESOLUTION_32_BIT, SCLK_CYCLES_32,
                              TRIGGER_LEVEL_4, STANDARD_MODE);
    }
    i2s_set_sample_rate(I2S_DEVICE_0, SAMPLE_RATE);

    init_array_geometry();

    dmac_set_irq(DMAC_CHANNEL4, i2s_dma_cb, NULL, 3);
    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);
    
    return mp_const_true;
}
MP_DEFINE_CONST_FUN_OBJ_KW(Maix_mic_array_init_obj, 0, Maix_mic_array_init);

STATIC mp_obj_t Maix_mic_array_deinit(void) {
    return mp_const_true;
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_deinit_obj, Maix_mic_array_deinit);

STATIC mp_obj_t Maix_mic_array_get_map(void) {
    mp_raise_ValueError("Thermal map is disabled. Using high-precision DOA instead.");
    return mp_const_none;
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_map_obj, Maix_mic_array_get_map);

STATIC mp_obj_t Maix_mic_array_get_dir(void)
{
    volatile uint8_t retry = 100;
    while(rx_flag == 0) {
        retry--;
        msleep(1);
        if(retry == 0) break;
    }
    
    if(rx_flag == 0 && retry == 0) {
        mp_raise_OSError(MP_ETIMEDOUT);
        return mp_const_false;
    }
    rx_flag = 0;

    for (int i = 0; i < FFT_N; i++) {
        mic_raw_float[0][i] = (float)(i2s_rx_buf[i*8 + 0] >> 16);
        mic_raw_float[1][i] = (float)(i2s_rx_buf[i*8 + 1] >> 16);
        mic_raw_float[2][i] = (float)(i2s_rx_buf[i*8 + 2] >> 16);
        mic_raw_float[3][i] = (float)(i2s_rx_buf[i*8 + 3] >> 16);
        mic_raw_float[4][i] = (float)(i2s_rx_buf[i*8 + 4] >> 16);
        mic_raw_float[5][i] = (float)(i2s_rx_buf[i*8 + 5] >> 16);
        mic_raw_float[6][i] = (float)(i2s_rx_buf[i*8 + 6] >> 16);
    }

    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);

    float final_angle = run_doa_pipeline(mic_raw_float);

    return mp_obj_new_float(final_angle);
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_dir_obj, Maix_mic_array_get_dir);

STATIC mp_obj_t Maix_mic_array_set_led(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args)
{
    int index, brightness[12] = {0}, led_color[12] = {0}, color[3] = {0};
    mp_obj_t *items;
    
    mp_obj_get_array_fixed_n(pos_args[0], 12, &items);
    for(index= 0; index < 12; index++) brightness[index] = mp_obj_get_int(items[index]);

    mp_obj_get_array_fixed_n(pos_args[1], 3, &items);
    for(index = 0; index < 3; index++) color[index] = mp_obj_get_int(items[index]);

    uint32_t set_color = (color[2] << 16) | (color[1] << 8) | (color[0]);

    for (index = 0; index < 12; index++) {
        led_color[index] = (brightness[index] / 2) > 1 ? (((0xe0 | (brightness[index] * 2)) << 24) | set_color) : 0xe0000000;
    }

    sysctl_disable_irq();
    sk9822_start_frame();
    for (index = 0; index < 12; index++) sk9822_send_data(led_color[index]);
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
