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
#include "py/mpthread.h"  

#include "sleep.h"
#include "lcd.h"
#include "sysctl.h"
#include "fpioa.h"
#include "sipeed_sk9822.h"
#include "py_image.h"
#include "i2s.h"
#include "dmac.h"

#define PLL2_OUTPUT_FREQ 45158400UL

// ============================================================================
// 第一部分：全局变量与双核异步共享内存
// ============================================================================
#define FFT_N 1024
#define NUM_MICS 7       
#define NUM_PAIRS 21     
#define SAMPLE_RATE 16000.0f
#define SOUND_SPEED 343.0f
#undef PI
#define PI 3.14159265358979323846f
#define DEG2RAD (PI / 180.0f)

typedef struct { int u; int v; float dist; float dx; float dy; } MicPairConf;
MicPairConf pair_conf[NUM_PAIRS];

STATIC volatile uint8_t rx_flag = 0;
int32_t i2s_rx_buf[FFT_N * 8] __attribute__((aligned(128)));

static volatile float current_target_angle = 0.0f;     
static float shared_doa_mic_data[NUM_MICS][FFT_N];     
static volatile int shared_data_ready = 0;             

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
            pair_conf[k].u = i; pair_conf[k].v = j;
            pair_conf[k].dx = mic_pos_2d[j][0] - mic_pos_2d[i][0];
            pair_conf[k].dy = mic_pos_2d[j][1] - mic_pos_2d[i][1];
            pair_conf[k].dist = sqrtf(pair_conf[k].dx * pair_conf[k].dx + pair_conf[k].dy * pair_conf[k].dy);
            k++;
        }
    }
}

static void fft_radix2(float* real, float* imag, int n, int is_inverse) {
    int i, j, k, l; float tx, ty, u1, u2, z; j = 0;
    for (i = 0; i < n - 1; i++) {
        if (i < j) {
            tx = real[i]; real[i] = real[j]; real[j] = tx;
            ty = imag[i]; imag[i] = imag[j]; imag[j] = ty;
        }
        k = n / 2; while (k <= j) { j -= k; k /= 2; } j += k;
    }
    for (l = 1; l < n; l *= 2) {
        z = PI / l; if (is_inverse) z = -z;
        u1 = 1.0f; u2 = 0.0f; float cos_z = cosf(z), sin_z = sinf(z);
        for (j = 0; j < l; j++) {
            for (i = j; i < n; i += 2 * l) {
                int p = i + l;
                tx = real[p] * u1 - imag[p] * u2; ty = real[p] * u2 + imag[p] * u1;
                real[p] = real[i] - tx; imag[p] = imag[i] - ty;
                real[i] += tx; imag[i] += ty;
            }
            float t = u1; u1 = t * cos_z - u2 * sin_z; u2 = t * sin_z + u2 * cos_z;
        }
    }
    if (is_inverse) { for (i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; } }
}

static float calculate_median(float* array, int size) {
    float temp[NUM_PAIRS];
    for(int i = 0; i < size; i++) temp[i] = array[i];
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (temp[j] > temp[j + 1]) {
                float swap = temp[j]; temp[j] = temp[j + 1]; temp[j + 1] = swap;
            }
        }
    }
    return temp[size / 2];
}

// ============================================================================
// 第二部分：时空级联核心管线 (M-估计空间层 + 鲁棒卡尔曼时间层)
// ============================================================================
static float run_doa_pipeline(float mic_data[NUM_MICS][FFT_N]) {
    static float mic_real[NUM_MICS][FFT_N]; static float mic_imag[NUM_MICS][FFT_N]; 
    for (int m = 0; m < NUM_MICS; m++) {
        for (int i = 0; i < FFT_N; i++) {
            float window = 0.54f - 0.46f * cosf(2.0f * PI * i / (FFT_N - 1));
            mic_real[m][i] = mic_data[m][i] * window; mic_imag[m][i] = 0.0f;
        }
        fft_radix2(mic_real[m], mic_imag[m], FFT_N, 0);
    }

    float Meas_TDOA[NUM_PAIRS] = {0}; float Qual_Score[NUM_PAIRS] = {0};
    static float R_real[FFT_N], R_imag[FFT_N];

    int k = 0;
    for (int u = 0; u < NUM_MICS; u++) {
        for (int v = u + 1; v < NUM_MICS; v++) {
            for (int i = 0; i < FFT_N; i++) {
                float cross_r = mic_real[u][i] * mic_real[v][i] + mic_imag[u][i] * mic_imag[v][i];
                float cross_i = mic_imag[u][i] * mic_real[v][i] - mic_real[u][i] * mic_imag[v][i];
                float mag = sqrtf(cross_r * cross_r + cross_i * cross_i) + 1e-9f;
                R_real[i] = cross_r / mag; R_imag[i] = cross_i / mag;
            }
            fft_radix2(R_real, R_imag, FFT_N, 1);
            float max_val = 0; int max_idx = 0; int search_range = 8; 
            for (int i = 0; i <= search_range; i++) { if (R_real[i] > max_val) { max_val = R_real[i]; max_idx = i; } }
            for (int i = FFT_N - search_range; i < FFT_N; i++) { if (R_real[i] > max_val) { max_val = R_real[i]; max_idx = i; } }
            float delta = 0;
            if (max_idx > 0 && max_idx < FFT_N - 1) {
                float y1 = R_real[max_idx - 1]; float y2 = R_real[max_idx]; float y3 = R_real[max_idx + 1];
                float denom = 2.0f * (y1 - 2.0f * y2 + y3);
                if (fabsf(denom) > 1e-9f) delta = (y1 - y3) / denom;
            }
            float tau_samples = (max_idx > FFT_N/2) ? (max_idx - FFT_N + delta) : (max_idx + delta);
            Meas_TDOA[k] = tau_samples / SAMPLE_RATE;
            Qual_Score[k] = 1.0f / (1.0f + expf(-15.0f * (max_val - 0.15f)));
            k++;
        }
    }

    float min_cost = FLT_MAX; float ang_coarse = 0.0f;
    for (int theta = -180; theta < 180; theta++) {
        float cost = 0.0f; float cos_t = cosf(theta * DEG2RAD); float sin_t = sinf(theta * DEG2RAD);
        for (int p = 0; p < NUM_PAIRS; p++) {
            if (Qual_Score[p] < 1e-3f) continue;
            float theo_tdoa = (pair_conf[p].dx * cos_t + pair_conf[p].dy * sin_t) / SOUND_SPEED;
            float err = theo_tdoa - Meas_TDOA[p];
            cost += Qual_Score[p] * pair_conf[p].dist * err * err;
        }
        if (cost < min_cost) { min_cost = cost; ang_coarse = (float)theta; }
    }

    float raw_residuals_t[NUM_PAIRS]; float cos_coarse = cosf(ang_coarse * DEG2RAD); float sin_coarse = sinf(ang_coarse * DEG2RAD);
    for (int p = 0; p < NUM_PAIRS; p++) {
        float theo_tdoa = (pair_conf[p].dx * cos_coarse + pair_conf[p].dy * sin_coarse) / SOUND_SPEED;
        raw_residuals_t[p] = fabsf(Meas_TDOA[p] - theo_tdoa);
    }
    float med_res_m = calculate_median(raw_residuals_t, NUM_PAIRS) * SOUND_SPEED;
    float sigma_adaptive_m = med_res_m * 1.5f;
    if (sigma_adaptive_m < 0.015f) sigma_adaptive_m = 0.015f;
    if (sigma_adaptive_m > 0.10f) sigma_adaptive_m = 0.10f;
    float sigma_adaptive_t = sigma_adaptive_m / SOUND_SPEED;

    float Final_W[NUM_PAIRS];
    for (int p = 0; p < NUM_PAIRS; p++) {
        float res_sq = raw_residuals_t[p] * raw_residuals_t[p];
        float w_consist = expf(-res_sq / (2.0f * sigma_adaptive_t * sigma_adaptive_t));
        Final_W[p] = Qual_Score[p] * ((1.0f - 0.2f) * w_consist + 0.2f) * sqrtf(pair_conf[p].dist);
    }

    float ang_gn = ang_coarse;
    for (int iter = 0; iter < 8; iter++) {
        float sum_WJr = 0.0f, sum_WJJ = 0.0f; int valid_count = 0;
        float cos_gn = cosf(ang_gn * DEG2RAD), sin_gn = sinf(ang_gn * DEG2RAD);
        for (int p = 0; p < NUM_PAIRS; p++) {
            if (Final_W[p] < 1e-4f) continue;
            valid_count++;
            float theo_tdoa = (pair_conf[p].dx * cos_gn + pair_conf[p].dy * sin_gn) / SOUND_SPEED;
            float r_p = theo_tdoa - Meas_TDOA[p];
            float J_p = DEG2RAD * (-pair_conf[p].dx * sin_gn + pair_conf[p].dy * cos_gn) / SOUND_SPEED;
            sum_WJr += Final_W[p] * J_p * r_p; sum_WJJ += Final_W[p] * J_p * J_p;
        }
        if (valid_count < 3) break;
        float delta_ang = -sum_WJr / (sum_WJJ + 1e-12f);
        ang_gn += delta_ang; if (fabsf(delta_ang) < 1e-3f) break;
    }
    
    // 获取底层测量出的瞬态角度 (带跳点噪声)
    float raw_angle = fmodf(ang_gn + 180.0f, 360.0f);
    if (raw_angle < 0) raw_angle += 360.0f;
    raw_angle -= 180.0f;

    // ============================================================================
    // 【核心注入】：鲁棒 Alpha-Beta 运动学滤波器 (基于 Huber M-估计)
    // ============================================================================
    static int tracker_initialized = 0;
    static float est_angle = 0.0f;
    static float est_velocity = 0.0f;
    static uint32_t last_time_ms = 0;
    
    // 卡尔曼增益参数设置
    const float ALPHA = 0.3f;           // 位置信任度
    const float BETA  = 0.02f;          // 速度信任度
    const float HUBER_THRESH = 15.0f;   // 拒绝瞬移的 M-估计容忍度(度)

    uint32_t current_time_ms = mp_hal_ticks_ms();

    if (!tracker_initialized) {
        est_angle = raw_angle;
        est_velocity = 0.0f;
        last_time_ms = current_time_ms;
        tracker_initialized = 1;
        return est_angle;
    }

    float dt = (current_time_ms - last_time_ms) / 1000.0f;
    if (dt <= 0.001f) dt = 0.01f;
    last_time_ms = current_time_ms;

    // 1. 状态预测
    float pred_angle = est_angle + est_velocity * dt;

    // 2. 残差计算与环形折叠
    float raw_residual = raw_angle - pred_angle;
    while (raw_residual > 180.0f)  raw_residual -= 360.0f;
    while (raw_residual < -180.0f) raw_residual += 360.0f;

    // 3. Huber M-估计权重分配
    float abs_res = fabsf(raw_residual);
    float robust_weight = 1.0f;
    if (abs_res > HUBER_THRESH) {
        robust_weight = HUBER_THRESH / abs_res; // 误差越大，权重被压得越低
    }
    float robust_residual = raw_residual * robust_weight;

    // 4. 状态更新
    est_angle = pred_angle + ALPHA * robust_residual;
    est_velocity = est_velocity + BETA * (robust_residual / dt);

    // 环形规范化
    while (est_angle > 180.0f)  est_angle -= 360.0f;
    while (est_angle < -180.0f) est_angle += 360.0f;

    return est_angle;
}

// ============================================================================
// 第三部分：核心 API 与异步并行接口
// ============================================================================
static int i2s_dma_cb(void *ctx) { rx_flag = 1; return 0; }

STATIC mp_obj_t Maix_mic_array_init(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    enum { ARG_i2s_d0, ARG_i2s_d1, ARG_i2s_d2, ARG_i2s_d3, ARG_i2s_ws, ARG_i2s_sclk, ARG_sk9822_dat, ARG_sk9822_clk, };
    static const mp_arg_t allowed_args[]={{MP_QSTR_i2s_d0, MP_ARG_INT, {.u_int = 23}}, {MP_QSTR_i2s_d1, MP_ARG_INT, {.u_int = 22}}, {MP_QSTR_i2s_d2, MP_ARG_INT, {.u_int = 21}}, {MP_QSTR_i2s_d3, MP_ARG_INT, {.u_int = 20}}, {MP_QSTR_i2s_ws, MP_ARG_INT, {.u_int = 19}}, {MP_QSTR_i2s_sclk, MP_ARG_INT, {.u_int = 18}}, {MP_QSTR_sk9822_dat, MP_ARG_INT, {.u_int = 24}}, {MP_QSTR_sk9822_clk, MP_ARG_INT, {.u_int = 25}},};
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)]; mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    fpioa_set_function(args[ARG_i2s_d0].u_int, FUNC_I2S0_IN_D0); fpioa_set_function(args[ARG_i2s_d1].u_int, FUNC_I2S0_IN_D1); fpioa_set_function(args[ARG_i2s_d2].u_int, FUNC_I2S0_IN_D2); fpioa_set_function(args[ARG_i2s_d3].u_int, FUNC_I2S0_IN_D3); fpioa_set_function(args[ARG_i2s_ws].u_int, FUNC_I2S0_WS); fpioa_set_function(args[ARG_i2s_sclk].u_int, FUNC_I2S0_SCLK); fpioa_set_function(args[ARG_sk9822_dat].u_int, FUNC_GPIOHS0 + SK9822_DAT_GPIONUM); fpioa_set_function(args[ARG_sk9822_clk].u_int, FUNC_GPIOHS0 + SK9822_CLK_GPIONUM);
    sipeed_init_mic_array_led();
    sysctl_pll_set_freq(SYSCTL_PLL2, PLL2_OUTPUT_FREQ); sysctl_clock_enable(SYSCTL_CLOCK_I2S0);
    i2s_init(I2S_DEVICE_0, I2S_RECEIVER, 0x0F);
    for(int i = 0; i < 4; i++) i2s_rx_channel_config(I2S_DEVICE_0, I2S_CHANNEL_0 + i, RESOLUTION_32_BIT, SCLK_CYCLES_32, TRIGGER_LEVEL_4, STANDARD_MODE);
    i2s_set_sample_rate(I2S_DEVICE_0, SAMPLE_RATE);
    init_array_geometry();
    dmac_set_irq(DMAC_CHANNEL4, i2s_dma_cb, NULL, 3);
    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);
    return mp_const_true;
}
MP_DEFINE_CONST_FUN_OBJ_KW(Maix_mic_array_init_obj, 0, Maix_mic_array_init);

STATIC mp_obj_t Maix_mic_array_deinit(void) { return mp_const_true; }
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_deinit_obj, Maix_mic_array_deinit);

// 波束成型现在提取的 target_angle 是经过卡尔曼滤波的极度平滑的值！
STATIC mp_obj_t Maix_mic_array_get_beam_audio(void) {
    volatile uint8_t retry = 100;
    while(rx_flag == 0) { retry--; msleep(1); if(retry == 0) break; }
    if(rx_flag == 0 && retry == 0) { mp_raise_OSError(MP_ETIMEDOUT); return mp_const_false; }
    rx_flag = 0;

    static float mic_raw[NUM_MICS][FFT_N];
    for (int i = 0; i < FFT_N; i++) {
        mic_raw[0][i] = (float)(i2s_rx_buf[i*8 + 0] >> 16);
        mic_raw[1][i] = (float)(i2s_rx_buf[i*8 + 1] >> 16);
        mic_raw[2][i] = (float)(i2s_rx_buf[i*8 + 2] >> 16);
        mic_raw[3][i] = (float)(i2s_rx_buf[i*8 + 3] >> 16);
        mic_raw[4][i] = (float)(i2s_rx_buf[i*8 + 4] >> 16);
        mic_raw[5][i] = (float)(i2s_rx_buf[i*8 + 5] >> 16);
        mic_raw[6][i] = (float)(i2s_rx_buf[i*8 + 6] >> 16); 
    }

    if (shared_data_ready == 0) {
        for(int m=0; m<NUM_MICS; m++) {
            for(int i=0; i<FFT_N; i++) shared_doa_mic_data[m][i] = mic_raw[m][i];
        }
        shared_data_ready = 1; 
    }

    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);

    float target_angle = current_target_angle;
    float cos_t = cosf(target_angle * DEG2RAD);
    float sin_t = sinf(target_angle * DEG2RAD);
    
    static int shift_samples[6];
    for (int m = 0; m < 6; m++) {
        float mx = 0.04f * cosf(m * 60.0f * DEG2RAD);
        float my = 0.04f * sinf(m * 60.0f * DEG2RAD);
        float tau = (mx * cos_t + my * sin_t) / SOUND_SPEED;
        shift_samples[m] = (int)roundf(tau * SAMPLE_RATE);
    }

    static float prev_x = 0.0f;
    static float prev_y = 0.0f;
    static int16_t pcm_out[FFT_N];

    for (int i = 0; i < FFT_N; i++) {
        float sum = 0.0f;
        for (int m = 0; m < 6; m++) {
            int target_idx = i - shift_samples[m];
            if (target_idx >= 0 && target_idx < FFT_N) sum += mic_raw[m][target_idx];
        }
        sum /= 6.0f; 

        float y = sum - prev_x + 0.995f * prev_y;
        prev_x = sum; prev_y = y;

        float amplified = y * 4.0f; 
        if(amplified > 32767.0f) amplified = 32767.0f;
        if(amplified < -32768.0f) amplified = -32768.0f;
        pcm_out[i] = (int16_t)amplified;
    }

    mp_obj_t tuple[2];
    tuple[0] = mp_obj_new_float(target_angle);
    tuple[1] = mp_obj_new_bytes((const byte*)pcm_out, FFT_N * sizeof(int16_t));
    return mp_obj_new_tuple(2, tuple);
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_beam_audio_obj, Maix_mic_array_get_beam_audio);

STATIC mp_obj_t Maix_mic_array_track_doa(void) {
    if (shared_data_ready == 0) return mp_obj_new_float(current_target_angle);

    static float local_mic_data[NUM_MICS][FFT_N];
    for(int m=0; m<NUM_MICS; m++) {
        for(int i=0; i<FFT_N; i++) local_mic_data[m][i] = shared_doa_mic_data[m][i];
    }
    shared_data_ready = 0; 

    MP_THREAD_GIL_EXIT(); 
    // 这个 new_angle 已经是经过双重过滤的完美角度！
    float new_angle = run_doa_pipeline(local_mic_data);
    MP_THREAD_GIL_ENTER();
    
    current_target_angle = new_angle;
    return mp_obj_new_float(new_angle);
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_track_doa_obj, Maix_mic_array_track_doa);

STATIC mp_obj_t Maix_mic_array_set_led(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    int index, brightness[12] = {0}, led_color[12] = {0}, color[3] = {0}; mp_obj_t *items;
    mp_obj_get_array_fixed_n(pos_args[0], 12, &items);
    for(index= 0; index < 12; index++) brightness[index] = mp_obj_get_int(items[index]);
    mp_obj_get_array_fixed_n(pos_args[1], 3, &items);
    for(index = 0; index < 3; index++) color[index] = mp_obj_get_int(items[index]);
    uint32_t set_color = (color[2] << 16) | (color[1] << 8) | (color[0]);
    for (index = 0; index < 12; index++) led_color[index] = (brightness[index] / 2) > 1 ? (((0xe0 | (brightness[index] * 2)) << 24) | set_color) : 0xe0000000;
    sysctl_disable_irq(); sk9822_start_frame();
    for (index = 0; index < 12; index++) sk9822_send_data(led_color[index]);
    sk9822_stop_frame(); sysctl_enable_irq();
    return mp_const_true;
}
MP_DEFINE_CONST_FUN_OBJ_KW(Maix_mic_array_set_led_obj, 2, Maix_mic_array_set_led);

STATIC const mp_rom_map_elem_t Maix_mic_array_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_init), MP_ROM_PTR(&Maix_mic_array_init_obj) },
    { MP_ROM_QSTR(MP_QSTR_deinit), MP_ROM_PTR(&Maix_mic_array_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_beam_audio), MP_ROM_PTR(&Maix_mic_array_get_beam_audio_obj) },
    { MP_ROM_QSTR(MP_QSTR_track_doa), MP_ROM_PTR(&Maix_mic_array_track_doa_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_led), MP_ROM_PTR(&Maix_mic_array_set_led_obj) },
};
STATIC MP_DEFINE_CONST_DICT(Maix_mic_array_dict, Maix_mic_array_locals_dict_table);

const mp_obj_type_t Maix_mic_array_type = {
    { &mp_type_type }, .name = MP_QSTR_MIC_ARRAY, .locals_dict = (mp_obj_dict_t*)&Maix_mic_array_dict,
};
