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
// 第一部分：全局变量与双核异步共享内存 (7 阵元与 21 对基线)
// ============================================================================
#define FFT_N 1024
#define NUM_MICS 7       // 7 颗麦克风
#define NUM_PAIRS 21     // 21 对互相关
#define SAMPLE_RATE 48000.0f
#define SOUND_SPEED 343.0f
#undef PI
#define PI 3.14159265358979323846f
#define DEG2RAD (PI / 180.0f)

typedef struct { int u; int v; float dist; float dx; float dy; } MicPairConf;
MicPairConf pair_conf[NUM_PAIRS];

STATIC volatile uint8_t rx_flag = 0;
int32_t i2s_rx_buf[FFT_N * 8] __attribute__((aligned(128)));

// 双核通信桥梁 (严格保持不变)
static volatile float current_target_angle = 0.0f;     
static float shared_doa_mic_data[NUM_MICS][FFT_N];     
static volatile int shared_data_ready = 0;             

// ============================================================================
// 第二部分：纯正的 Broadband MUSIC 数学管线 (自带 7 帧滑动缓存)
// ============================================================================
static void init_array_geometry(void) {
    float R = 0.04f;
    float theta_mic[6] = {0, 60, 120, 180, 240, 300};
    float mic_pos_2d[NUM_MICS][2];
    for(int i = 0; i < 6; i++) {
        mic_pos_2d[i][0] = R * cosf(theta_mic[i] * DEG2RAD);
        mic_pos_2d[i][1] = R * sinf(theta_mic[i] * DEG2RAD);
    }
    // 中心麦克风
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

// --- MUSIC 算法专属静态内存池 (移出栈空间防止内核崩溃) ---
#define N_FRAMES 7
#define MAX_BINS 200
static float X_history_r[N_FRAMES][NUM_MICS][MAX_BINS];
static float X_history_i[N_FRAMES][NUM_MICS][MAX_BINS];
static int frame_idx = 0;
static int frames_collected = 0;
static float last_angle = 0.0f;

// 【数学内核替换区】：宽带 MUSIC 特征分解定位算法
static float run_doa_pipeline(float mic_data[NUM_MICS][FFT_N]) {
    static float mic_real[NUM_MICS][FFT_N]; 
    static float mic_imag[NUM_MICS][FFT_N]; 
    
    // 1. 加窗与正向 FFT
    for (int m = 0; m < NUM_MICS; m++) {
        for (int i = 0; i < FFT_N; i++) {
            float window = 0.54f - 0.46f * cosf(2.0f * PI * i / (FFT_N - 1));
            mic_real[m][i] = mic_data[m][i] * window;
            mic_imag[m][i] = 0.0f;
        }
        fft_radix2(mic_real[m], mic_imag[m], FFT_N, 0);
    }

    // 麦克风物理坐标 (用于生成导向矢量 a)
    static float mic_pos_x[NUM_MICS], mic_pos_y[NUM_MICS];
    static int geom_init = 0;
    if (!geom_init) {
        for (int i = 0; i < 6; i++) {
            mic_pos_x[i] = 0.04f * cosf(i * 60.0f * DEG2RAD);
            mic_pos_y[i] = 0.04f * sinf(i * 60.0f * DEG2RAD);
        }
        mic_pos_x[6] = 0.0f; mic_pos_y[6] = 0.0f;
        geom_init = 1;
    }

    float df = SAMPLE_RATE / FFT_N;
    int start_bin = (int)ceilf(50.0f / df);
    int end_bin = (int)floorf(8000.0f / df);
    if(end_bin >= MAX_BINS) end_bin = MAX_BINS - 1;

    // 将新一帧数据推入滑动缓存区
    for (int m = 0; m < NUM_MICS; m++) {
        for (int bin = start_bin; bin <= end_bin; bin++) {
            X_history_r[frame_idx][m][bin] = mic_real[m][bin];
            X_history_i[frame_idx][m][bin] = mic_imag[m][bin];
        }
    }
    frame_idx = (frame_idx + 1) % N_FRAMES;
    
    // 如果缓存还没填满 7 帧，无法构建满秩协方差矩阵，先返回上次角度
    if (frames_collected < N_FRAMES) {
        frames_collected++;
        return last_angle;
    }

    static float P_MUSIC[360] = {0};
    for (int i = 0; i < 360; i++) P_MUSIC[i] = 0.0f;

    // 2. 遍历每一个有效频点，执行标准的 MUSIC 管线
    for (int bin = start_bin; bin <= end_bin; bin++) {
        float freq = bin * df;
        float omega = 2.0f * PI * freq;

        // --- A. 建立 7x7 协方差矩阵 Rxx ---
        float Rxx_r[NUM_MICS][NUM_MICS] = {0};
        float Rxx_i[NUM_MICS][NUM_MICS] = {0};

        for (int f = 0; f < N_FRAMES; f++) {
            for (int i = 0; i < NUM_MICS; i++) {
                float xr = X_history_r[f][i][bin];
                float xi = X_history_i[f][i][bin];
                for (int j = 0; j < NUM_MICS; j++) {
                    float yr = X_history_r[f][j][bin];
                    float yi = X_history_i[f][j][bin];
                    Rxx_r[i][j] += (xr * yr + xi * yi);
                    Rxx_i[i][j] += (xi * yr - xr * yi);
                }
            }
        }
        for (int i = 0; i < NUM_MICS; i++) {
            for (int j = 0; j < NUM_MICS; j++) {
                Rxx_r[i][j] /= N_FRAMES;
                Rxx_i[i][j] /= N_FRAMES;
            }
        }

        // --- B. 幂迭代法求解主特征向量 (Signal Subspace V) ---
        float V_r[NUM_MICS] = {1, 1, 1, 1, 1, 1, 1};
        float V_i[NUM_MICS] = {0, 0, 0, 0, 0, 0, 0};
        for (int iter = 0; iter < 20; iter++) {
            float V_new_r[NUM_MICS] = {0};
            float V_new_i[NUM_MICS] = {0};
            float norm_sq = 0.0f;
            for (int i = 0; i < NUM_MICS; i++) {
                for (int j = 0; j < NUM_MICS; j++) {
                    V_new_r[i] += (Rxx_r[i][j] * V_r[j] - Rxx_i[i][j] * V_i[j]);
                    V_new_i[i] += (Rxx_r[i][j] * V_i[j] + Rxx_i[i][j] * V_r[j]);
                }
                norm_sq += (V_new_r[i] * V_new_r[i] + V_new_i[i] * V_new_i[i]);
            }
            float norm = sqrtf(norm_sq) + 1e-12f;
            for (int i = 0; i < NUM_MICS; i++) {
                V_r[i] = V_new_r[i] / norm;
                V_i[i] = V_new_i[i] / norm;
            }
        }

        // --- C. 构建噪声子空间投影矩阵 Pn = I - V*V^H ---
        float Pn_r[NUM_MICS][NUM_MICS];
        float Pn_i[NUM_MICS][NUM_MICS];
        for (int i = 0; i < NUM_MICS; i++) {
            for (int j = 0; j < NUM_MICS; j++) {
                float vvH_r = V_r[i] * V_r[j] + V_i[i] * V_i[j];
                float vvH_i = V_i[i] * V_r[j] - V_r[i] * V_i[j];
                if (i == j) {
                    Pn_r[i][j] = 1.0f - vvH_r;
                    Pn_i[i][j] = -vvH_i;
                } else {
                    Pn_r[i][j] = -vvH_r;
                    Pn_i[i][j] = -vvH_i;
                }
            }
        }

        // --- D. 全向空间谱扫描 P_music = 1 / (a^H * Pn * a) ---
        for (int theta = -180; theta < 180; theta++) {
            float cos_t = cosf(theta * DEG2RAD);
            float sin_t = sinf(theta * DEG2RAD);

            float a_r[NUM_MICS], a_i[NUM_MICS];
            for (int m = 0; m < NUM_MICS; m++) {
                float phase = omega * (mic_pos_x[m] * cos_t + mic_pos_y[m] * sin_t) / SOUND_SPEED;
                a_r[m] = cosf(phase);
                a_i[m] = -sinf(phase); 
            }

            float y_r[NUM_MICS] = {0}, y_i[NUM_MICS] = {0};
            for (int i = 0; i < NUM_MICS; i++) {
                for (int j = 0; j < NUM_MICS; j++) {
                    y_r[i] += (Pn_r[i][j] * a_r[j] - Pn_i[i][j] * a_i[j]);
                    y_i[i] += (Pn_r[i][j] * a_i[j] + Pn_i[i][j] * a_r[j]);
                }
            }

            float val_r = 0.0f;
            for (int i = 0; i < NUM_MICS; i++) {
                val_r += (a_r[i] * y_r[i] + a_i[i] * y_i[i]);
            }

            int theta_idx = theta + 180;
            P_MUSIC[theta_idx] += 1.0f / (val_r + 1e-12f);
        }
    }

    // 3. 在全频带叠加空间谱中寻找峰值
    float max_P = 0;
    float best_ang = 0;
    for (int i = 0; i < 360; i++) {
        if (P_MUSIC[i] > max_P) {
            max_P = P_MUSIC[i];
            best_ang = (float)(i - 180);
        }
    }

    last_angle = best_ang;
    float final_ang = fmodf(best_ang + 180.0f, 360.0f);
    if (final_ang < 0) final_ang += 360.0f;
    return final_ang - 180.0f;
}

// ============================================================================
// 第三部分：核心 API 与异步并行接口 (严密锁死不变)
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

// 录音兵接口：提取 7 颗用于测向，波束成型只使用前 6 颗！
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

// 雷达兵接口：GIL 解锁 + 运行 Broadband MUSIC
STATIC mp_obj_t Maix_mic_array_track_doa(void) {
    if (shared_data_ready == 0) return mp_obj_new_float(current_target_angle);

    static float local_mic_data[NUM_MICS][FFT_N];
    for(int m=0; m<NUM_MICS; m++) {
        for(int i=0; i<FFT_N; i++) local_mic_data[m][i] = shared_doa_mic_data[m][i];
    }
    shared_data_ready = 0; 

    // 解锁，让另一个核心安心录音
    MP_THREAD_GIL_EXIT(); 
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

STATIC mp_obj_t Maix_mic_array_get_map(void) { return mp_const_none; }
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_map_obj, Maix_mic_array_get_map);

STATIC const mp_rom_map_elem_t Maix_mic_array_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_init), MP_ROM_PTR(&Maix_mic_array_init_obj) },
    { MP_ROM_QSTR(MP_QSTR_deinit), MP_ROM_PTR(&Maix_mic_array_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_beam_audio), MP_ROM_PTR(&Maix_mic_array_get_beam_audio_obj) },
    { MP_ROM_QSTR(MP_QSTR_track_doa), MP_ROM_PTR(&Maix_mic_array_track_doa_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_led), MP_ROM_PTR(&Maix_mic_array_set_led_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_map), MP_ROM_PTR(&Maix_mic_array_get_map_obj) },
};
STATIC MP_DEFINE_CONST_DICT(Maix_mic_array_dict, Maix_mic_array_locals_dict_table);

const mp_obj_type_t Maix_mic_array_type = {
    { &mp_type_type }, .name = MP_QSTR_MIC_ARRAY, .locals_dict = (mp_obj_dict_t*)&Maix_mic_array_dict,
};
