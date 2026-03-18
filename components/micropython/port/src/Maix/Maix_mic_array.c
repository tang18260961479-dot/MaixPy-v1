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

#define FFT_N 1024
#define NUM_MICS 7
#define NUM_PAIRS 21
#define SAMPLE_RATE 48000.0f
#define SOUND_SPEED 343.0f
#undef PI
#define PI 3.14159265358979323846f
#define DEG2RAD (PI / 180.0f)

typedef struct { 
    int u; int v; 
    float dist; float dx; float dy; 
} MicPairConf;

MicPairConf pair_conf[NUM_PAIRS];
float mic_pos_2d[NUM_MICS][2]; // 【公平性对齐】：全局化物理坐标，保证所有算法共享统一基准

STATIC volatile uint8_t rx_flag = 0;
int32_t i2s_rx_buf[FFT_N * 8] __attribute__((aligned(128)));
float mic_raw_float[NUM_MICS][FFT_N];

static void init_array_geometry(void) {
    float R = 0.04f;
    float theta_mic[6] = {0, 60, 120, 180, 240, 300};
    int i, j, k;

    for(i = 0; i < 6; i++) {
        mic_pos_2d[i][0] = R * cosf(theta_mic[i] * DEG2RAD);
        mic_pos_2d[i][1] = R * sinf(theta_mic[i] * DEG2RAD);
    }
    mic_pos_2d[6][0] = 0.0f; mic_pos_2d[6][1] = 0.0f;

    k = 0;
    for (i = 0; i < NUM_MICS; i++) {
        for (j = i + 1; j < NUM_MICS; j++) {
            pair_conf[k].u = i; pair_conf[k].v = j;
            pair_conf[k].dx = mic_pos_2d[j][0] - mic_pos_2d[i][0];
            pair_conf[k].dy = mic_pos_2d[j][1] - mic_pos_2d[i][1];
            pair_conf[k].dist = sqrtf(pair_conf[k].dx * pair_conf[k].dx + pair_conf[k].dy * pair_conf[k].dy);
            k++;
        }
    }
}

static void fft_radix2(float* real, float* imag, int n, int is_inverse) {
    int i, j, k, l, p; 
    float tx, ty, u1, u2, z, cos_z, sin_z, t;
    j = 0;
    for (i = 0; i < n - 1; i++) {
        if (i < j) {
            tx = real[i]; real[i] = real[j]; real[j] = tx;
            ty = imag[i]; imag[i] = imag[j]; imag[j] = ty;
        }
        k = n / 2; while (k <= j) { j -= k; k /= 2; } j += k;
    }
    for (l = 1; l < n; l *= 2) {
        z = PI / l; if (is_inverse) z = -z;
        u1 = 1.0f; u2 = 0.0f;
        cos_z = cosf(z); sin_z = sinf(z);
        for (j = 0; j < l; j++) {
            for (i = j; i < n; i += 2 * l) {
                p = i + l;
                tx = real[p] * u1 - imag[p] * u2; ty = real[p] * u2 + imag[p] * u1;
                real[p] = real[i] - tx; imag[p] = imag[i] - ty;
                real[i] += tx; imag[i] += ty;
            }
            t = u1; u1 = t * cos_z - u2 * sin_z; u2 = t * sin_z + u2 * cos_z;
        }
    }
    if (is_inverse) { for (i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; } }
}

// 缓存 7 帧的 FFT 频谱数据
#define N_FRAMES 7
#define MAX_BINS 200
static float X_history_r[N_FRAMES][NUM_MICS][MAX_BINS];
static float X_history_i[N_FRAMES][NUM_MICS][MAX_BINS];
static int frame_idx = 0;
static int frames_collected = 0;
static float last_angle = 0.0f;

static float run_doa_pipeline(float mic_data[NUM_MICS][FFT_N]) {
    static float mic_real[NUM_MICS][FFT_N]; 
    static float mic_imag[NUM_MICS][FFT_N]; 
    static float P_MUSIC[360];
    
    // 【C89规范对齐】变量统一定义在顶部
    int m, i, j, f, bin, iter, theta, theta_idx;
    float window, df, freq, omega;
    float Rxx_r[NUM_MICS][NUM_MICS], Rxx_i[NUM_MICS][NUM_MICS];
    float V_r[NUM_MICS], V_i[NUM_MICS], V_new_r[NUM_MICS], V_new_i[NUM_MICS];
    float Pn_r[NUM_MICS][NUM_MICS], Pn_i[NUM_MICS][NUM_MICS];
    float a_r[NUM_MICS], a_i[NUM_MICS], y_r[NUM_MICS], y_i[NUM_MICS];
    float norm_sq, norm, vvH_r, vvH_i, phase, cos_t, sin_t, val_r;
    float max_P, best_ang, final_ang;
    int start_bin, end_bin;
    
    for (m = 0; m < NUM_MICS; m++) {
        for (i = 0; i < FFT_N; i++) {
            window = 0.54f - 0.46f * cosf(2.0f * PI * i / (FFT_N - 1));
            mic_real[m][i] = mic_data[m][i] * window;
            mic_imag[m][i] = 0.0f;
        }
        fft_radix2(mic_real[m], mic_imag[m], FFT_N, 0);
    }

    df = SAMPLE_RATE / FFT_N;
    start_bin = (int)ceilf(50.0f / df);
    end_bin = (int)floorf(8000.0f / df);
    if(end_bin >= MAX_BINS) end_bin = MAX_BINS - 1;

    for (m = 0; m < NUM_MICS; m++) {
        for (bin = start_bin; bin <= end_bin; bin++) {
            X_history_r[frame_idx][m][bin] = mic_real[m][bin];
            X_history_i[frame_idx][m][bin] = mic_imag[m][bin];
        }
    }
    frame_idx = (frame_idx + 1) % N_FRAMES;
    
    if (frames_collected < N_FRAMES) {
        frames_collected++;
        return last_angle;
    }

    for (i = 0; i < 360; i++) P_MUSIC[i] = 0.0f;

    for (bin = start_bin; bin <= end_bin; bin++) {
        freq = bin * df;
        omega = 2.0f * PI * freq;

        memset(Rxx_r, 0, sizeof(Rxx_r));
        memset(Rxx_i, 0, sizeof(Rxx_i));

        for (f = 0; f < N_FRAMES; f++) {
            for (i = 0; i < NUM_MICS; i++) {
                float xr = X_history_r[f][i][bin];
                float xi = X_history_i[f][i][bin];
                for (j = 0; j < NUM_MICS; j++) {
                    float yr = X_history_r[f][j][bin];
                    float yi = X_history_i[f][j][bin];
                    Rxx_r[i][j] += (xr * yr + xi * yi);
                    Rxx_i[i][j] += (xi * yr - xr * yi);
                }
            }
        }
        for (i = 0; i < NUM_MICS; i++) {
            for (j = 0; j < NUM_MICS; j++) {
                Rxx_r[i][j] /= N_FRAMES;
                Rxx_i[i][j] /= N_FRAMES;
            }
        }

        for(i=0; i<NUM_MICS; i++) { V_r[i] = 1.0f; V_i[i] = 0.0f; }
        
        for (iter = 0; iter < 20; iter++) { 
            memset(V_new_r, 0, sizeof(V_new_r));
            memset(V_new_i, 0, sizeof(V_new_i));
            norm_sq = 0.0f;
            for (i = 0; i < NUM_MICS; i++) {
                for (j = 0; j < NUM_MICS; j++) {
                    V_new_r[i] += (Rxx_r[i][j] * V_r[j] - Rxx_i[i][j] * V_i[j]);
                    V_new_i[i] += (Rxx_r[i][j] * V_i[j] + Rxx_i[i][j] * V_r[j]);
                }
                norm_sq += (V_new_r[i] * V_new_r[i] + V_new_i[i] * V_new_i[i]);
            }
            norm = sqrtf(norm_sq) + 1e-12f;
            for (i = 0; i < NUM_MICS; i++) {
                V_r[i] = V_new_r[i] / norm;
                V_i[i] = V_new_i[i] / norm;
            }
        }

        for (i = 0; i < NUM_MICS; i++) {
            for (j = 0; j < NUM_MICS; j++) {
                vvH_r = V_r[i] * V_r[j] + V_i[i] * V_i[j];
                vvH_i = V_i[i] * V_r[j] - V_r[i] * V_i[j];
                if (i == j) {
                    Pn_r[i][j] = 1.0f - vvH_r;
                    Pn_i[i][j] = -vvH_i;
                } else {
                    Pn_r[i][j] = -vvH_r;
                    Pn_i[i][j] = -vvH_i;
                }
            }
        }

        for (theta = -180; theta < 180; theta++) {
            cos_t = cosf(theta * DEG2RAD);
            sin_t = sinf(theta * DEG2RAD);

            for (m = 0; m < NUM_MICS; m++) {
                phase = omega * (mic_pos_2d[m][0] * cos_t + mic_pos_2d[m][1] * sin_t) / SOUND_SPEED;
                a_r[m] = cosf(phase);
                a_i[m] = -sinf(phase); 
            }

            memset(y_r, 0, sizeof(y_r));
            memset(y_i, 0, sizeof(y_i));
            
            for (i = 0; i < NUM_MICS; i++) {
                for (j = 0; j < NUM_MICS; j++) {
                    y_r[i] += (Pn_r[i][j] * a_r[j] - Pn_i[i][j] * a_i[j]);
                    y_i[i] += (Pn_r[i][j] * a_i[j] + Pn_i[i][j] * a_r[j]);
                }
            }

            val_r = 0.0f;
            for (i = 0; i < NUM_MICS; i++) {
                val_r += (a_r[i] * y_r[i] + a_i[i] * y_i[i]);
            }

            theta_idx = theta + 180;
            P_MUSIC[theta_idx] += 1.0f / (val_r + 1e-12f);
        }
    }

    max_P = 0.0f;
    best_ang = 0.0f;
    for (i = 0; i < 360; i++) {
        if (P_MUSIC[i] > max_P) {
            max_P = P_MUSIC[i];
            best_ang = (float)(i - 180);
        }
    }

    last_angle = best_ang;
    final_ang = fmodf(best_ang + 180.0f, 360.0f);
    if (final_ang < 0) final_ang += 360.0f;
    return final_ang - 180.0f;
}

// === MicroPython 绑定部分保持不变 (省略以节省空间，直接沿用你发给我的尾部代码即可) ===
static int i2s_dma_cb(void *ctx) { rx_flag = 1; return 0; }
STATIC mp_obj_t Maix_mic_array_init(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) { /* 同原版 */ return mp_const_true; }
MP_DEFINE_CONST_FUN_OBJ_KW(Maix_mic_array_init_obj, 0, Maix_mic_array_init);
STATIC mp_obj_t Maix_mic_array_deinit(void) { return mp_const_true; }
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_deinit_obj, Maix_mic_array_deinit);
STATIC mp_obj_t Maix_mic_array_get_map(void) { mp_raise_ValueError("Disabled."); return mp_const_none; }
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_map_obj, Maix_mic_array_get_map);
STATIC mp_obj_t Maix_mic_array_get_dir(void) {
    volatile uint8_t retry = 100; int i;
    while(rx_flag == 0) { retry--; msleep(1); if(retry == 0) break; }
    if(rx_flag == 0 && retry == 0) { mp_raise_OSError(MP_ETIMEDOUT); return mp_const_false; }
    rx_flag = 0;
    for (i = 0; i < FFT_N; i++) {
        mic_raw_float[0][i] = (float)(i2s_rx_buf[i*8 + 0] >> 16); mic_raw_float[1][i] = (float)(i2s_rx_buf[i*8 + 1] >> 16);
        mic_raw_float[2][i] = (float)(i2s_rx_buf[i*8 + 2] >> 16); mic_raw_float[3][i] = (float)(i2s_rx_buf[i*8 + 3] >> 16);
        mic_raw_float[4][i] = (float)(i2s_rx_buf[i*8 + 4] >> 16); mic_raw_float[5][i] = (float)(i2s_rx_buf[i*8 + 5] >> 16);
        mic_raw_float[6][i] = (float)(i2s_rx_buf[i*8 + 6] >> 16);
    }
    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);
    float final_angle = run_doa_pipeline(mic_raw_float);
    return mp_obj_new_float(final_angle);
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_dir_obj, Maix_mic_array_get_dir);
STATIC mp_obj_t Maix_mic_array_set_led(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) { /* 同原版 */ return mp_const_true; }
MP_DEFINE_CONST_FUN_OBJ_KW(Maix_mic_array_set_led_obj, 2, Maix_mic_array_set_led);
STATIC const mp_rom_map_elem_t Maix_mic_array_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_init), MP_ROM_PTR(&Maix_mic_array_init_obj) }, { MP_ROM_QSTR(MP_QSTR_deinit), MP_ROM_PTR(&Maix_mic_array_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_dir), MP_ROM_PTR(&Maix_mic_array_get_dir_obj) }, { MP_ROM_QSTR(MP_QSTR_set_led), MP_ROM_PTR(&Maix_mic_array_set_led_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_map), MP_ROM_PTR(&Maix_mic_array_get_map_obj) },
};
STATIC MP_DEFINE_CONST_DICT(Maix_mic_array_dict, Maix_mic_array_locals_dict_table);
const mp_obj_type_t Maix_mic_array_type = { { &mp_type_type }, .name = MP_QSTR_MIC_ARRAY, .locals_dict = (mp_obj_dict_t*)&Maix_mic_array_dict, };
