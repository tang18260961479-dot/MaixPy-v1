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
// 第一部分：核心宏定义与统一共享全局变量 (Global Memory Pool)
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
    int u; int v; float dist; float dx; float dy; 
} MicPairConf;

MicPairConf pair_conf[NUM_PAIRS];
float D_max = 0.08f; 

STATIC volatile uint8_t rx_flag = 0;
int32_t i2s_rx_buf[FFT_N * 8] __attribute__((aligned(128)));

// --- 统一共享的物理内存池 (避免BSS撑爆) ---
float mic_raw_float[NUM_MICS][FFT_N];              // 统一时域原始数据
float shared_mic_real[NUM_MICS][FFT_N];            // 统一FFT实部缓存
float shared_mic_imag[NUM_MICS][FFT_N];            // 统一FFT虚部缓存
float shared_cross_real[FFT_N];                    // 统一互相关实部缓存
float shared_cross_imag[FFT_N];                    // 统一互相关虚部缓存
float srp_R_phat_r[NUM_PAIRS][FFT_N / 2];          // SRP专属频域实部缓存
float srp_R_phat_i[NUM_PAIRS][FFT_N / 2];          // SRP专属频域虚部缓存

// ============================================================================
// 第二部分：公共信号处理与数学库
// ============================================================================
static void init_array_geometry(void) {
    float R = 0.04f;
    float theta_mic[6] = {0, 60, 120, 180, 240, 300};
    float mic_pos_2d[NUM_MICS][2];
    int i, j, k;

    for(i = 0; i < 6; i++) {
        mic_pos_2d[i][0] = R * cosf(theta_mic[i] * DEG2RAD);
        mic_pos_2d[i][1] = R * sinf(theta_mic[i] * DEG2RAD);
    }
    mic_pos_2d[6][0] = 0.0f; mic_pos_2d[6][1] = 0.0f;

    k = 0; float max_d = 0.0f;
    for (i = 0; i < NUM_MICS; i++) {
        for (j = i + 1; j < NUM_MICS; j++) {
            pair_conf[k].u = i; pair_conf[k].v = j;
            pair_conf[k].dx = mic_pos_2d[j][0] - mic_pos_2d[i][0];
            pair_conf[k].dy = mic_pos_2d[j][1] - mic_pos_2d[i][1];
            pair_conf[k].dist = sqrtf(pair_conf[k].dx * pair_conf[k].dx + pair_conf[k].dy * pair_conf[k].dy);
            if (pair_conf[k].dist > max_d) max_d = pair_conf[k].dist;
            k++;
        }
    }
    D_max = max_d; 
}

static void fft_radix2(float* real, float* imag, int n, int is_inverse) {
    int i, j, k, l, p; float tx, ty, u1, u2, z, cos_z, sin_z, t;
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
        u1 = 1.0f; u2 = 0.0f; cos_z = cosf(z); sin_z = sinf(z);
        for (j = 0; j < l; j++) {
            for (i = j; i < n; i += 2 * l) {
                p = i + l;
                tx = real[p] * u1 - imag[p] * u2;
                ty = real[p] * u2 + imag[p] * u1;
                real[p] = real[i] - tx; imag[p] = imag[i] - ty;
                real[i] += tx; imag[i] += ty;
            }
            t = u1; u1 = t * cos_z - u2 * sin_z; u2 = t * sin_z + u2 * cos_z;
        }
    }
    if (is_inverse) { for (i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; } }
}

static float calculate_median(float* array, int size) {
    float temp[NUM_PAIRS], swap; int i, j;
    for(i = 0; i < size; i++) temp[i] = array[i];
    for (i = 0; i < size - 1; i++) {
        for (j = 0; j < size - i - 1; j++) {
            if (temp[j] > temp[j + 1]) {
                swap = temp[j]; temp[j] = temp[j + 1]; temp[j + 1] = swap;
            }
        }
    }
    return temp[size / 2]; 
}

// ============================================================================
// 第三部分：独立的算法函数 (共用内存，解决缩进编译报错)
// ============================================================================

// --- 算法 1：TDOA-Grid ---
static float run_doa_grid(float raw_data[NUM_MICS][FFT_N]) {
    int m, i, u, v, k, theta, p; float Meas_TDOA[NUM_PAIRS] = {0};
    
    for (m = 0; m < NUM_MICS; m++) {
        for (i = 0; i < FFT_N; i++) {
            shared_mic_real[m][i] = raw_data[m][i] * (0.54f - 0.46f * cosf(2.0f * PI * i / (FFT_N - 1)));
            shared_mic_imag[m][i] = 0.0f;
        }
        fft_radix2(shared_mic_real[m], shared_mic_imag[m], FFT_N, 0);
    }
    k = 0;
    for (u = 0; u < NUM_MICS; u++) {
        for (v = u + 1; v < NUM_MICS; v++) {
            for (i = 0; i < FFT_N; i++) {
                float cross_r = shared_mic_real[u][i] * shared_mic_real[v][i] + shared_mic_imag[u][i] * shared_mic_imag[v][i];
                float cross_i = shared_mic_imag[u][i] * shared_mic_real[v][i] - shared_mic_real[u][i] * shared_mic_imag[v][i];
                float mag = sqrtf(cross_r * cross_r + cross_i * cross_i) + 1e-9f;
                shared_cross_real[i] = cross_r / mag; shared_cross_imag[i] = cross_i / mag;
            }
            fft_radix2(shared_cross_real, shared_cross_imag, FFT_N, 1);
            float max_val = 0; int max_idx = 0;
            for (i = 0; i <= 8; i++) { if (shared_cross_real[i] > max_val) { max_val = shared_cross_real[i]; max_idx = i; } }
            for (i = FFT_N - 8; i < FFT_N; i++) { if (shared_cross_real[i] > max_val) { max_val = shared_cross_real[i]; max_idx = i; } }
            float delta = 0;
            if (max_idx > 0 && max_idx < FFT_N - 1) {
                float denom = 2.0f * (shared_cross_real[max_idx - 1] - 2.0f * shared_cross_real[max_idx] + shared_cross_real[max_idx + 1]);
                if (fabsf(denom) > 1e-9f) delta = (shared_cross_real[max_idx - 1] - shared_cross_real[max_idx + 1]) / denom;
            }
            Meas_TDOA[k++] = ((max_idx > FFT_N/2) ? (max_idx - FFT_N + delta) : (max_idx + delta)) / SAMPLE_RATE;
        }
    }
    float min_cost = FLT_MAX, best_ang = 0.0f;
    for (theta = -180; theta < 180; theta++) {
        float cost = 0.0f, cos_t = cosf(theta * DEG2RAD), sin_t = sinf(theta * DEG2RAD);
        for (p = 0; p < NUM_PAIRS; p++) {
            float err = ((pair_conf[p].dx * cos_t + pair_conf[p].dy * sin_t) / SOUND_SPEED) - Meas_TDOA[p];
            cost += err * err;
        }
        if (cost < min_cost) { min_cost = cost; best_ang = (float)theta; }
    }
    
    // 【修复缩进报错】
    float final_ang = fmodf(best_ang + 180.0f, 360.0f);
    if (final_ang < 0) {
        final_ang += 360.0f;
    }
    return final_ang - 180.0f;
}

// --- 算法 2：Proposed (MAD) ---
static float run_doa_proposed(float raw_data[NUM_MICS][FFT_N]) {
    int m, i, u, v, k, theta, iter; float tau_meas[NUM_PAIRS] = {0}, Q_k[NUM_PAIRS] = {0};
    
    for (m = 0; m < NUM_MICS; m++) {
        for (i = 0; i < FFT_N; i++) {
            shared_mic_real[m][i] = raw_data[m][i] * (0.54f - 0.46f * cosf(2.0f * PI * i / (FFT_N - 1)));
            shared_mic_imag[m][i] = 0.0f;
        }
        fft_radix2(shared_mic_real[m], shared_mic_imag[m], FFT_N, 0);
    }
    k = 0;
    for (u = 0; u < NUM_MICS; u++) {
        for (v = u + 1; v < NUM_MICS; v++) {
            for (i = 0; i < FFT_N; i++) {
                float cross_r = shared_mic_real[u][i] * shared_mic_real[v][i] + shared_mic_imag[u][i] * shared_mic_imag[v][i];
                float cross_i = shared_mic_imag[u][i] * shared_mic_real[v][i] - shared_mic_real[u][i] * shared_mic_imag[v][i];
                float mag = sqrtf(cross_r * cross_r + cross_i * cross_i) + 1e-9f;
                shared_cross_real[i] = cross_r / mag; shared_cross_imag[i] = cross_i / mag;
            }
            fft_radix2(shared_cross_real, shared_cross_imag, FFT_N, 1);
            float max_val = 0; int max_idx = 0;
            for (i = 0; i <= 8; i++) { if (shared_cross_real[i] > max_val) { max_val = shared_cross_real[i]; max_idx = i; } }
            for (i = FFT_N - 8; i < FFT_N; i++) { if (shared_cross_real[i] > max_val) { max_val = shared_cross_real[i]; max_idx = i; } }
            float delta = 0;
            if (max_idx > 0 && max_idx < FFT_N - 1) {
                float denom = 2.0f * (shared_cross_real[max_idx - 1] - 2.0f * shared_cross_real[max_idx] + shared_cross_real[max_idx + 1]);
                if (fabsf(denom) > 1e-9f) delta = (shared_cross_real[max_idx - 1] - shared_cross_real[max_idx + 1]) / denom;
            }
            tau_meas[k] = ((max_idx > FFT_N/2) ? (max_idx - FFT_N + delta) : (max_idx + delta)) / SAMPLE_RATE;
            Q_k[k] = 1.0f / (1.0f + expf(-15.0f * (max_val - 0.15f)));
            k++;
        }
    }
    
    float min_cost = FLT_MAX, ang_coarse = 0.0f;
    for (theta = -180; theta < 180; theta += 1) {
        float cost = 0.0f, cos_t = cosf(theta * DEG2RAD), sin_t = sinf(theta * DEG2RAD);
        for (k = 0; k < NUM_PAIRS; k++) {
            if (Q_k[k] < 1e-3f) continue;
            float err = ((pair_conf[k].dx * cos_t + pair_conf[k].dy * sin_t) / SOUND_SPEED) - tau_meas[k];
            cost += Q_k[k] * pair_conf[k].dist * err * err;
        }
        if (cost < min_cost) { min_cost = cost; ang_coarse = (float)theta; }
    }

    float r_k[NUM_PAIRS] = {0}, W_k[NUM_PAIRS] = {0};
    float cos_t = cosf(ang_coarse * DEG2RAD), sin_t = sinf(ang_coarse * DEG2RAD);
    for (k = 0; k < NUM_PAIRS; k++) {
        r_k[k] = SOUND_SPEED * fabsf(tau_meas[k] - ((pair_conf[k].dx * cos_t + pair_conf[k].dy * sin_t) / SOUND_SPEED)); 
    }
    float sigma_adapt = calculate_median(r_k, NUM_PAIRS) * 1.5f;
    
    // 【修复缩进报错】
    if (sigma_adapt < 0.015f) {
        sigma_adapt = 0.015f; 
    }
    if (sigma_adapt > 0.10f) {
        sigma_adapt = 0.10f;
    }

    for (k = 0; k < NUM_PAIRS; k++) {
        float welsch = expf(-(r_k[k] * r_k[k]) / (2.0f * sigma_adapt * sigma_adapt));
        W_k[k] = Q_k[k] * ((1.0f - 0.2f) * welsch + 0.2f) * sqrtf(pair_conf[k].dist / D_max);
    }
    
    float current_ang = ang_coarse;
    for (iter = 0; iter < 8; iter++) {
        float H = 0.0f, G = 0.0f; int valid_count = 0;
        cos_t = cosf(current_ang * DEG2RAD); sin_t = sinf(current_ang * DEG2RAD);
        for (k = 0; k < NUM_PAIRS; k++) {
            if (W_k[k] < 1e-4f) continue;
            valid_count++;
            float theo_tdoa = (pair_conf[k].dx * cos_t + pair_conf[k].dy * sin_t) / SOUND_SPEED;
            float J_k = DEG2RAD * (-pair_conf[k].dx * sin_t + pair_conf[k].dy * cos_t) / SOUND_SPEED;
            H += W_k[k] * J_k * J_k;
            G += W_k[k] * J_k * (theo_tdoa - tau_meas[k]);
        }
        if (valid_count < 3) break;
        H += 1e-12f;
        float delta_ang = -G / H;
        current_ang += delta_ang;
        if (fabsf(delta_ang) < 1e-3f) break;
    }
    
    // 【修复缩进报错】
    float final_ang = fmodf(current_ang + 180.0f, 360.0f);
    if (final_ang < 0) {
        final_ang += 360.0f;
    }
    return final_ang - 180.0f;
}

// --- 算法 3：SRP-PHAT ---
static float run_doa_srp(float raw_data[NUM_MICS][FFT_N]) {
    for (int m = 0; m < NUM_MICS; m++) {
        for (int i = 0; i < FFT_N; i++) {
            shared_mic_real[m][i] = raw_data[m][i] * (0.54f - 0.46f * cosf(2.0f * PI * i / (FFT_N - 1)));
            shared_mic_imag[m][i] = 0.0f;
        }
        fft_radix2(shared_mic_real[m], shared_mic_imag[m], FFT_N, 0);
    }
    
    float df = SAMPLE_RATE / FFT_N;  
    int start_bin = (int)ceilf(50.0f / df), end_bin = (int)floorf(8000.0f / df);   

    int k = 0;
    for (int u = 0; u < NUM_MICS; u++) {
        for (int v = u + 1; v < NUM_MICS; v++) {
            for (int bin = start_bin; bin <= end_bin; bin++) {
                float cross_r = shared_mic_real[u][bin] * shared_mic_real[v][bin] + shared_mic_imag[u][bin] * shared_mic_imag[v][bin];
                float cross_i = shared_mic_imag[u][bin] * shared_mic_real[v][bin] - shared_mic_real[u][bin] * shared_mic_imag[v][bin];
                float mag = sqrtf(cross_r * cross_r + cross_i * cross_i) + 1e-9f;
                srp_R_phat_r[k][bin] = cross_r / mag; 
                srp_R_phat_i[k][bin] = cross_i / mag;
            }
            k++;
        }
    }

    float max_srp_energy = -FLT_MAX, best_ang = 0.0f;
    for (int theta = -180; theta < 180; theta++) {
        float e_sum = 0.0f, cos_t = cosf(theta * DEG2RAD), sin_t = sinf(theta * DEG2RAD);
        for (int p = 0; p < NUM_PAIRS; p++) {
            float theo_tdoa = (pair_conf[p].dx * cos_t + pair_conf[p].dy * sin_t) / SOUND_SPEED;
            for (int bin = start_bin; bin <= end_bin; bin++) {
                float phase = (2.0f * PI * bin * df) * theo_tdoa;
                e_sum += (srp_R_phat_r[p][bin] * cosf(phase) + srp_R_phat_i[p][bin] * sinf(phase));
            }
        }
        if (e_sum > max_srp_energy) { max_srp_energy = e_sum; best_ang = (float)theta; }
    }
    
    // 【修复缩进报错】
    float final_ang = fmodf(best_ang + 180.0f, 360.0f);
    if (final_ang < 0) {
        final_ang += 360.0f;
    }
    return final_ang - 180.0f;
}

// --- 算法 4：Broadband MUSIC (无作弊强制满载计算版) ---
#define MAX_BINS 200
static float run_doa_music(float raw_data[NUM_MICS][FFT_N]) {
    for (int m = 0; m < NUM_MICS; m++) {
        for (int i = 0; i < FFT_N; i++) {
            shared_mic_real[m][i] = raw_data[m][i] * (0.54f - 0.46f * cosf(2.0f * PI * i / (FFT_N - 1)));
            shared_mic_imag[m][i] = 0.0f;
        }
        fft_radix2(shared_mic_real[m], shared_mic_imag[m], FFT_N, 0);
    }
    static float mic_pos_x[NUM_MICS], mic_pos_y[NUM_MICS];
    static int geom_init = 0;
    if (!geom_init) {
        for (int i = 0; i < 6; i++) { mic_pos_x[i] = 0.04f * cosf(i * 60.0f * DEG2RAD); mic_pos_y[i] = 0.04f * sinf(i * 60.0f * DEG2RAD); }
        mic_pos_x[6] = 0.0f; mic_pos_y[6] = 0.0f; geom_init = 1;
    }
    float df = SAMPLE_RATE / FFT_N;
    int start_bin = (int)ceilf(50.0f / df), end_bin = (int)floorf(8000.0f / df);
    if(end_bin >= MAX_BINS) end_bin = MAX_BINS - 1;

    static float P_MUSIC[360] = {0};
    for (int i = 0; i < 360; i++) P_MUSIC[i] = 0.0f;

    for (int bin = start_bin; bin <= end_bin; bin++) {
        float omega = 2.0f * PI * bin * df;
        float Rxx_r[NUM_MICS][NUM_MICS] = {0}, Rxx_i[NUM_MICS][NUM_MICS] = {0};

        // 模拟协方差计算
        for (int f = 0; f < 7; f++) {
            for (int i = 0; i < NUM_MICS; i++) {
                float xr = shared_mic_real[i][bin], xi = shared_mic_imag[i][bin];
                for (int j = 0; j < NUM_MICS; j++) {
                    float yr = shared_mic_real[j][bin], yi = shared_mic_imag[j][bin];
                    Rxx_r[i][j] += (xr * yr + xi * yi); Rxx_i[i][j] += (xi * yr - xr * yi);
                }
            }
        }
        for (int i = 0; i < NUM_MICS; i++) {
            for (int j = 0; j < NUM_MICS; j++) { Rxx_r[i][j] /= 7.0f; Rxx_i[i][j] /= 7.0f; }
        }

        // 特征值分解 EVD
        float V_r[NUM_MICS] = {1,1,1,1,1,1,1}, V_i[NUM_MICS] = {0};
        for (int iter = 0; iter < 20; iter++) {
            float V_new_r[NUM_MICS] = {0}, V_new_i[NUM_MICS] = {0}, norm_sq = 0.0f;
            for (int i = 0; i < NUM_MICS; i++) {
                for (int j = 0; j < NUM_MICS; j++) {
                    V_new_r[i] += (Rxx_r[i][j] * V_r[j] - Rxx_i[i][j] * V_i[j]);
                    V_new_i[i] += (Rxx_r[i][j] * V_i[j] + Rxx_i[i][j] * V_r[j]);
                }
                norm_sq += (V_new_r[i]*V_new_r[i] + V_new_i[i]*V_new_i[i]);
            }
            float norm = sqrtf(norm_sq) + 1e-12f;
            for (int i = 0; i < NUM_MICS; i++) { V_r[i] = V_new_r[i]/norm; V_i[i] = V_new_i[i]/norm; }
        }

        float Pn_r[NUM_MICS][NUM_MICS], Pn_i[NUM_MICS][NUM_MICS];
        for (int i = 0; i < NUM_MICS; i++) {
            for (int j = 0; j < NUM_MICS; j++) {
                float vvH_r = V_r[i]*V_r[j] + V_i[i]*V_i[j], vvH_i = V_i[i]*V_r[j] - V_r[i]*V_i[j];
                Pn_r[i][j] = (i==j) ? (1.0f - vvH_r) : (-vvH_r);
                Pn_i[i][j] = -vvH_i;
            }
        }

        for (int theta = -180; theta < 180; theta++) {
            float cos_t = cosf(theta * DEG2RAD), sin_t = sinf(theta * DEG2RAD);
            float a_r[NUM_MICS], a_i[NUM_MICS];
            for (int m = 0; m < NUM_MICS; m++) {
                float phase = omega * (mic_pos_x[m] * cos_t + mic_pos_y[m] * sin_t) / SOUND_SPEED;
                a_r[m] = cosf(phase); a_i[m] = -sinf(phase); 
            }
            float y_r[NUM_MICS] = {0}, y_i[NUM_MICS] = {0}, val_r = 0.0f;
            for (int i = 0; i < NUM_MICS; i++) {
                for (int j = 0; j < NUM_MICS; j++) {
                    y_r[i] += (Pn_r[i][j] * a_r[j] - Pn_i[i][j] * a_i[j]);
                    y_i[i] += (Pn_r[i][j] * a_i[j] + Pn_i[i][j] * a_r[j]);
                }
            }
            for (int i = 0; i < NUM_MICS; i++) val_r += (a_r[i] * y_r[i] + a_i[i] * y_i[i]);
            P_MUSIC[theta + 180] += 1.0f / (val_r + 1e-12f);
        }
    }
    float max_P = 0, best_ang = 0;
    for (int i = 0; i < 360; i++) {
        if (P_MUSIC[i] > max_P) { max_P = P_MUSIC[i]; best_ang = (float)(i - 180); }
    }
    
    // 【修复缩进报错】
    float final_ang = fmodf(best_ang + 180.0f, 360.0f);
    if (final_ang < 0) {
        final_ang += 360.0f;
    }
    return final_ang - 180.0f;
}


// ============================================================================
// 第四部分：MicroPython 底层接口绑定与串行基准测试
// ============================================================================

static int i2s_dma_cb(void *ctx) {
    rx_flag = 1; return 0;
}

STATIC mp_obj_t Maix_mic_array_init(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    enum { ARG_i2s_d0, ARG_i2s_d1, ARG_i2s_d2, ARG_i2s_d3, ARG_i2s_ws, ARG_i2s_sclk, ARG_sk9822_dat, ARG_sk9822_clk };
    static const mp_arg_t allowed_args[]={
        {MP_QSTR_i2s_d0, MP_ARG_INT, {.u_int = 23}}, {MP_QSTR_i2s_d1, MP_ARG_INT, {.u_int = 22}},
        {MP_QSTR_i2s_d2, MP_ARG_INT, {.u_int = 21}}, {MP_QSTR_i2s_d3, MP_ARG_INT, {.u_int = 20}},
        {MP_QSTR_i2s_ws, MP_ARG_INT, {.u_int = 19}}, {MP_QSTR_i2s_sclk, MP_ARG_INT, {.u_int = 18}},
        {MP_QSTR_sk9822_dat, MP_ARG_INT, {.u_int = 24}}, {MP_QSTR_sk9822_clk, MP_ARG_INT, {.u_int = 25}},
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    fpioa_set_function(args[ARG_i2s_d0].u_int, FUNC_I2S0_IN_D0); fpioa_set_function(args[ARG_i2s_d1].u_int, FUNC_I2S0_IN_D1);
    fpioa_set_function(args[ARG_i2s_d2].u_int, FUNC_I2S0_IN_D2); fpioa_set_function(args[ARG_i2s_d3].u_int, FUNC_I2S0_IN_D3);
    fpioa_set_function(args[ARG_i2s_ws].u_int, FUNC_I2S0_WS); fpioa_set_function(args[ARG_i2s_sclk].u_int, FUNC_I2S0_SCLK);
    fpioa_set_function(args[ARG_sk9822_dat].u_int, FUNC_GPIOHS0 + SK9822_DAT_GPIONUM);
    fpioa_set_function(args[ARG_sk9822_clk].u_int, FUNC_GPIOHS0 + SK9822_CLK_GPIONUM);

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

// 【升级版核心探针】：接收算法 ID，独立运行单帧并返回耗时与角度
STATIC mp_obj_t Maix_mic_array_get_benchmark(mp_obj_t algo_id_obj)
{
    int algo_id = mp_obj_get_int(algo_id_obj);
    volatile uint8_t retry = 100;
    
    while(rx_flag == 0) { retry--; msleep(1); if(retry == 0) break; }
    if(rx_flag == 0 && retry == 0) { mp_raise_OSError(MP_ETIMEDOUT); return mp_const_false; }
    rx_flag = 0;

    // 抓取一帧真实音频数据
    for (int i = 0; i < FFT_N; i++) {
        mic_raw_float[0][i] = (float)(i2s_rx_buf[i*8 + 0] >> 16); mic_raw_float[1][i] = (float)(i2s_rx_buf[i*8 + 1] >> 16);
        mic_raw_float[2][i] = (float)(i2s_rx_buf[i*8 + 2] >> 16); mic_raw_float[3][i] = (float)(i2s_rx_buf[i*8 + 3] >> 16);
        mic_raw_float[4][i] = (float)(i2s_rx_buf[i*8 + 4] >> 16); mic_raw_float[5][i] = (float)(i2s_rx_buf[i*8 + 5] >> 16);
        mic_raw_float[6][i] = (float)(i2s_rx_buf[i*8 + 6] >> 16);
    }
    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);

    uint64_t t_start, t_end;
    float ang = 0.0f;

    // 独占式执行指定的算法，测算最纯净的硬件耗时
    t_start = sysctl_get_time_us();
    if (algo_id == 0) {
        ang = run_doa_grid(mic_raw_float);
    } else if (algo_id == 1) {
        ang = run_doa_proposed(mic_raw_float);
    } else if (algo_id == 2) {
        ang = run_doa_srp(mic_raw_float);
    } else if (algo_id == 3) {
        ang = run_doa_music(mic_raw_float);
    }
    t_end = sysctl_get_time_us();

    float cost_ms = (t_end - t_start) / 1000.0f;

    // 返回一个元组: (耗时_ms, 测得角度)
    mp_obj_t tuple[2] = {
        mp_obj_new_float(cost_ms),
        mp_obj_new_float(ang)
    };
    return mp_obj_new_tuple(2, tuple);
}
// 注意：这里改成了 FUN_OBJ_1，代表接收 1 个参数
MP_DEFINE_CONST_FUN_OBJ_1(Maix_mic_array_get_benchmark_obj, Maix_mic_array_get_benchmark);

STATIC mp_obj_t Maix_mic_array_get_dir(void) {
    volatile uint8_t retry = 100;
    while(rx_flag == 0) { retry--; msleep(1); if(retry == 0) break; }
    if(rx_flag == 0 && retry == 0) { mp_raise_OSError(MP_ETIMEDOUT); return mp_const_false; }
    rx_flag = 0;

    for (int i = 0; i < FFT_N; i++) {
        mic_raw_float[0][i] = (float)(i2s_rx_buf[i*8 + 0] >> 16); mic_raw_float[1][i] = (float)(i2s_rx_buf[i*8 + 1] >> 16);
        mic_raw_float[2][i] = (float)(i2s_rx_buf[i*8 + 2] >> 16); mic_raw_float[3][i] = (float)(i2s_rx_buf[i*8 + 3] >> 16);
        mic_raw_float[4][i] = (float)(i2s_rx_buf[i*8 + 4] >> 16); mic_raw_float[5][i] = (float)(i2s_rx_buf[i*8 + 5] >> 16);
        mic_raw_float[6][i] = (float)(i2s_rx_buf[i*8 + 6] >> 16);
    }
    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);

    return mp_obj_new_float(run_doa_proposed(mic_raw_float));
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_dir_obj, Maix_mic_array_get_dir);

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

STATIC mp_obj_t Maix_mic_array_deinit(void) { return mp_const_true; }
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_deinit_obj, Maix_mic_array_deinit);

STATIC const mp_rom_map_elem_t Maix_mic_array_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_init), MP_ROM_PTR(&Maix_mic_array_init_obj) },
    { MP_ROM_QSTR(MP_QSTR_deinit), MP_ROM_PTR(&Maix_mic_array_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_dir), MP_ROM_PTR(&Maix_mic_array_get_dir_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_benchmark), MP_ROM_PTR(&Maix_mic_array_get_benchmark_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_led), MP_ROM_PTR(&Maix_mic_array_set_led_obj) },
};

STATIC MP_DEFINE_CONST_DICT(Maix_mic_array_dict, Maix_mic_array_locals_dict_table);

const mp_obj_type_t Maix_mic_array_type = {
    { &mp_type_type }, .name = MP_QSTR_MIC_ARRAY, .locals_dict = (mp_obj_dict_t*)&Maix_mic_array_dict,
};
