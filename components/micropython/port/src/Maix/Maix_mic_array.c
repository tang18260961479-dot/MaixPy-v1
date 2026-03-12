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
    int i, j, k = 0;

    for(i = 0; i < 6; i++) {
        mic_pos_2d[i][0] = R * cosf(theta_mic[i] * DEG2RAD);
        mic_pos_2d[i][1] = R * sinf(theta_mic[i] * DEG2RAD);
    }
    mic_pos_2d[6][0] = 0.0f; 
    mic_pos_2d[6][1] = 0.0f;

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

// ============================================================================
// 第二部分：DOA 定位追踪管线 (【内核替换】: Broadband MUSIC + 卡尔曼)
// ============================================================================

// MUSIC 算法专属静态滑动窗口，缓存 7 帧 FFT 频谱数据
#define N_FRAMES 7
#define MAX_BINS 200
static float X_history_r[N_FRAMES][NUM_MICS][MAX_BINS];
static float X_history_i[N_FRAMES][NUM_MICS][MAX_BINS];
static int frame_idx = 0;
static int frames_collected = 0;
static float last_angle = 0.0f;

static float run_doa_pipeline(float mic_data[NUM_MICS][FFT_N]) {
    // -------------------------------------------------------------
    // 【C90 严苛规范】：将 MUSIC 算法所有海量变量在函数最顶层统一声明
    // -------------------------------------------------------------
    static float mic_real[NUM_MICS][FFT_N]; 
    static float mic_imag[NUM_MICS][FFT_N]; 
    static float mic_pos_x[NUM_MICS], mic_pos_y[NUM_MICS];
    static int geom_init = 0;
    static float P_MUSIC[360];

    int m, i, j, f, bin, iter, theta, theta_idx;
    float window, df, freq, omega, phase;
    int start_bin, end_bin;
    
    float Rxx_r[NUM_MICS][NUM_MICS], Rxx_i[NUM_MICS][NUM_MICS];
    float xr, xi, yr, yi;
    float V_r[NUM_MICS], V_i[NUM_MICS], V_new_r[NUM_MICS], V_new_i[NUM_MICS];
    float norm_sq, norm, vvH_r, vvH_i;
    float Pn_r[NUM_MICS][NUM_MICS], Pn_i[NUM_MICS][NUM_MICS];
    float cos_t, sin_t;
    float a_r[NUM_MICS], a_i[NUM_MICS];
    float y_r[NUM_MICS], y_i[NUM_MICS];
    float val_r, max_P, best_ang, raw_angle;

    // 1. 物理阵列坐标初始化
    if (!geom_init) {
        for (i = 0; i < 6; i++) {
            mic_pos_x[i] = 0.04f * cosf(i * 60.0f * DEG2RAD);
            mic_pos_y[i] = 0.04f * sinf(i * 60.0f * DEG2RAD);
        }
        mic_pos_x[6] = 0.0f; mic_pos_y[6] = 0.0f;
        geom_init = 1;
    }

    // 2. 加窗与正向 FFT
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

    // 3. 将新一帧数据推入滑动缓存区
    for (m = 0; m < NUM_MICS; m++) {
        for (bin = start_bin; bin <= end_bin; bin++) {
            X_history_r[frame_idx][m][bin] = mic_real[m][bin];
            X_history_i[frame_idx][m][bin] = mic_imag[m][bin];
        }
    }
    frame_idx = (frame_idx + 1) % N_FRAMES;
    
    // 如果缓存还没填满 7 帧，无法构建满秩协方差矩阵，先返回历史角度
    if (frames_collected < N_FRAMES) {
        frames_collected++;
        return last_angle;
    }

    for (i = 0; i < 360; i++) P_MUSIC[i] = 0.0f;

    // 4. 【核心管线】：宽带 MUSIC 特征分解
    for (bin = start_bin; bin <= end_bin; bin++) {
        freq = bin * df;
        omega = 2.0f * PI * freq;

        // A. 初始化并计算 7x7 协方差矩阵
        for (i = 0; i < NUM_MICS; i++) {
            for (j = 0; j < NUM_MICS; j++) {
                Rxx_r[i][j] = 0.0f;
                Rxx_i[i][j] = 0.0f;
            }
            V_r[i] = 1.0f;
            V_i[i] = 0.0f;
        }

        for (f = 0; f < N_FRAMES; f++) {
            for (i = 0; i < NUM_MICS; i++) {
                xr = X_history_r[f][i][bin];
                xi = X_history_i[f][i][bin];
                for (j = 0; j < NUM_MICS; j++) {
                    yr = X_history_r[f][j][bin];
                    yi = X_history_i[f][j][bin];
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

        // B. 幂迭代法求解主特征向量 (Signal Subspace V)
        for (iter = 0; iter < 20; iter++) {
            norm_sq = 0.0f;
            for (i = 0; i < NUM_MICS; i++) {
                V_new_r[i] = 0.0f;
                V_new_i[i] = 0.0f;
            }
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

        // C. 构建噪声子空间投影矩阵 Pn = I - V*V^H
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

        // D. 全向空间谱扫描
        for (theta = -180; theta < 180; theta++) {
            cos_t = cosf(theta * DEG2RAD);
            sin_t = sinf(theta * DEG2RAD);

            for (m = 0; m < NUM_MICS; m++) {
                phase = omega * (mic_pos_x[m] * cos_t + mic_pos_y[m] * sin_t) / SOUND_SPEED;
                a_r[m] = cosf(phase);
                a_i[m] = -sinf(phase); // 补偿 FFT 共轭，保证复平面指向正确
            }

            for (i = 0; i < NUM_MICS; i++) {
                y_r[i] = 0.0f;
                y_i[i] = 0.0f;
            }

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

    // 5. 寻找最优角度
    max_P = 0.0f;
    best_ang = 0.0f;
    for (i = 0; i < 360; i++) {
        if (P_MUSIC[i] > max_P) {
            max_P = P_MUSIC[i];
            best_ang = (float)(i - 180);
        }
    }

    last_angle = best_ang;
    raw_angle = fmodf(best_ang + 180.0f, 360.0f);
    if (raw_angle < 0) raw_angle += 360.0f;
    raw_angle -= 180.0f;

    // ============================================================================
    // 6. 【系统锁定】：鲁棒 Alpha-Beta 卡尔曼滤波器 (完全继承，防瞬移)
    // ============================================================================
    {
        static int tracker_initialized = 0;
        static float est_angle = 0.0f;
        static float est_velocity = 0.0f;
        
        const float ALPHA = 0.3f;           
        const float BETA  = 0.02f;          
        const float HUBER_THRESH = 15.0f;   
        const float dt = 0.064f; // 16000Hz 下处理 1024 个点的严格物理耗时
        
        float pred_angle, raw_residual, abs_res, robust_weight, robust_residual;

        if (!tracker_initialized) {
            est_angle = raw_angle;
            est_velocity = 0.0f;
            tracker_initialized = 1;
            return est_angle;
        }

        pred_angle = est_angle + est_velocity * dt;

        raw_residual = raw_angle - pred_angle;
        while (raw_residual > 180.0f)  raw_residual -= 360.0f;
        while (raw_residual < -180.0f) raw_residual += 360.0f;

        abs_res = fabsf(raw_residual);
        robust_weight = 1.0f;
        if (abs_res > HUBER_THRESH) {
            robust_weight = HUBER_THRESH / abs_res;
        }
        robust_residual = raw_residual * robust_weight;

        est_angle = pred_angle + ALPHA * robust_residual;
        est_velocity = est_velocity + BETA * (robust_residual / dt);

        while (est_angle > 180.0f)  est_angle -= 360.0f;
        while (est_angle < -180.0f) est_angle += 360.0f;

        return est_angle;
    }
}

// ============================================================================
// 第三部分：核心 API 与 【物理级 FD-PCF 频域束波成型 (绝对锁死)】
// ============================================================================
static int i2s_dma_cb(void *ctx) { rx_flag = 1; return 0; }

STATIC mp_obj_t Maix_mic_array_init(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    enum { ARG_i2s_d0, ARG_i2s_d1, ARG_i2s_d2, ARG_i2s_d3, ARG_i2s_ws, ARG_i2s_sclk, ARG_sk9822_dat, ARG_sk9822_clk, };
    static const mp_arg_t allowed_args[]={{MP_QSTR_i2s_d0, MP_ARG_INT, {.u_int = 23}}, {MP_QSTR_i2s_d1, MP_ARG_INT, {.u_int = 22}}, {MP_QSTR_i2s_d2, MP_ARG_INT, {.u_int = 21}}, {MP_QSTR_i2s_d3, MP_ARG_INT, {.u_int = 20}}, {MP_QSTR_i2s_ws, MP_ARG_INT, {.u_int = 19}}, {MP_QSTR_i2s_sclk, MP_ARG_INT, {.u_int = 18}}, {MP_QSTR_sk9822_dat, MP_ARG_INT, {.u_int = 24}}, {MP_QSTR_sk9822_clk, MP_ARG_INT, {.u_int = 25}},};
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)]; int i;
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    fpioa_set_function(args[ARG_i2s_d0].u_int, FUNC_I2S0_IN_D0); fpioa_set_function(args[ARG_i2s_d1].u_int, FUNC_I2S0_IN_D1); fpioa_set_function(args[ARG_i2s_d2].u_int, FUNC_I2S0_IN_D2); fpioa_set_function(args[ARG_i2s_d3].u_int, FUNC_I2S0_IN_D3); fpioa_set_function(args[ARG_i2s_ws].u_int, FUNC_I2S0_WS); fpioa_set_function(args[ARG_i2s_sclk].u_int, FUNC_I2S0_SCLK); fpioa_set_function(args[ARG_sk9822_dat].u_int, FUNC_GPIOHS0 + SK9822_DAT_GPIONUM); fpioa_set_function(args[ARG_sk9822_clk].u_int, FUNC_GPIOHS0 + SK9822_CLK_GPIONUM);
    sipeed_init_mic_array_led();
    sysctl_pll_set_freq(SYSCTL_PLL2, PLL2_OUTPUT_FREQ); sysctl_clock_enable(SYSCTL_CLOCK_I2S0);
    i2s_init(I2S_DEVICE_0, I2S_RECEIVER, 0x0F);
    for(i = 0; i < 4; i++) i2s_rx_channel_config(I2S_DEVICE_0, I2S_CHANNEL_0 + i, RESOLUTION_32_BIT, SCLK_CYCLES_32, TRIGGER_LEVEL_4, STANDARD_MODE);
    i2s_set_sample_rate(I2S_DEVICE_0, SAMPLE_RATE);
    init_array_geometry();
    dmac_set_irq(DMAC_CHANNEL4, i2s_dma_cb, NULL, 3);
    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);
    return mp_const_true;
}
MP_DEFINE_CONST_FUN_OBJ_KW(Maix_mic_array_init_obj, 0, Maix_mic_array_init);

STATIC mp_obj_t Maix_mic_array_deinit(void) { return mp_const_true; }
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_deinit_obj, Maix_mic_array_deinit);

// 录音外设：锁定 FD-PCF 零相位掩膜与亚采样级对齐
STATIC mp_obj_t Maix_mic_array_get_beam_audio(void) {
    volatile uint8_t retry = 100;
    static float mic_r_all[NUM_MICS][FFT_N];
    float target_angle, cos_t, sin_t;
    float mx[6] = {0.04f*cosf(0), 0.04f*cosf(60*DEG2RAD), 0.04f*cosf(120*DEG2RAD), 0.04f*cosf(180*DEG2RAD), 0.04f*cosf(240*DEG2RAD), 0.04f*cosf(300*DEG2RAD)};
    float my[6] = {0.04f*sinf(0), 0.04f*sinf(60*DEG2RAD), 0.04f*sinf(120*DEG2RAD), 0.04f*sinf(180*DEG2RAD), 0.04f*sinf(240*DEG2RAD), 0.04f*sinf(300*DEG2RAD)};
    
    static float mic_r[6][FFT_N];
    static float mic_i[6][FFT_N];
    static float out_r[FFT_N], out_i[FFT_N];
    static float G_smooth[FFT_N/2 + 1] = {0};
    
    static int16_t pcm_out[FFT_N];
    static float prev_x = 0.0f, prev_y = 0.0f;
    int i, m, k;
    float freq, omega, sum_r, sum_i, incoh_energy, tau, phase, a_r, a_i, Xr, Xi, Xa_r, Xa_i, coh_energy, pcf, gain, y, amplified;
    mp_obj_t tuple[2];

    while(rx_flag == 0) { retry--; msleep(1); if(retry == 0) break; }
    if(rx_flag == 0 && retry == 0) { mp_raise_OSError(MP_ETIMEDOUT); return mp_const_false; }
    rx_flag = 0;

    for (i = 0; i < FFT_N; i++) {
        mic_r_all[0][i] = (float)(i2s_rx_buf[i*8 + 0] >> 16);
        mic_r_all[1][i] = (float)(i2s_rx_buf[i*8 + 1] >> 16);
        mic_r_all[2][i] = (float)(i2s_rx_buf[i*8 + 2] >> 16);
        mic_r_all[3][i] = (float)(i2s_rx_buf[i*8 + 3] >> 16);
        mic_r_all[4][i] = (float)(i2s_rx_buf[i*8 + 4] >> 16);
        mic_r_all[5][i] = (float)(i2s_rx_buf[i*8 + 5] >> 16);
        mic_r_all[6][i] = (float)(i2s_rx_buf[i*8 + 6] >> 16); 
    }

    if (shared_data_ready == 0) {
        for(m=0; m<NUM_MICS; m++) {
            for(i=0; i<FFT_N; i++) shared_doa_mic_data[m][i] = mic_r_all[m][i];
        }
        shared_data_ready = 1; 
    }

    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);

    target_angle = current_target_angle;
    cos_t = cosf(target_angle * DEG2RAD);
    sin_t = sinf(target_angle * DEG2RAD);

    for(m=0; m<6; m++) {
        for(i=0; i<FFT_N; i++) {
            mic_r[m][i] = mic_r_all[m][i];
            mic_i[m][i] = 0.0f;
        }
        fft_radix2(mic_r[m], mic_i[m], FFT_N, 0);
    }

    out_r[0] = 0; out_i[0] = 0;
    out_r[FFT_N/2] = 0; out_i[FFT_N/2] = 0;

    for (k = 1; k < FFT_N/2; k++) {
        freq = (float)k * SAMPLE_RATE / FFT_N;
        omega = 2.0f * PI * freq;

        sum_r = 0.0f; sum_i = 0.0f;
        incoh_energy = 0.0f;

        for (m = 0; m < 6; m++) {
            tau = (mx[m]*cos_t + my[m]*sin_t) / SOUND_SPEED;
            phase = omega * tau;
            
            a_r = cosf(phase);
            a_i = sinf(phase);

            Xr = mic_r[m][k];
            Xi = mic_i[m][k];
            
            Xa_r = Xr * a_r - Xi * a_i;
            Xa_i = Xr * a_i + Xi * a_r;

            sum_r += Xa_r;
            sum_i += Xa_i;
            incoh_energy += (Xr*Xr + Xi*Xi);
        }
        
        sum_r /= 6.0f;
        sum_i /= 6.0f;
        incoh_energy /= 6.0f;

        coh_energy = sum_r*sum_r + sum_i*sum_i;
        
        pcf = coh_energy / (incoh_energy + 1e-12f);
        if (pcf > 1.0f) pcf = 1.0f;

        gain = pcf * pcf * pcf;
        G_smooth[k] = 0.6f * G_smooth[k] + 0.4f * gain;

        out_r[k] = sum_r * G_smooth[k] * 2.5f; 
        out_i[k] = sum_i * G_smooth[k] * 2.5f;
        out_r[FFT_N - k] = out_r[k];
        out_i[FFT_N - k] = -out_i[k];
    }

    fft_radix2(out_r, out_i, FFT_N, 1);

    for (i = 0; i < FFT_N; i++) {
        y = out_r[i] - prev_x + 0.995f * prev_y;
        prev_x = out_r[i];
        prev_y = y;

        amplified = y * 1.5f; 
        if(amplified > 32767.0f) amplified = 32767.0f;
        if(amplified < -32768.0f) amplified = -32768.0f;
        pcm_out[i] = (int16_t)amplified;
    }

    tuple[0] = mp_obj_new_float(target_angle);
    tuple[1] = mp_obj_new_bytes((const byte*)pcm_out, FFT_N * sizeof(int16_t));
    return mp_obj_new_tuple(2, tuple);
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_beam_audio_obj, Maix_mic_array_get_beam_audio);

STATIC mp_obj_t Maix_mic_array_track_doa(void) {
    static float local_mic_data[NUM_MICS][FFT_N];
    int m, i; float new_angle;

    if (shared_data_ready == 0) return mp_obj_new_float(current_target_angle);
    for(m=0; m<NUM_MICS; m++) { for(i=0; i<FFT_N; i++) local_mic_data[m][i] = shared_doa_mic_data[m][i]; }
    shared_data_ready = 0; 

    MP_THREAD_GIL_EXIT(); new_angle = run_doa_pipeline(local_mic_data); MP_THREAD_GIL_ENTER();
    
    current_target_angle = new_angle;
    return mp_obj_new_float(new_angle);
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_track_doa_obj, Maix_mic_array_track_doa);

STATIC mp_obj_t Maix_mic_array_set_led(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    int index, brightness[12] = {0}, led_color[12] = {0}, color[3] = {0}; mp_obj_t *items; uint32_t set_color;
    mp_obj_get_array_fixed_n(pos_args[0], 12, &items);
    for(index= 0; index < 12; index++) brightness[index] = mp_obj_get_int(items[index]);
    mp_obj_get_array_fixed_n(pos_args[1], 3, &items);
    for(index = 0; index < 3; index++) color[index] = mp_obj_get_int(items[index]);
    set_color = (color[2] << 16) | (color[1] << 8) | (color[0]);
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
