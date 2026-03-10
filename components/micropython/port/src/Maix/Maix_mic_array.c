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
// 第一部分：全局变量与阵列参数
// ============================================================================
#define FFT_N 1024
#define NUM_MICS 7       
#define NUM_PAIRS 21     
#define SAMPLE_RATE 16000.0f  // 严格设定为 16000Hz
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

// 【新增】：MVDR 专用的空间协方差矩阵 (SCM)，静态分配至 BSS 段防止爆栈 (大约占用 200KB)
static float R_mat_r[FFT_N/2][NUM_MICS][NUM_MICS];
static float R_mat_i[FFT_N/2][NUM_MICS][NUM_MICS];
static int mvdr_initialized = 0;

static void init_array_geometry(void) {
    float R = 0.04f;
    float theta_mic[6] = {0, 60, 120, 180, 240, 300};
    float mic_pos_2d[NUM_MICS][2];
    int i, j, k = 0;

    for(i = 0; i < 6; i++) {
        mic_pos_2d[i][0] = R * cosf(theta_mic[i] * DEG2RAD);
        mic_pos_2d[i][1] = R * sinf(theta_mic[i] * DEG2RAD);
    }
    mic_pos_2d[6][0] = 0.0f; mic_pos_2d[6][1] = 0.0f;

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

// 【新增】：7x7 复数矩阵求逆 (Gauss-Jordan消元法)，用于 MVDR 核心计算
static int invert_7x7_complex(float mat_r[NUM_MICS][NUM_MICS], float mat_i[NUM_MICS][NUM_MICS], 
                              float inv_r[NUM_MICS][NUM_MICS], float inv_i[NUM_MICS][NUM_MICS]) {
    int i, j, k;
    float tr, ti, diag_r, diag_i, mag2, f_r, f_i;
    for(i=0; i<NUM_MICS; i++){
        for(j=0; j<NUM_MICS; j++){
            inv_r[i][j] = (i==j) ? 1.0f : 0.0f;
            inv_i[i][j] = 0.0f;
        }
    }
    for(i=0; i<NUM_MICS; i++){
        diag_r = mat_r[i][i];
        diag_i = mat_i[i][i];
        mag2 = diag_r*diag_r + diag_i*diag_i;
        if(mag2 < 1e-12f) return -1; // 矩阵奇异保护
        
        for(j=0; j<NUM_MICS; j++){
            tr = mat_r[i][j]; ti = mat_i[i][j];
            mat_r[i][j] = (tr*diag_r + ti*diag_i)/mag2;
            mat_i[i][j] = (ti*diag_r - tr*diag_i)/mag2;
            
            tr = inv_r[i][j]; ti = inv_i[i][j];
            inv_r[i][j] = (tr*diag_r + ti*diag_i)/mag2;
            inv_i[i][j] = (ti*diag_r - tr*diag_i)/mag2;
        }
        for(k=0; k<NUM_MICS; k++){
            if(k == i) continue;
            f_r = mat_r[k][i]; f_i = mat_i[k][i];
            for(j=0; j<NUM_MICS; j++){
                mat_r[k][j] -= (f_r*mat_r[i][j] - f_i*mat_i[i][j]);
                mat_i[k][j] -= (f_r*mat_i[i][j] + f_i*mat_r[i][j]);
                
                inv_r[k][j] -= (f_r*inv_r[i][j] - f_i*inv_i[i][j]);
                inv_i[k][j] -= (f_r*inv_i[i][j] + f_i*inv_r[i][j]);
            }
        }
    }
    return 0;
}

// ============================================================================
// 第二部分：DOA 追踪保留（为了兼容之前的卡尔曼平滑框架）
// ============================================================================
static float calculate_median(float* array, int size) {
    float temp[NUM_PAIRS]; int i, j; float swap;
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

static float run_doa_pipeline(float mic_data[NUM_MICS][FFT_N]) {
    // 省略：这部分逻辑与你提供的代码完全一致，保持原样即可
    // 为了回复不超长且避免干扰，DOA 估计算法维持你的 GCC-PHAT + 卡尔曼滤波框架
    // ... [DOA 内部代码保持与你提供的一致] ...
    return 0.0f; // 实际请保留你原代码中的 DOA 逻辑
}

// ============================================================================
// 第三部分：核心 API (频域 Robust MVDR 重写)
// ============================================================================
static int i2s_dma_cb(void *ctx) { rx_flag = 1; return 0; }

STATIC mp_obj_t Maix_mic_array_init(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    // ... [初始化代码保持一致] ...
    enum { ARG_i2s_d0, ARG_i2s_d1, ARG_i2s_d2, ARG_i2s_d3, ARG_i2s_ws, ARG_i2s_sclk, ARG_sk9822_dat, ARG_sk9822_clk, };
    static const mp_arg_t allowed_args[]={{MP_QSTR_i2s_d0, MP_ARG_INT, {.u_int = 23}}, {MP_QSTR_i2s_d1, MP_ARG_INT, {.u_int = 22}}, {MP_QSTR_i2s_d2, MP_ARG_INT, {.u_int = 21}}, {MP_QSTR_i2s_d3, MP_ARG_INT, {.u_int = 20}}, {MP_QSTR_i2s_ws, MP_ARG_INT, {.u_int = 19}}, {MP_QSTR_i2s_sclk, MP_ARG_INT, {.u_int = 18}}, {MP_QSTR_sk9822_dat, MP_ARG_INT, {.u_int = 24}}, {MP_QSTR_sk9822_clk, MP_ARG_INT, {.u_int = 25}},};
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)]; 
    int i;
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

// ============================================================================
// 【算法核心】：鲁棒 MVDR 频域滤波引擎
// ============================================================================
STATIC mp_obj_t Maix_mic_array_get_beam_audio(void) {
    volatile uint8_t retry = 100;
    static float mic_r[NUM_MICS][FFT_N], mic_i[NUM_MICS][FFT_N];
    static float out_r[FFT_N], out_i[FFT_N];
    static int16_t pcm_out[FFT_N];
    
    int i, m, n, k;
    float target_angle, cos_t, sin_t, freq, omega, tau, phase, bin_energy, rho;
    float temp_r[NUM_MICS][NUM_MICS], temp_i[NUM_MICS][NUM_MICS];
    float inv_r[NUM_MICS][NUM_MICS], inv_i[NUM_MICS][NUM_MICS];
    float a_r[NUM_MICS], a_i[NUM_MICS], v_r[NUM_MICS], v_i[NUM_MICS], w_r[NUM_MICS], w_i[NUM_MICS];
    float denom_r, denom_i, out_val_r, out_val_i, cross_r, cross_i;
    
    // 算法超参数（可用于写论文的调参依据）
    const float ALPHA = 0.8f;      // 协方差矩阵的遗忘因子 (控制自适应速度)
    const float DL_FACTOR = 0.05f; // 对角加载系数 (增强小孔径的鲁棒性)
    
    float mx[7] = {0.04f*cosf(0), 0.04f*cosf(60*DEG2RAD), 0.04f*cosf(120*DEG2RAD), 0.04f*cosf(180*DEG2RAD), 0.04f*cosf(240*DEG2RAD), 0.04f*cosf(300*DEG2RAD), 0.0f};
    float my[7] = {0.04f*sinf(0), 0.04f*sinf(60*DEG2RAD), 0.04f*sinf(120*DEG2RAD), 0.04f*sinf(180*DEG2RAD), 0.04f*sinf(240*DEG2RAD), 0.04f*sinf(300*DEG2RAD), 0.0f};

    mp_obj_t tuple[2];

    while(rx_flag == 0) { retry--; msleep(1); if(retry == 0) break; }
    if(rx_flag == 0 && retry == 0) { mp_raise_OSError(MP_ETIMEDOUT); return mp_const_false; }
    rx_flag = 0;

    // 1. 数据对齐与预处理
    for (i = 0; i < FFT_N; i++) {
        mic_r[0][i] = (float)(i2s_rx_buf[i*8 + 0] >> 16);
        mic_r[1][i] = (float)(i2s_rx_buf[i*8 + 1] >> 16);
        mic_r[2][i] = (float)(i2s_rx_buf[i*8 + 2] >> 16);
        mic_r[3][i] = (float)(i2s_rx_buf[i*8 + 3] >> 16);
        mic_r[4][i] = (float)(i2s_rx_buf[i*8 + 4] >> 16);
        mic_r[5][i] = (float)(i2s_rx_buf[i*8 + 5] >> 16);
        mic_r[6][i] = (float)(i2s_rx_buf[i*8 + 6] >> 16); 
        for(m=0; m<NUM_MICS; m++) mic_i[m][i] = 0.0f;
    }
    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);

    // SCM 初始化矩阵
    if (!mvdr_initialized) {
        for(k = 0; k < FFT_N/2; k++) {
            for(m = 0; m < NUM_MICS; m++) {
                for(n = 0; n < NUM_MICS; n++) {
                    R_mat_r[k][m][n] = (m == n) ? 1.0f : 0.0f;
                    R_mat_i[k][m][n] = 0.0f;
                }
            }
        }
        mvdr_initialized = 1;
    }

    target_angle = current_target_angle;
    cos_t = cosf(target_angle * DEG2RAD);
    sin_t = sinf(target_angle * DEG2RAD);

    // 2. 变换到频域
    for(m = 0; m < NUM_MICS; m++) {
        fft_radix2(mic_r[m], mic_i[m], FFT_N, 0);
    }

    out_r[0] = 0; out_i[0] = 0;             // 去除直流分量
    out_r[FFT_N/2] = 0; out_i[FFT_N/2] = 0; // 去除奈奎斯特频率

    // 3. 频带级 Robust MVDR 处理
    for (k = 1; k < FFT_N/2; k++) {
        // --- 核心 A：Tyler M-估计对抗脉冲噪声 ---
        bin_energy = 0;
        for(m = 0; m < NUM_MICS; m++) {
            bin_energy += (mic_r[m][k]*mic_r[m][k] + mic_i[m][k]*mic_i[m][k]);
        }
        rho = NUM_MICS / (bin_energy + 1e-6f); // 计算 M-估计惩罚权重

        // 更新空间协方差矩阵 (SCM) 并应用对角加载 (DL)
        for(m = 0; m < NUM_MICS; m++) {
            for(n = 0; n < NUM_MICS; n++) {
                cross_r = mic_r[m][k]*mic_r[n][k] + mic_i[m][k]*mic_i[n][k]; 
                cross_i = mic_i[m][k]*mic_r[n][k] - mic_r[m][k]*mic_i[n][k];
                R_mat_r[k][m][n] = ALPHA * R_mat_r[k][m][n] + (1.0f - ALPHA) * rho * cross_r;
                R_mat_i[k][m][n] = ALPHA * R_mat_i[k][m][n] + (1.0f - ALPHA) * rho * cross_i;
                
                temp_r[m][n] = R_mat_r[k][m][n];
                temp_i[m][n] = R_mat_i[k][m][n];
            }
            temp_r[m][m] += DL_FACTOR; // 注入白噪声，打破奇异性
        }

        // --- 核心 B：计算导向矢量与权重求逆 ---
        freq = (float)k * SAMPLE_RATE / FFT_N;
        omega = 2.0f * PI * freq;
        for(m = 0; m < NUM_MICS; m++) {
            tau = (mx[m]*cos_t + my[m]*sin_t) / SOUND_SPEED;
            phase = omega * tau;
            a_r[m] = cosf(phase);  
            a_i[m] = -sinf(phase); 
        }

        // 计算矩阵的逆 R^-1
        invert_7x7_complex(temp_r, temp_i, inv_r, inv_i);

        // 计算分子：V = R^-1 * a
        for(m = 0; m < NUM_MICS; m++) {
            v_r[m] = 0; v_i[m] = 0;
            for(n = 0; n < NUM_MICS; n++) {
                v_r[m] += inv_r[m][n]*a_r[n] - inv_i[m][n]*a_i[n];
                v_i[m] += inv_r[m][n]*a_i[n] + inv_i[m][n]*a_r[n];
            }
        }

        // 计算分母：denom = a^H * V
        denom_r = 0; denom_i = 0;
        for(m = 0; m < NUM_MICS; m++) {
            denom_r += a_r[m]*v_r[m] + a_i[m]*v_i[m];
            denom_i += a_r[m]*v_i[m] - a_i[m]*v_r[m];
        }

        // 计算最优权重：W = V / denom
        for(m = 0; m < NUM_MICS; m++) {
            w_r[m] = (v_r[m]*denom_r + v_i[m]*denom_i) / (denom_r*denom_r + denom_i*denom_i + 1e-12f);
            w_i[m] = (v_i[m]*denom_r - v_r[m]*denom_i) / (denom_r*denom_r + denom_i*denom_i + 1e-12f);
        }

        // --- 核心 C：应用空间滤波权重 ---
        out_val_r = 0; out_val_i = 0;
        for(m = 0; m < NUM_MICS; m++) {
            // Y = W^H * X
            out_val_r += w_r[m]*mic_r[m][k] + w_i[m]*mic_i[m][k];
            out_val_i += w_r[m]*mic_i[m][k] - w_i[m]*mic_r[m][k];
        }
        
        // 保存共轭对称数据，为安全的傅里叶逆变换准备
        out_r[k] = out_val_r; 
        out_i[k] = out_val_i;
        out_r[FFT_N - k] = out_val_r; 
        out_i[FFT_N - k] = -out_val_i;
    }

    // 4. 重构回时域信号
    fft_radix2(out_r, out_i, FFT_N, 1);

    // 5. 增益控制与量化输出
    for(i = 0; i < FFT_N; i++) {
        float amplified = out_r[i] * 1.5f; // 可以根据需要调节整体音量
        if(amplified > 32767.0f) amplified = 32767.0f;
        if(amplified < -32768.0f) amplified = -32768.0f;
        pcm_out[i] = (int16_t)amplified;
    }

    tuple[0] = mp_obj_new_float(target_angle);
    tuple[1] = mp_obj_new_bytes((const byte*)pcm_out, FFT_N * sizeof(int16_t));
    return mp_obj_new_tuple(2, tuple);
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_beam_audio_obj, Maix_mic_array_get_beam_audio);

// ... [Track DOA 和 LED 代码保留原状] ...
STATIC mp_obj_t Maix_mic_array_track_doa(void) {
    // 省略内部代码，保持原状...
    return mp_obj_new_float(current_target_angle);
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_track_doa_obj, Maix_mic_array_track_doa);

STATIC mp_obj_t Maix_mic_array_set_led(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    // 省略内部代码，保持原状...
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
