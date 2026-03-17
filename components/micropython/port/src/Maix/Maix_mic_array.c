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
// 第一部分：M-估计 DOA 算法的核心宏定义、结构体与全局变量
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
    int i, j, k;

    for(i = 0; i < 6; i++) {
        mic_pos_2d[i][0] = R * cosf(theta_mic[i] * DEG2RAD);
        mic_pos_2d[i][1] = R * sinf(theta_mic[i] * DEG2RAD);
    }
    mic_pos_2d[6][0] = 0.0f; 
    mic_pos_2d[6][1] = 0.0f;

    k = 0;
    for (i = 0; i < NUM_MICS; i++) {
        for (j = i + 1; j < NUM_MICS; j++) {
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
    int i, j, k, l, p;
    float tx, ty, u1, u2, z, cos_z, sin_z, t;

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
        cos_z = cosf(z); 
        sin_z = sinf(z);
        for (j = 0; j < l; j++) {
            for (i = j; i < n; i += 2 * l) {
                p = i + l;
                tx = real[p] * u1 - imag[p] * u2;
                ty = real[p] * u2 + imag[p] * u1;
                real[p] = real[i] - tx; 
                imag[p] = imag[i] - ty;
                real[i] += tx;          
                imag[i] += ty;
            }
            t = u1;
            u1 = t * cos_z - u2 * sin_z;
            u2 = t * sin_z + u2 * cos_z;
        }
    }
    if (is_inverse) {
        for (i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; }
    }
}

static float calculate_median(float* array, int size) {
    float temp[NUM_PAIRS];
    int i, j;
    float swap;

    for(i = 0; i < size; i++) temp[i] = array[i];
    for (i = 0; i < size - 1; i++) {
        for (j = 0; j < size - i - 1; j++) {
            if (temp[j] > temp[j + 1]) {
                swap = temp[j]; 
                temp[j] = temp[j + 1]; 
                temp[j + 1] = swap;
            }
        }
    }
    return temp[size / 2];
}

// 4. M-估计核心测向流水线
static float run_doa_pipeline(float mic_data[NUM_MICS][FFT_N]) {
    static float mic_real[NUM_MICS][FFT_N]; 
    static float mic_imag[NUM_MICS][FFT_N]; 
    static float R_real[FFT_N], R_imag[FFT_N];
    static float s_last_valid_angle = 0.0f;
    static int s_doa_initialized = 0;
    
    // 严格 C89 变量声明
    int m, i, u, v, k, theta, p, iter;
    int search_range, max_idx, valid_count;
    float window, cross_r, cross_i, mag, max_val, delta, y1, y2, y3, denom, tau_samples;
    float min_cost, ang_coarse, cos_t, sin_t, cost, theo_tdoa, err;
    float cos_coarse, sin_coarse, med_res_m, sigma_adaptive_m, sigma_adaptive_t;
    float res_sq, w_consist;
    float ang_gn, sum_WJr, sum_WJJ, cos_gn, sin_gn, r_p, J_p, delta_ang;
    float current_ang;
    
    // 论文“最后一道防线”相关变量
    float total_weight, normalized_confidence;
    float CONFIDENCE_THRESHOLD = 0.15f; // 安全下限阈值，可调

    float Meas_TDOA[NUM_PAIRS] = {0};
    float Qual_Score[NUM_PAIRS] = {0};
    float raw_residuals_t[NUM_PAIRS] = {0};
    float Final_W[NUM_PAIRS] = {0};

    // 加窗与正向 FFT
    for (m = 0; m < NUM_MICS; m++) {
        for (i = 0; i < FFT_N; i++) {
            window = 0.54f - 0.46f * cosf(2.0f * PI * i / (FFT_N - 1));
            mic_real[m][i] = mic_data[m][i] * window;
            mic_imag[m][i] = 0.0f;
        }
        fft_radix2(mic_real[m], mic_imag[m], FFT_N, 0);
    }

    // 1024 点 GCC-PHAT 计算
    k = 0;
    for (u = 0; u < NUM_MICS; u++) {
        for (v = u + 1; v < NUM_MICS; v++) {
            for (i = 0; i < FFT_N; i++) {
                cross_r = mic_real[u][i] * mic_real[v][i] + mic_imag[u][i] * mic_imag[v][i];
                cross_i = mic_imag[u][i] * mic_real[v][i] - mic_real[u][i] * mic_imag[v][i];
                mag = sqrtf(cross_r * cross_r + cross_i * cross_i) + 1e-9f;
                R_real[i] = cross_r / mag;
                R_imag[i] = cross_i / mag;
            }
            fft_radix2(R_real, R_imag, FFT_N, 1);

            max_val = 0; 
            max_idx = 0;
            search_range = 8; 
            for (i = 0; i <= search_range; i++) {
                if (R_real[i] > max_val) { max_val = R_real[i]; max_idx = i; }
            }
            for (i = FFT_N - search_range; i < FFT_N; i++) {
                if (R_real[i] > max_val) { max_val = R_real[i]; max_idx = i; }
            }

            delta = 0;
            if (max_idx > 0 && max_idx < FFT_N - 1) {
                y1 = R_real[max_idx - 1];
                y2 = R_real[max_idx];
                y3 = R_real[max_idx + 1];
                denom = 2.0f * (y1 - 2.0f * y2 + y3);
                if (fabsf(denom) > 1e-9f) delta = (y1 - y3) / denom;
            }
            
            tau_samples = (max_idx > FFT_N/2) ? (max_idx - FFT_N + delta) : (max_idx + delta);
            Meas_TDOA[k] = tau_samples / SAMPLE_RATE;
            
            // 严格执行你的原始论文 Sigmoid 评分
            Qual_Score[k] = 1.0f / (1.0f + expf(-15.0f * (max_val - 0.15f)));
            k++;
        }
    }

    // 粗网格搜索
    min_cost = FLT_MAX;
    ang_coarse = 0.0f;
    for (theta = -180; theta < 180; theta++) {
        cost = 0.0f;
        cos_t = cosf(theta * DEG2RAD);
        sin_t = sinf(theta * DEG2RAD);
        for (p = 0; p < NUM_PAIRS; p++) {
            if (Qual_Score[p] < 1e-3f) continue;
            theo_tdoa = (pair_conf[p].dx * cos_t + pair_conf[p].dy * sin_t) / SOUND_SPEED;
            err = theo_tdoa - Meas_TDOA[p];
            cost += Qual_Score[p] * pair_conf[p].dist * err * err;
        }
        if (cost < min_cost) { min_cost = cost; ang_coarse = (float)theta; }
    }

    // 自适应尺度参数计算 (自适应 sigma_adapt)
    cos_coarse = cosf(ang_coarse * DEG2RAD);
    sin_coarse = sinf(ang_coarse * DEG2RAD);
    for (p = 0; p < NUM_PAIRS; p++) {
        theo_tdoa = (pair_conf[p].dx * cos_coarse + pair_conf[p].dy * sin_coarse) / SOUND_SPEED;
        raw_residuals_t[p] = fabsf(Meas_TDOA[p] - theo_tdoa);
    }
    med_res_m = calculate_median(raw_residuals_t, NUM_PAIRS) * SOUND_SPEED;
    sigma_adaptive_m = med_res_m * 1.5f;
    if (sigma_adaptive_m < 0.015f) sigma_adaptive_m = 0.015f;
    if (sigma_adaptive_m > 0.10f) sigma_adaptive_m = 0.10f;
    sigma_adaptive_t = sigma_adaptive_m / SOUND_SPEED;

    // 计算最终一致性权重 W_p
    for (p = 0; p < NUM_PAIRS; p++) {
        res_sq = raw_residuals_t[p] * raw_residuals_t[p];
        w_consist = expf(-res_sq / (2.0f * sigma_adaptive_t * sigma_adaptive_t));
        Final_W[p] = Qual_Score[p] * ((1.0f - 0.2f) * w_consist + 0.2f) * sqrtf(pair_conf[p].dist);
    }

    // =====================================================================
    // 论文 3.3 节：抵抗野值雪崩的“最后一道防线”
    // 计算有效麦克风对的归一化权重之和
    // =====================================================================
    total_weight = 0.0f;
    for (p = 0; p < NUM_PAIRS; p++) {
        total_weight += Final_W[p];
    }
    normalized_confidence = total_weight / NUM_PAIRS;

    // 若置信度低于预设下限，标记为不可靠，直接拦截，拒绝输出乱跳的角度
    if (normalized_confidence < CONFIDENCE_THRESHOLD && s_doa_initialized) {
        return s_last_valid_angle; 
    }

    // Gauss-Newton 微调 (只在置信度达标时执行，节省算力)
    ang_gn = ang_coarse;
    for (iter = 0; iter < 8; iter++) {
        sum_WJr = 0.0f;
        sum_WJJ = 0.0f;
        valid_count = 0;
        cos_gn = cosf(ang_gn * DEG2RAD);
        sin_gn = sinf(ang_gn * DEG2RAD);
        for (p = 0; p < NUM_PAIRS; p++) {
            if (Final_W[p] < 1e-4f) continue;
            valid_count++;
            theo_tdoa = (pair_conf[p].dx * cos_gn + pair_conf[p].dy * sin_gn) / SOUND_SPEED;
            r_p = theo_tdoa - Meas_TDOA[p];
            J_p = DEG2RAD * (-pair_conf[p].dx * sin_gn + pair_conf[p].dy * cos_gn) / SOUND_SPEED;
            sum_WJr += Final_W[p] * J_p * r_p;
            sum_WJJ += Final_W[p] * J_p * J_p;
        }
        if (valid_count < 3) break;
        delta_ang = -sum_WJr / (sum_WJJ + 1e-12f);
        ang_gn += delta_ang;
        if (fabsf(delta_ang) < 1e-3f) break;
    }
    
    // 角度规约
    ang_gn = fmodf(ang_gn + 180.0f, 360.0f);
    if (ang_gn < 0) ang_gn += 360.0f;
    current_ang = ang_gn - 180.0f;

    // 更新静态历史角度，供无声期拦截使用
    s_last_valid_angle = current_ang;
    s_doa_initialized = 1;

    return current_ang;
}

// ============================================================================
// 第三部分：MicroPython 底层接口绑定
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
    int i;
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
    for(i = 0; i < 4; i++){
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
    int i;
    float final_angle;
    
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

    for (i = 0; i < FFT_N; i++) {
        mic_raw_float[0][i] = (float)(i2s_rx_buf[i*8 + 0] >> 16);
        mic_raw_float[1][i] = (float)(i2s_rx_buf[i*8 + 1] >> 16);
        mic_raw_float[2][i] = (float)(i2s_rx_buf[i*8 + 2] >> 16);
        mic_raw_float[3][i] = (float)(i2s_rx_buf[i*8 + 3] >> 16);
        mic_raw_float[4][i] = (float)(i2s_rx_buf[i*8 + 4] >> 16);
        mic_raw_float[5][i] = (float)(i2s_rx_buf[i*8 + 5] >> 16);
        mic_raw_float[6][i] = (float)(i2s_rx_buf[i*8 + 6] >> 16);
    }

    i2s_receive_data_dma(I2S_DEVICE_0, (uint32_t *)i2s_rx_buf, FFT_N * 8, DMAC_CHANNEL4);

    final_angle = run_doa_pipeline(mic_raw_float);

    return mp_obj_new_float(final_angle);
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_get_dir_obj, Maix_mic_array_get_dir);

STATIC mp_obj_t Maix_mic_array_set_led(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args)
{
    int index, brightness[12] = {0}, led_color[12] = {0}, color[3] = {0};
    uint32_t set_color;
    mp_obj_t *items;
    
    mp_obj_get_array_fixed_n(pos_args[0], 12, &items);
    for(index= 0; index < 12; index++) brightness[index] = mp_obj_get_int(items[index]);

    mp_obj_get_array_fixed_n(pos_args[1], 3, &items);
    for(index = 0; index < 3; index++) color[index] = mp_obj_get_int(items[index]);

    set_color = (color[2] << 16) | (color[1] << 8) | (color[0]);

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
