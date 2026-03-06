#include "m_estimation.h"
#include <math.h>
#include <float.h>

#define PI 3.14159265358979323846f
#define DEG2RAD (PI / 180.0f)

typedef struct { int u; int v; float dist; float dx; float dy; } MicPairConf;
MicPairConf pair_conf[NUM_PAIRS];

// --- 阵列初始化 (与上次代码相同) ---
void init_array_geometry(void) {
    float R = 0.04f;
    float theta_mic[6] = {0, 60, 120, 180, 240, 300};
    float mic_pos_2d[NUM_MICS][2];
    for(int i=0; i<6; i++) {
        mic_pos_2d[i][0] = R * cosf(theta_mic[i] * DEG2RAD);
        mic_pos_2d[i][1] = R * sinf(theta_mic[i] * DEG2RAD);
    }
    mic_pos_2d[6][0] = 0.0f; mic_pos_2d[6][1] = 0.0f;

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

// --- 纯 C 语言 1024 点基带 FFT (Cooley-Tukey) ---
void fft_radix2(float* real, float* imag, int n, int is_inverse) {
    int i, j, k, l;
    float tx, ty, u1, u2, z;
    // 位反转置换
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
    // 蝶形运算
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
                real[p] = real[i] - tx; imag[p] = imag[i] - ty;
                real[i] += tx;          imag[i] += ty;
            }
            float t = u1;
            u1 = t * cos_z - u2 * sin_z;
            u2 = t * sin_z + u2 * cos_z;
        }
    }
    // IFFT 缩放
    if (is_inverse) {
        for (i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; }
    }
}

// --- 辅助数学函数 ---
float calculate_median(float* array, int size) {
    float temp[NUM_PAIRS];
    for(int i=0; i<size; i++) temp[i] = array[i];
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (temp[j] > temp[j + 1]) {
                float swap = temp[j]; temp[j] = temp[j + 1]; temp[j + 1] = swap;
            }
        }
    }
    return temp[size / 2];
}

// --- 核心管线：跑满全流程 ---
float run_doa_pipeline(float mic_data[NUM_MICS][FFT_N]) {
    float mic_real[NUM_MICS][FFT_N];
    float mic_imag[NUM_MICS][FFT_N];
    
    // 1. 汉明窗 + 7 次正向 FFT
    for (int m = 0; m < NUM_MICS; m++) {
        for (int i = 0; i < FFT_N; i++) {
            float window = 0.54f - 0.46f * cosf(2.0f * PI * i / (FFT_N - 1));
            mic_real[m][i] = mic_data[m][i] * window;
            mic_imag[m][i] = 0.0f;
        }
        fft_radix2(mic_real[m], mic_imag[m], FFT_N, 0);
    }

    float Meas_TDOA[NUM_PAIRS] = {0};
    float Qual_Score[NUM_PAIRS] = {0};
    float R_real[FFT_N], R_imag[FFT_N];

    // 2. 21 组 GCC-PHAT
    int k = 0;
    for (int u = 0; u < NUM_MICS; u++) {
        for (int v = u + 1; v < NUM_MICS; v++) {
            for (int i = 0; i < FFT_N; i++) {
                // 互功率谱 (带频带限制 50Hz-8000Hz 略化为简单的全带计算，可根据需要加上频带掩码)
                float cross_r = mic_real[u][i] * mic_real[v][i] + mic_imag[u][i] * mic_imag[v][i];
                float cross_i = mic_imag[u][i] * mic_real[v][i] - mic_real[u][i] * mic_imag[v][i];
                float mag = sqrtf(cross_r * cross_r + cross_i * cross_i) + 1e-9f;
                R_real[i] = cross_r / mag;
                R_imag[i] = cross_i / mag;
            }
            // IFFT
            fft_radix2(R_real, R_imag, FFT_N, 1);

            // 寻找广义互相关峰值及抛物线插值
            float max_val = 0; int max_idx = 0;
            // 搜索限制在物理最大延迟内 (最大约 4cm / 343 * 48000 ≈ 6 个采样点)
            int search_range = 8; 
            for (int i = 0; i <= search_range; i++) {
                if (R_real[i] > max_val) { max_val = R_real[i]; max_idx = i; }
            }
            for (int i = FFT_N - search_range; i < FFT_N; i++) {
                if (R_real[i] > max_val) { max_val = R_real[i]; max_idx = i; }
            }

            float delta = 0;
            if (max_idx > 0 && max_idx < FFT_N - 1) {
                float y1 = R_real[max_idx - 1];
                float y2 = R_real[max_idx];
                float y3 = R_real[max_idx + 1];
                float denom = 2.0f * (y1 - 2.0f * y2 + y3);
                if (fabsf(denom) > 1e-9f) delta = (y1 - y3) / denom;
            }
            
            float tau_samples = (max_idx > FFT_N/2) ? (max_idx - FFT_N + delta) : (max_idx + delta);
            Meas_TDOA[k] = tau_samples / SAMPLE_RATE;
            Qual_Score[k] = 1.0f / (1.0f + expf(-15.0f * (max_val - 0.15f))); // 与 MATLAB 对齐
            k++;
        }
    }

    // 3. M-估计与高斯牛顿微调 (接力上次的代码)
    float min_cost = FLT_MAX;
    float ang_coarse = 0.0f;
    for (int theta = -180; theta < 180; theta++) {
        float cost = 0.0f;
        float cos_t = cosf(theta * DEG2RAD);
        float sin_t = sinf(theta * DEG2RAD);
        for (int p = 0; p < NUM_PAIRS; p++) {
            if (Qual_Score[p] < 1e-3f) continue;
            float theo_tdoa = (pair_conf[p].dx * cos_t + pair_conf[p].dy * sin_t) / SOUND_SPEED;
            float err = theo_tdoa - Meas_TDOA[p];
            cost += Qual_Score[p] * pair_conf[p].dist * err * err;
        }
        if (cost < min_cost) { min_cost = cost; ang_coarse = (float)theta; }
    }

    float raw_residuals_t[NUM_PAIRS];
    float cos_coarse = cosf(ang_coarse * DEG2RAD);
    float sin_coarse = sinf(ang_coarse * DEG2RAD);
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
        float sum_WJr = 0.0f, sum_WJJ = 0.0f;
        int valid_count = 0;
        float cos_gn = cosf(ang_gn * DEG2RAD), sin_gn = sinf(ang_gn * DEG2RAD);
        for (int p = 0; p < NUM_PAIRS; p++) {
            if (Final_W[p] < 1e-4f) continue;
            valid_count++;
            float theo_tdoa = (pair_conf[p].dx * cos_gn + pair_conf[p].dy * sin_gn) / SOUND_SPEED;
            float r_p = theo_tdoa - Meas_TDOA[p];
            float J_p = DEG2RAD * (-pair_conf[p].dx * sin_gn + pair_conf[p].dy * cos_gn) / SOUND_SPEED;
            sum_WJr += Final_W[p] * J_p * r_p;
            sum_WJJ += Final_W[p] * J_p * J_p;
        }
        if (valid_count < 3) break;
        float delta_ang = -sum_WJr / (sum_WJJ + 1e-12f);
        ang_gn += delta_ang;
        if (fabsf(delta_ang) < 1e-3f) break;
    }
    
    ang_gn = fmodf(ang_gn + 180.0f, 360.0f);
    if (ang_gn < 0) ang_gn += 360.0f;
    return ang_gn - 180.0f;
}
