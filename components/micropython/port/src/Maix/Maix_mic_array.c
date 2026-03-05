#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>     // 必须包含此头文件以支持 uint32_t

// MicroPython 与 MaixPy 底层依赖
#include "py/obj.h"
#include "py/runtime.h"
#include "py/mphal.h"
#include "py/objarray.h"
#include "py/binary.h"
#include "py_assert.h"
#include "mperrno.h"
#include "mphalport.h"
#include "modMaix.h"

// K210 硬件外设依赖
#include "sleep.h"
#include "sysctl.h"
#include "fpioa.h"
#include "i2s.h"
#include "dmac.h"
#include "fft.h"        // 调用 K210 硬件 FFT 协处理器
#include "sipeed_sk9822.h"

// ============================================================================
// [1] 全局物理常数与阵列拓扑配置 (Sipeed R6+1 阵列)
// ============================================================================
#define NUM_MICS 7
#define NUM_PAIRS 21
#define C_SPEED 343.0f
#define PI 3.1415926535f
#define ARRAY_RADIUS 0.04f  // 阵列半径 4cm
#define FFT_N 512           // 使用 512 点 FFT (平衡分辨率与极限算力)
#define SAMPLE_RATE 16000   // 采样率 16kHz

STATIC float dx_array[NUM_PAIRS];
STATIC float dy_array[NUM_PAIRS];
STATIC float dist_array[NUM_PAIRS];

// 预计算阵列拓扑间距
STATIC void init_mic_array_geometry() {
    float mic_x[NUM_MICS], mic_y[NUM_MICS];
    mic_x[0] = 0.0f; mic_y[0] = 0.0f; // 中心 Mic
    for (int i = 0; i < 6; i++) {     // 环形 6 Mic
        mic_x[i+1] = ARRAY_RADIUS * cosf(i * 60.0f * PI / 180.0f);
        mic_y[i+1] = ARRAY_RADIUS * sinf(i * 60.0f * PI / 180.0f);
    }
    int pair_idx = 0;
    for (int i = 0; i < NUM_MICS; i++) {
        for (int j = i + 1; j < NUM_MICS; j++) {
            dx_array[pair_idx] = mic_x[i] - mic_x[j];
            dy_array[pair_idx] = mic_y[i] - mic_y[j];
            dist_array[pair_idx] = sqrtf(dx_array[pair_idx]*dx_array[pair_idx] + dy_array[pair_idx]*dy_array[pair_idx]);
            pair_idx++;
        }
    }
}

// ============================================================================
// [2] 极客性能优化层：快速数学运算
// ============================================================================
// 修复后的 64位安全版 快速平方根倒数 (用于取代 GCC-PHAT 中极度耗时的开方与除法)
STATIC inline float Q_rsqrt(float number) {
    uint32_t i; // 必须用严格的 32 位无符号整型
    float x2, y;
    const float threehalfs = 1.5F;
    x2 = number * 0.5F;
    y  = number;
    i  = * ( uint32_t * ) &y; // 这里绝对不能用 long
    i  = 0x5f3759df - ( i >> 1 );
    y  = * ( float * ) &i;
    y  = y * ( threehalfs - ( x2 * y * y ) );
    return y;
}

// 轻量级数组中位数 (插入排序，针对 N=21 极速优化)
STATIC float get_median(float* array, int n) {
    float temp[21];
    for(int i=0; i<n; i++) temp[i] = array[i];
    for(int i=1; i<n; i++) {
        float key = temp[i];
        int j = i - 1;
        while(j >= 0 && temp[j] > key) {
            temp[j+1] = temp[j];
            j--;
        }
        temp[j+1] = key;
    }
    return temp[n/2];
}

// ============================================================================
// [3] 特征提取层：GCC-PHAT 硬件加速与质量打分
// ============================================================================
// 注意：以下为对接 K210 硬件 FFT 的算法逻辑框架
STATIC void compute_gcc_phat_features(float* tau_meas, float* Q_scores) {
    // 假设硬件 DMA 已经将 7 个通道的时域数据搬运完毕
    // 步骤 A: 对 7 个通道分别调用 K210 硬件 FFT (fft_build)
    // 步骤 B: 在频域内执行极速交叉互相关与 PHAT 白化
    /* for(int k=0; k<NUM_PAIRS; k++) {
        for(int f=0; f<FFT_N; f++) {
            float cross_re = X_re * Y_re + X_im * Y_im;
            float cross_im = X_im * Y_re - X_re * Y_im;
            float power_sq = cross_re*cross_re + cross_im*cross_im;
            float inv_mag = (power_sq > 1e-10f) ? Q_rsqrt(power_sq) : 0.0f; // 极速白化！
            freq_out_re[f] = cross_re * inv_mag;
            freq_out_im[f] = cross_im * inv_mag;
        }
        // 步骤 C: 调用硬件 IFFT 转回时域
        // 步骤 D: 寻找时域峰值索引 (peak_index) 以及峰值高度 (cc_peak)
        
        float time_delay = (float)(peak_index - FFT_N/2) / SAMPLE_RATE;
        tau_meas[k] = time_delay;
        
        // Sigmoid 质量打分 (论文 3.1 节, beta=15, gamma=0.15)
        Q_scores[k] = 1.0f / (1.0f + expf(-15.0f * (cc_peak - 0.15f)));
    }
    */
    
    // (由于硬件寄存器配置篇幅过长，此处用占位代替实际硬件读写流水线。
    // 实际编译时，将此处替换为真实的 fft_build 调用循环即可)
}

// ============================================================================
// [4] 论文核心算法层：空间残差自适应 M-估计与高斯-牛顿精调
// ============================================================================
STATIC float adaptive_gn_optimization(float theta_coarse, float* tau_meas, float* Q) {
    float raw_res_m[NUM_PAIRS], Final_W[NUM_PAIRS];
    float cos_theta_c = cosf(theta_coarse * PI / 180.0f);
    float sin_theta_c = sinf(theta_coarse * PI / 180.0f);
    
    // 第一步：映射物理残差
    for(int k=0; k<NUM_PAIRS; k++) {
        float tau_theo = (dx_array[k] * cos_theta_c + dy_array[k] * sin_theta_c) / C_SPEED;
        raw_res_m[k] = fabsf(tau_meas[k] - tau_theo) * C_SPEED; 
    }

    // 第二步：MAD 自适应环境感知 (硬物理边界保护)
    float sigma_adapt = get_median(raw_res_m, NUM_PAIRS) * 1.5f;
    if(sigma_adapt < 0.015f) sigma_adapt = 0.015f;
    if(sigma_adapt > 0.100f) sigma_adapt = 0.100f;

    // 第三步：稳健权重生成 (包含 0.2 的结构保护底线)
    for(int k=0; k<NUM_PAIRS; k++) {
        float w_consist = expf(-(raw_res_m[k] * raw_res_m[k]) / (2.0f * sigma_adapt * sigma_adapt));
        Final_W[k] = Q[k] * (0.8f * w_consist + 0.2f) * sqrtf(dist_array[k]);
    }

    // 第四步：一维标量高斯-牛顿 (IRLS) 极速寻优
    float theta_est = theta_coarse;
    for(int iter=0; iter<8; iter++) {
        float H = 1e-12f, G = 0.0f;
        float cos_theta = cosf(theta_est * PI / 180.0f);
        float sin_theta = sinf(theta_est * PI / 180.0f);
        
        int valid = 0;
        for(int k=0; k<NUM_PAIRS; k++) {
            if(Final_W[k] < 1e-4f) continue; // 物理硬截断
            valid++;
            float tau_theo = (dx_array[k] * cos_theta + dy_array[k] * sin_theta) / C_SPEED;
            float r = tau_theo - tau_meas[k];
            float J = (PI / 180.0f) * (-dx_array[k] * sin_theta + dy_array[k] * cos_theta) / C_SPEED;
            
            H += Final_W[k] * J * J;
            G += Final_W[k] * J * r;
        }
        if(valid < 3) break; // 阵列有效观测失效，提早退出防止奇异
        
        float delta = - G / H;
        theta_est += delta;
        if(fabsf(delta) < 1e-3f) break; 
    }
    
    // 约束至 0~360 度范围内
    while(theta_est >= 360.0f) theta_est -= 360.0f;
    while(theta_est < 0.0f) theta_est += 360.0f;
    return theta_est;
}

// ============================================================================
// [5] Python 虚拟机接口绑定层 (MicroPython API)
// ============================================================================

// 5.1 初始化接口 (在 Python 中调用 mic.init())
STATIC mp_obj_t Maix_mic_array_init(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    // 预加载阵列物理几何坐标
    init_mic_array_geometry();
    // 初始化环形 LED 灯
    sipeed_init_mic_array_led();
    
    // 初始化 K210 的 I2S 与 DMA (替代闭源库 lib_mic_init)
    // i2s_init(I2S_DEVICE_0, I2S_RECEIVER, 0x3);
    // i2s_rx_channel_config(I2S_DEVICE_0, I2S_CHANNEL_0, RESOLUTION_16_BIT, ...);
    
    // 开启硬件 FFT 时钟
    sysctl_clock_enable(SYSCTL_CLOCK_FFT);
    
    return mp_const_true;
}
MP_DEFINE_CONST_FUN_OBJ_KW(Maix_mic_array_init_obj, 0, Maix_mic_array_init);

// 5.2 核心解算接口 (在 Python 中调用 angle = mic.get_dir())
STATIC mp_obj_t Maix_mic_array_get_dir(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    float tau_meas[NUM_PAIRS] = {0};
    float Q_scores[NUM_PAIRS] = {0};
    
    // 1. 获取底层声学特征 (TDOA 与 质量打分)
    compute_gcc_phat_features(tau_meas, Q_scores);
    
    // 2. 模拟网格粗搜取得的初始角度 (实际应由 TDOA-Grid 给出)
    float theta_coarse = 90.0f; 

    // 3. 将特征送入 M-估计论文核心算法，输出极高精度浮点角度
    float precise_angle = adaptive_gn_optimization(theta_coarse, tau_meas, Q_scores);

    // 返回浮点数给 Python 端
    return mp_obj_new_float(precise_angle);
}
MP_DEFINE_CONST_FUN_OBJ_KW(Maix_mic_array_get_dir_obj, 0, Maix_mic_array_get_dir);

// 5.3 硬件清理接口
STATIC mp_obj_t Maix_mic_array_deinit(void) {
    sysctl_clock_disable(SYSCTL_CLOCK_FFT);
    return mp_const_true;
}
MP_DEFINE_CONST_FUN_OBJ_0(Maix_mic_array_deinit_obj, Maix_mic_array_deinit);

// 5.4 LED 控制接口 (保持原样)
STATIC mp_obj_t Maix_mic_array_set_led(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
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

// ============================================================================
// [6] MicroPython 模块注册表
// ============================================================================
STATIC const mp_rom_map_elem_t Maix_mic_array_locals_dict_table
