#ifndef _M_ESTIMATION_H_
#define _M_ESTIMATION_H_

#include <stdint.h>

#define FFT_N 1024
#define NUM_MICS 7
#define NUM_PAIRS 21
#define SAMPLE_RATE 48000.0f
#define SOUND_SPEED 343.0f

// 初始化阵列物理拓扑 (系统启动时调用)
void init_array_geometry(void);

// 核心流水线：传入 7个麦克风的 1024点 Float 音频数据，输出最终的 DoA 角度
float run_doa_pipeline(float mic_data[NUM_MICS][FFT_N]);

#endif
