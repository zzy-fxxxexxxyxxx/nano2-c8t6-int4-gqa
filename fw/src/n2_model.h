// n2_model.h —— nano2 int4 + GQA + 滑窗注意力的推理核
#ifndef N2_MODEL_H
#define N2_MODEL_H

#include <stdint.h>

// 复位 KV 环形窗口与位置计数（每次提问前调用）
void N2_Reset(void);

// 生成：prompt 为 "q:...\na:" 字节序列；返回写入 out 的字节数
int N2_Generate(const uint8_t *prompt, int plen, uint8_t *out, int max_out,
                void (*on_byte)(uint8_t), void (*on_tick)(void));

// 诊断：最近一次前向的 top-n token id
int N2_DebugTop(uint8_t *ids, int n);

#endif // N2_MODEL_H
