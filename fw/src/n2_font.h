// n2_font.h —— 12x12 中文点阵字库（自动生成，勿手工修改）
//
// 生成脚本：nano2/tools/make_cjk_font.py
// 字体    ：Deng.ttf（12px 超采样 1x → 12x12 灰度 → 阈值 96）
// 字符集  ：79 个字（来源：charset.txt + 内置占位字符集）
//
// 字模用法：一个汉字纵向占 12 像素，跨 OLED 的两页；起始页存第 0..7 行，
// 下一页的低 4 位存第 8..11 行。绘制见 n2_ui.c 的 blit12()。
#ifndef N2_FONT_H
#define N2_FONT_H

#include <stdint.h>

#define N2FONT_W 12
#define N2FONT_H 12
#define N2FONT_STEP 13
#define N2FONT_COUNT 79

// Unicode 码位表（升序，N2Font_Lookup 用它做二分查找）
extern const uint16_t n2_font_codes[N2FONT_COUNT];
// 字模表：每字 24 字节 = 12 行 x 2 字节，每行高 12 位有效，bit15 是最左列
extern const uint8_t n2_font_bitmaps[][24];

// 查字模：命中返回该字的 24 字节字模，未收录返回 0
const uint8_t *N2Font_Lookup(uint16_t code);

#endif
