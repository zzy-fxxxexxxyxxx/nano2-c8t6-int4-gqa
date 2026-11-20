// n2_ui.h —— OLED 排版接口：6x8 ASCII + 12x12 汉字混排（字模见 n2_font.c）
#ifndef N2_UI_H
#define N2_UI_H

#include <stdint.h>

/* 从 (x, page) 开始排一段 UTF-8 文本：ASCII 步进 6（6x8），
   汉字步进 13（12x12，纵向跨两页）；一行放不下自动换行。 */
void N2UI_Text(uint8_t x, uint8_t page, const char *utf8);

void N2UI_Boot(void);
void N2UI_Header(const char *right);
void N2UI_Question(const char *q);
void N2UI_Answer(const char *a);
void N2UI_Thinking(uint8_t phase);

#endif
