// n2_ui.c —— 128x64 OLED 排版：页眉 + 问题 + 回答，支持 6x8 ASCII 与 12x12 中文混排
#include <string.h>

#include "OLED.h"
#include "n2_font.h"
#include "n2_ui.h"

#define COLS      21      /* 128 / 6：一行能放几个 6x8 ASCII */
#define ANS_PAGE  3
#define ANS_ROWS  5

/* ---------------- 基础绘制 ---------------- */

/* 清掉从 page 起的连续一页 */
static void page_clear(uint8_t page)
{
    uint8_t x;
    for (x = 0; x < 128; x++) OLED_Buffer[page][x] = 0;
}

/* 清掉 page0 起共 n 页（超出 7 页的部分忽略） */
static void clear_pages(uint8_t page0, uint8_t n)
{
    uint8_t p;
    for (p = 0; p < n; p++) {
        if ((uint8_t)(page0 + p) > 7) break;
        page_clear((uint8_t)(page0 + p));
    }
}

/* 只画可打印 ASCII 的一行字符串（页眉/开机画面用，位置固定不折行） */
static void put(uint8_t page, uint8_t col, const char *s)
{
    OLED_ShowString((uint8_t)(col * 6), page, s);
}

/* 把 12x12 字模按位写进帧缓冲，竖直方向跨两页：
   - 第 0..7 行 -> OLED_Buffer[page] 的 bit0..bit7（bit0 在最上面）
   - 第 8..11 行 -> OLED_Buffer[page+1] 的 bit0..bit3
   字模每行 2 字节、共 12 行，bit15 是最左列（高位在左）。
   用 |= 叠加，调用方保证目标页已经清过。 */
static void blit12(uint8_t x, uint8_t page, const uint8_t *glyph)
{
    uint8_t col, row;

    for (col = 0; col < 12; col++) {
        uint8_t mask = (uint8_t)(0x80u >> (col & 7)); /* 本列在 2 字节里的位 */
        uint8_t idx = (uint8_t)(col >> 3);            /* 0：前 8 列，1：后 4 列 */
        uint8_t top = 0;                              /* 写 page 的一列 */
        uint8_t bot = 0;                              /* 写 page+1 的一列 */

        if ((uint16_t)x + col > 127) break;           /* 右边越界，剩下的列不画 */

        for (row = 0; row < 8; row++) {
            if (glyph[row * 2 + idx] & mask) top |= (uint8_t)(1u << row);
        }
        for (row = 8; row < 12; row++) {
            if (glyph[row * 2 + idx] & mask) bot |= (uint8_t)(1u << (row - 8));
        }

        OLED_Buffer[page][x + col] |= top;
        if (page < 7) OLED_Buffer[page + 1][x + col] |= bot; /* 末页没有下一页 */
    }
}

/* ---------------- UTF-8 ---------------- */

/* 一个 UTF-8 字符占几字节：1=ASCII，2/3=常用汉字，0=4 字节或不合法 */
static uint8_t utf8_len(unsigned char c)
{
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    return 0;
}

/* 解出一个码位并让 *ps 前进；返回 0 表示结束。
   坏字节只跳一个继续解析，避免死循环；BMP 以外的字返回 0xFFFF（字库不会有）。 */
static int utf8_next(const char **ps, uint16_t *code)
{
    const unsigned char *s = (const unsigned char *)*ps;

    while (*s) {
        uint8_t n = utf8_len(*s);
        uint8_t k;

        if (n == 0) {
            /* 4 字节字符：整段跳过（码表是 uint16_t，收不了） */
            if ((*s & 0xF8) == 0xF0 && (s[1] & 0xC0) == 0x80 &&
                (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
                s += 4;
                *ps = (const char *)s;
                *code = 0xFFFF;
                return 1;
            }
            s++;
            continue;
        }
        for (k = 1; k < n; k++) {
            if ((s[k] & 0xC0) != 0x80) break;   /* 后续字节不合法，当坏字节处理 */
        }
        if (k < n) {
            s++;
            continue;
        }

        if (n == 1) *code = s[0];
        else if (n == 2) *code = (uint16_t)(((s[0] & 0x1Fu) << 6) | (s[1] & 0x3Fu));
        else *code = (uint16_t)(((s[0] & 0x0Fu) << 12) | ((s[1] & 0x3Fu) << 6) |
                                (s[2] & 0x3Fu));
        s += n;
        *ps = (const char *)s;
        return 1;
    }

    *ps = (const char *)s;
    return 0;
}

/* 按整字符把 src 拷进 dst（最多 cap 字节），不会把汉字截成半个；返回写入的字节数 */
static uint8_t copy_utf8(char *dst, const char *src, uint8_t cap)
{
    uint8_t n = 0;

    while (*src) {
        uint8_t len = utf8_len((unsigned char)*src);
        uint8_t i;

        if (len == 0) len = 1;                       /* 坏字节按 1 字节拷，后面解码时再跳 */
        if ((uint16_t)n + len > cap) break;
        for (i = 0; i < len; i++) dst[n + i] = src[i];
        n = (uint8_t)(n + len);
        src += len;
    }
    dst[n] = 0;
    return n;
}

/* ---------------- 混排 ---------------- */

/* 从 (x, page) 开始排一段 UTF-8 文本：
   - ASCII 走 6x8 字模，步进 6 像素、占 1 页
   - 汉字走 12x12 字模，步进 13 像素（12 + 1 间距）、占 2 页
   - 一行放不下就换行：ASCII 行下移 1 页，含汉字的行下移 2 页
   - 起始页超过 last_page 就停，留在屏幕外的不画 */
static void draw_text(uint8_t x, uint8_t page, uint8_t last_page, const char *utf8)
{
    const char *s = utf8;
    uint16_t code;
    uint8_t line_pages = 1;      /* 当前这一行占了几页 */

    while (utf8_next(&s, &code)) {
        uint8_t w, h;

        if (code < 0x80) {
            if (code < 0x20 || code > 0x7E) continue;   /* 控制字符不画，避免字模越界 */
            w = 6;
            h = 1;
        } else {
            w = N2FONT_STEP;
            h = 2;
        }

        if ((uint16_t)x + w > 128) {                    /* 本行放不下 -> 换行 */
            page = (uint8_t)(page + line_pages);
            x = 0;
            line_pages = 1;
        }
        if ((uint16_t)page + h - 1 > last_page) return; /* 已经到可用区域底部 */

        if (h == 1) {
            OLED_ShowChar(x, page, (char)code);
        } else {
            const uint8_t *g = N2Font_Lookup(code);
            if (g == 0) continue;                       /* 字库没收录：跳过，不留空位 */
            blit12(x, page, g);
        }

        x = (uint8_t)(x + w);
        if (h > line_pages) line_pages = h;
    }
}

/* ---------------- 对外接口 ---------------- */

void N2UI_Text(uint8_t x, uint8_t page, const char *utf8)
{
    draw_text(x, page, 7, utf8);
}

void N2UI_Header(const char *right)
{
    page_clear(0);
    put(0, 0, "nano2");
    if (right && *right) {
        uint8_t len = (uint8_t)strlen(right);
        if (len > COLS - 6) { right += len - (COLS - 6); len = COLS - 6; }
        put(0, (uint8_t)(COLS - len), right);
    }
    OLED_Update();
}

void N2UI_Boot(void)
{
    uint8_t p;
    for (p = 0; p < 8; p++) page_clear(p);
    put(0, 0, "nano2");
    put(0, COLS - 4, "boot");
    put(2, 0, "int4 gqa win64");
    put(3, 0, "q: who is dana li");
    put(4, 0, "a: ...");
    OLED_Update();
}

void N2UI_Question(const char *q)
{
    char buf[64];
    uint8_t n = 0;

    buf[n++] = 'q';
    buf[n++] = ':';
    n = (uint8_t)(n + copy_utf8(&buf[n], q, (uint8_t)(sizeof(buf) - 1 - n)));

    clear_pages(1, 2);          /* 问题区：第 1~2 页，够放一行 12x12 或两行 6x8 */
    draw_text(0, 1, 2, buf);
    page_clear(ANS_PAGE);       /* 顺手把旧的回答开头擦掉 */
    OLED_Update();
}

void N2UI_Answer(const char *a)
{
    char buf[80];
    uint8_t n = 0;

    buf[n++] = 'a';
    buf[n++] = ':';
    n = (uint8_t)(n + copy_utf8(&buf[n], a, (uint8_t)(sizeof(buf) - 1 - n)));

    clear_pages(ANS_PAGE, ANS_ROWS);   /* 回答区：第 3~7 页 */
    draw_text(0, ANS_PAGE, 7, buf);
    OLED_Update();
}

void N2UI_Thinking(uint8_t phase)
{
    static const char *dots[3] = { ".", "..", "..." };
    page_clear(0);
    put(0, 0, "nano2");
    put(0, (uint8_t)(COLS - 3), dots[phase % 3]);
    OLED_Update();
}
