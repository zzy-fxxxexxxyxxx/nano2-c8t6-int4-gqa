// main.c —— nano2：串口收英文问题 -> 片上 int4 推理 -> 串口 + OLED 输出
// 接线：USART1 PA9(TX)/PA10(RX) 115200；OLED 软件 IIC SCL=PA6 / SDA=PA5
// 语料全英文（ASCII），所以不需要中文字库，屏上不会出现方框。

#include <string.h>

#include "stm32f10x.h"
#include "n2_weights.h"
#include "n2_model.h"
#include "nano_uart.h"
#include "n2_ui.h"
#include "OLED.h"

#define MAX_NEW     48
#define REPLY_MAX   56
#define PROMPT_MAX  96

static char s_reply[REPLY_MAX + 2];
static int  s_reply_len;

static void on_byte(uint8_t b)
{
    NanoUart_SendByte(b);
    if (s_reply_len < REPLY_MAX) {
        s_reply[s_reply_len++] = (char)b;
        s_reply[s_reply_len] = 0;
        N2UI_Answer(s_reply);
    }
}

static uint32_t s_tick_last;
static uint8_t  s_tick_phase;

static void on_tick(void)
{
    uint32_t now = Nano_Millis();
    if (now - s_tick_last < 250u) return;
    s_tick_last = now;
    N2UI_Thinking(s_tick_phase);
    s_tick_phase = (uint8_t)((s_tick_phase + 1) % 3);
}

// 保留 UTF-8 字节（>=0x80，中文要用），ASCII 只留可打印并把 A-Z 转小写
static int normalize(char *s, int n)
{
    int i, k = 0;
    for (i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 0x80) { s[k++] = (char)c; continue; }   /* UTF-8 多字节原样保留 */
        if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
        if (c >= 0x20 && c <= 0x7E) s[k++] = (char)c;
    }
    s[k] = 0;
    return k;
}

static void fmt_status(char *buf, uint32_t ms, int bytes)
{
    char tmp[12];
    int i = 0, k = 0;
    buf[k++] = (char)('0' + (ms / 10000) % 10);
    buf[k++] = (char)('0' + (ms / 1000) % 10);
    buf[k++] = '.';
    buf[k++] = (char)('0' + (ms / 100) % 10);
    buf[k++] = 's';
    buf[k++] = ' ';
    if (bytes == 0) tmp[i++] = '0';
    while (bytes > 0) { tmp[i++] = (char)('0' + bytes % 10); bytes /= 10; }
    while (i > 0) buf[k++] = tmp[--i];
    buf[k++] = 'B';
    buf[k] = 0;
}

int main(void)
{
    char question[64];
    uint8_t prompt[PROMPT_MAX];
    uint8_t reply[MAX_NEW];
    char status[24];
    uint8_t first_boot = 1;

    NanoUart_Init();
    /* 让调试口在 sleep/stop/standby 下也能用：否则 MCU 跑固件时 AHB-AP 不可访问，
       每次烧录都得先拨 BOOT0=1。这里置 DBGMCU_CR 的 DBG_SLEEP/STOP/STANDBY 三位。 */
    DBGMCU->CR |= 0x00000007u;

    OLED_Init();      /* 必须先初始化 SSD1306（含软件 IIC 引脚与电荷泵） */
    N2UI_Boot();

    for (;;)
    {
        int qlen = NanoUart_ReadLine(question, sizeof(question) - 1);
        int plen = 0, n, i;
        uint32_t t0, dt;

        if (qlen <= 0) { __WFI(); continue; }

        qlen = normalize(question, qlen);
        if (qlen == 0) continue;

        // prompt = "q:" + 问题 + "\na:"，长度不超过 PROMPT_MAX
        prompt[plen++] = 'q';
        prompt[plen++] = ':';
        for (i = 0; i < qlen && plen < PROMPT_MAX - 8; i++) prompt[plen++] = (uint8_t)question[i];
        prompt[plen++] = '\n';
        prompt[plen++] = 'a';
        prompt[plen++] = ':';

        s_reply_len = 0;
        s_reply[0] = 0;
        s_tick_last = 0;
        s_tick_phase = 0;

        if (first_boot) {
            RCC_ClocksTypeDef clk;
            RCC_GetClocksFreq(&clk);
            NanoUart_Send("\r\ndbg clk sysclk=");
            NanoUart_SendNum(clk.SYSCLK_Frequency);
            NanoUart_Send(" hclk=");
            NanoUart_SendNum(clk.HCLK_Frequency);
            NanoUart_Send(" pclk2=");
            NanoUart_SendNum(clk.PCLK2_Frequency);
            NanoUart_Send("\r\n");
            first_boot = 0;
        }

        N2UI_Header("ready");
        N2UI_Question(question);

        NanoUart_Send("nano2> ");
        t0 = Nano_Millis();
        n = N2_Generate(prompt, plen, reply, MAX_NEW, on_byte, on_tick);
        dt = Nano_Millis() - t0;
        reply[n] = 0;

        N2UI_Answer(s_reply);
        fmt_status(status, dt, n);
        N2UI_Header(status);

        NanoUart_Send("\r\n[");
        NanoUart_Send(status);
        NanoUart_Send("]\r\n");
        NanoUart_Send("dbg prompt[");
        NanoUart_SendNum((uint32_t)plen);
        NanoUart_Send("]=");
        NanoUart_SendHex(prompt, plen);
        NanoUart_Send("\r\ndbg hex=");
        NanoUart_SendHex(reply, n);
        NanoUart_Send("\r\n");
    }
}
