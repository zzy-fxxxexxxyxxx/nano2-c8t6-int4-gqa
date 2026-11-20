# -*- coding: utf-8 -*-
"""在 PC 上预览固件的 OLED 排版：编译 n2_ui.c + 假 OLED 驱动，输出 PNG。

不碰硬件，验证的是真正要跑在板子上的 n2_ui.c 的折行/分页/页眉逻辑。
"""
import os
import re
import subprocess
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.join(HERE, "fw", "src")
HOST = os.path.join(HERE, "host_test")
GCC = r"D:\software_installation\msys64\ucrt64\bin\gcc.exe"


def extract_font():
    src = open(os.path.join(FW, "OLED.c"), encoding="utf-8", errors="replace").read()
    m = re.search(r"OLED_F6x8\s*\[[^\]]*\]\s*\[\s*6\s*\]\s*=\s*\{(.*?)\n\};", src, re.S)
    if not m:
        sys.exit("OLED_F6x8 not found")
    body = m.group(1)
    rows = re.findall(r"\{([^{}]*)\}", body)
    if len(rows) != 95:
        sys.exit("expect 95 glyphs, got %d" % len(rows))
    out = ["#ifndef FONT6X8_H", "#define FONT6X8_H", "#include <stdint.h>",
           "static const uint8_t OLED_F6x8[95][6] = {"]
    for r in rows:
        vals = [int(x, 16) for x in re.findall(r"0x([0-9A-Fa-f]{2})", r)]
        vals += [0] * (6 - len(vals))
        out.append("    {" + ",".join("0x%02X" % v for v in vals[:6]) + "},")
    out += ["};", "#endif"]
    path = os.path.join(HOST, "font6x8.h")
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    return path


STUB_C = r"""
#include <string.h>
#include "font6x8.h"
#include "OLED.h"

uint8_t OLED_Buffer[8][128];

void OLED_Clear(void) { memset(OLED_Buffer, 0, sizeof(OLED_Buffer)); }
void OLED_Update(void) { }

void OLED_ShowChar(uint8_t x, uint8_t page, char ch)
{
    int i;
    if (x > 127 || page > 7) return;
    if (ch < ' ' || ch > '~') return;
    for (i = 0; i < 6; i++) {
        if (x + i < 128) OLED_Buffer[page][x + i] = OLED_F6x8[ch - ' '][i];
    }
}

void OLED_ShowString(uint8_t x, uint8_t page, const char *str)
{
    while (*str) { OLED_ShowChar(x, page, *str); x += 6; str++; }
}
"""

PREVIEW_C = r"""
#include <stdio.h>
#include <string.h>
#include "OLED.h"
#include "n2_ui.h"

static void dump(const char *path)
{
    FILE *f = fopen(path, "wb");
    int x, y, p;
    fprintf(f, "P2\n128 64\n255\n");
    for (y = 0; y < 64; y++) {
        for (x = 0; x < 128; x++) {
            p = y >> 3;
            fprintf(f, "%d ", (OLED_Buffer[p][x] >> (y & 7)) & 1 ? 255 : 0);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main(void)
{
    OLED_Clear();
    N2UI_Boot();
    dump("ui_boot.pgm");

    OLED_Clear();
    N2UI_Header("12.3s 20B");
    N2UI_Question("who is dana li");
    N2UI_Answer("dana li is the best teacher in lands.");
    dump("ui_qa.pgm");

    OLED_Clear();
    N2UI_Header("");
    N2UI_Question("who is blair hou");
    N2UI_Answer("blair hou is my dad's senior.");
    N2UI_Thinking(2);
    dump("ui_thinking.pgm");
    return 0;
}
"""


def main():
    extract_font()
    open(os.path.join(HOST, "oled_stub.c"), "w", encoding="utf-8").write(STUB_C)
    open(os.path.join(HOST, "ui_preview.c"), "w", encoding="utf-8").write(PREVIEW_C)
    exe = os.path.join(HOST, "ui_preview.exe")
    cmd = [GCC, "-O2", "-std=c99", "-o", exe,
           os.path.join(HOST, "ui_preview.c"),
           os.path.join(HOST, "oled_stub.c"),
           os.path.join(FW, "n2_ui.c"),
           os.path.join(FW, "n2_font.c"),
           "-I", HOST, "-I", FW]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit("编译失败")
    subprocess.run([exe], cwd=HOST, check=True)
    for name in ("ui_boot", "ui_qa", "ui_thinking"):
        pgm = os.path.join(HOST, name + ".pgm")
        img = Image.open(pgm).convert("L").resize((512, 256), Image.NEAREST)
        out = os.path.join(HERE, name + ".png")
        img.save(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
