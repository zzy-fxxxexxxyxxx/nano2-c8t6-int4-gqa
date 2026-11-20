# -*- coding: utf-8 -*-
"""nano2 12x12 中文点阵字库生成器。

把 Windows 字体（默认 等线 Deng.ttf，缺失时用 微软雅黑 msyh.ttc）里的汉字
渲染成 12x12 点阵，生成固件用的 n2_font.c / n2_font.h。

用法：
    python tools/make_cjk_font.py                                  # 内置占位字符集
    python tools/make_cjk_font.py --charset chars.txt              # 从 txt 取字
    python tools/make_cjk_font.py --charset "谁是甲同学"            # 直接给一串字
    python tools/make_cjk_font.py --from-corpus ../data/train.jsonl # 从语料 q/a 取字
    python tools/make_cjk_font.py --charset a.txt --charset b.txt   # 多个来源取并集

默认只保留非 ASCII 字符（ASCII 由固件里的 6x8 字模负责），加 --all 才会一起收录。

字模格式（和固件 n2_font.h 里的约定一致）：
    每字 24 字节 = 12 行 x 2 字节，第 0 行在最上面；
    每行 16 位里高 12 位有效，bit15 是最左列（高位在左），低 4 位补 0。

渲染流程：
    字体按 12*--scale 像素绘制（默认 2x，即 24px），裁出"以字体原点为左上角、
    边长 12*scale 的正方形"字框，再按整数倍 BOX 面积平均（24:12 = 2:1）缩到
    12x12 灰度，最后阈值 128。
    必须整数倍缩放：非整数倍会把 1px 横画拆到相邻两行，灰度减半后在 128 处整条丢失。
    每个字还会在 ±1 像素内挑一个对齐偏移（取阈值 128 下墨点最多的那个），
    等价于栅格器的点阵对齐，避免"横画正好压在两行交界、整条被吃掉"。
    个别字（全角标点 ，。 之类）在 12x12 下墨迹覆盖率不足 50%，会整字消失；
    此时对这些字逐级降低阈值重试，保证至少画出 --min-ink 个点，并打印告警。
"""

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # nano2/
FW = os.path.join(ROOT, "fw", "src")

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\Deng.ttf",                  # 等线（细体，12px 下笔画最干净）
    r"C:\Windows\Fonts\msyh.ttc",                  # 微软雅黑（回退）
]

# 内置占位字符集：先用示例句子里出现过的那批字，后面换成 --from-corpus 即可
DEFAULT_CHARSET = (
    "谁是甲同学乙同学丙同学丁同学戊同学佟庚同学辛同学壬同学我爸最好的朋友"
    "americanailab股神交大phd舍友师兄同级硕士生对也有可能久留美是真的吗？，。！"
)

# 标定字：只用全角汉字，用来确定"12x12 里字要画多大"，避免标点/字母撑大或缩小小字框
CALIB_TEXT = "国永事是陈皓谁"

# 阈值阶梯：默认 128；墨迹太淡的字（，。！？ 这类）逐级降低，直到达到 min_ink
THRESHOLD_LADDER = (128, 112, 96, 80, 64, 48, 32)


def die(msg):
    sys.exit("错误: " + msg)


def pick_font(path):
    """选定字体文件；path 为空时按 FONT_CANDIDATES 顺序找。"""
    if path:
        if not os.path.isfile(path):
            die("字体不存在: " + path)
        return path
    for cand in FONT_CANDIDATES:
        if os.path.isfile(cand):
            return cand
    die("找不到字体，试过: " + "、".join(FONT_CANDIDATES))


def utf8_len(first_byte):
    """返回 UTF-8 序列长度（1..4），非法首字节返回 0。"""
    if first_byte < 0x80:
        return 1
    if (first_byte & 0xE0) == 0xC0:
        return 2
    if (first_byte & 0xF0) == 0xE0:
        return 3
    if (first_byte & 0xF8) == 0xF0:
        return 4
    return 0


def read_text_source(arg):
    """--charset 的参数既可以是 txt 文件路径，也可以是直接的字符串。"""
    if os.path.isfile(arg):
        with open(arg, "r", encoding="utf-8-sig", errors="replace") as f:
            return f.read()
    return arg


def chars_from_corpus(path):
    """读 jsonl，把 q/a（以及 question/answer/text 这些常见键）里的字符都收进来。"""
    if not os.path.isfile(path):
        die("语料文件不存在: " + path)
    keys = ("q", "a", "question", "answer", "text")
    out = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                print("  [跳过] 第 %d 行不是合法 JSON" % lineno)
                continue
            if isinstance(rec, dict):
                for k in keys:
                    v = rec.get(k)
                    if isinstance(v, str):
                        out.append(v)
            elif isinstance(rec, str):
                out.append(rec)
    return "".join(out)


def collect_chars(args):
    """汇总字符集：--charset / --from-corpus 可以给多次，取并集，去重后按码位升序排。"""
    raws = []
    sources = []
    for item in args.charset or []:
        raws.append(read_text_source(item))
        sources.append(os.path.basename(item) if os.path.isfile(item) else "命令行字符串")
    for path in args.from_corpus or []:
        raws.append(chars_from_corpus(path))
        sources.append(os.path.basename(path))
    if args.with_default:
        raws.append(DEFAULT_CHARSET)
        sources.append("内置占位字符集")
    if not raws:
        raws = [DEFAULT_CHARSET]
        sources = ["内置占位字符集"]

    raw = "".join(raws)
    keep = set()
    n_ascii = n_space = n_wide = 0
    for ch in raw:
        code = ord(ch)
        if ch.isspace():
            n_space += 1
            continue
        if code < 0x20 or code == 0x7F:
            continue
        if code < 0x80 and not args.all:
            n_ascii += 1
            continue
        if code > 0xFFFF:
            n_wide += 1
            continue
        keep.add(ch)

    if n_ascii:
        print("  跳过 %d 个 ASCII 字符（固件用 6x8 字模，需要时加 --all 收录）" % n_ascii)
    if n_wide:
        print("  跳过 %d 个 BMP 以外的字符（字库码位是 uint16_t）" % n_wide)
    if n_space:
        print("  跳过 %d 个空白字符" % n_space)
    if not keep:
        die("字符集为空（语料/文件里没有非 ASCII 字符？加 --all 或换 --charset）")
    return sorted(keep, key=ord), " + ".join(sources)


class Rasterizer(object):
    """把单个字渲染成 12x12 的 0/1 点阵。"""

    def __init__(self, font_path, cell=12, scale=2, threshold=128, min_ink=2, gain=1.0):
        self.cell = cell
        self.scale = scale
        self.threshold = threshold
        self.min_ink = min_ink
        self.gain = gain
        self.px = cell * scale
        self.font = ImageFont.truetype(font_path, self.px)
        self.thresholds = tuple(t for t in THRESHOLD_LADDER if t <= threshold)
        if not self.thresholds or self.thresholds[0] != threshold:
            self.thresholds = (threshold,) + self.thresholds
        # 对齐偏移候选：以字体原点为中心 ±1，按"离中心近"优先排序，平手时选中心的
        self.offsets = sorted(((dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)),
                              key=lambda d: (abs(d[0]) + abs(d[1]), d[1], d[0]))
        self.ref = self._calibration_box()

    # ---- 绘制原始灰度（画在足够大的画布上，原点固定，方便逐字对齐） ----
    def _ink(self, ch):
        img = Image.new("L", (self.px * 3, self.px * 3), 0)
        ImageDraw.Draw(img).text((self.px, self.px), ch, font=self.font, fill=255)
        return img

    def _calibration_box(self):
        """基准字框：以字体原点为左上角、边长 12*scale 的正方形。

        边长正好是 12*scale，缩放比才是整数（2:1），1px 的横画才可能整条落在同一行；
        非整数倍会把横画拆到相邻两行、灰度减半后在阈值 128 处整条消失。
        内置标定字只用来检查字有没有超出字框，不参与定位。
        """
        boxes = [self._ink(ch).getbbox() for ch in CALIB_TEXT]
        boxes = [b for b in boxes if b]
        if not boxes:
            die("标定字渲染失败，字体可能没有全角汉字")
        self.calib_union = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                            max(b[2] for b in boxes), max(b[3] for b in boxes))
        return (self.px, self.px, 2 * self.px, 2 * self.px)

    # ---- 单字 + 偏移 -> 12x12 灰度（行优先 144 字节） ----
    def _cell_bytes(self, ch, dx, dy):
        left = self.ref[0] + dx
        top = self.ref[1] + dy
        cell = self._ink(ch).crop((left, top, left + self.px, top + self.px)).resize(
            (self.cell, self.cell), Image.Resampling.BOX)
        if self.gain != 1.0:
            cell = cell.point(lambda v: min(255, int(v * self.gain)))
        return cell.tobytes()

    def _binarize(self, data, threshold):
        """灰度 -> (12 行整数, 墨点数)；整数里 bit15 是最左列。"""
        n = self.cell
        rows = []
        ink = 0
        for y in range(n):
            value = 0
            for x in range(n):
                if data[y * n + x] >= threshold:
                    value |= 1 << (15 - x)
                    ink += 1
            rows.append(value)
        return rows, ink

    def bitmap(self, ch):
        """返回 (12 行整数, 实际阈值, 墨点数)。

        汉字墨迹略大于字框（等线 24px 下约 25px），横画落点也随字而异；
        所以在 ±1 像素内挑一个偏移，取阈值 128 下墨点最多的那个 ——
        效果等价于栅格器的点阵对齐，能避免"横画正好跨在两行之间、整条消失"。
        """
        best = None
        for dx, dy in self.offsets:
            data = self._cell_bytes(ch, dx, dy)
            rows, ink = self._binarize(data, self.threshold)
            score = (ink, -(abs(dx) + abs(dy)))
            if best is None or score > best[0]:
                best = (score, data, rows, ink)
        _, data, rows, ink = best

        threshold = self.threshold
        if ink < self.min_ink:                      # 全角标点：逐级降阈值，至少留 min_ink 个点
            for t in self.thresholds[1:]:
                rows, ink = self._binarize(data, t)
                threshold = t
                if ink >= self.min_ink:
                    break
        return rows, threshold, ink


def pack_rows(rows):
    """12 行整数 -> 24 字节（每行 2 字节，高位在左）。"""
    out = []
    for value in rows:
        out.append((value >> 8) & 0xFF)
        out.append(value & 0xFF)
    return out


def c_bytes(vals):
    return ",".join("0x%02X" % v for v in vals)


def write_header(path, font_path, n_chars, cell, scale, threshold, source_desc):
    font_name = os.path.basename(font_path)
    lines = [
        "// n2_font.h —— 12x12 中文点阵字库（自动生成，勿手工修改）",
        "//",
        "// 生成脚本：nano2/tools/make_cjk_font.py",
        "// 字体    ：%s（%dpx 超采样 %dx → %dx%d 灰度 → 阈值 %d）"
        % (font_name, cell * scale, scale, cell, cell, threshold),
        "// 字符集  ：%d 个字（来源：%s）" % (n_chars, source_desc),
        "//",
        "// 字模用法：一个汉字纵向占 12 像素，跨 OLED 的两页；起始页存第 0..7 行，",
        "// 下一页的低 4 位存第 8..11 行。绘制见 n2_ui.c 的 blit12()。",
        "#ifndef N2_FONT_H",
        "#define N2_FONT_H",
        "",
        "#include <stdint.h>",
        "",
        "#define N2FONT_W 12",
        "#define N2FONT_H 12",
        "#define N2FONT_STEP 13",
        "#define N2FONT_COUNT %d" % n_chars,
        "",
        "// Unicode 码位表（升序，N2Font_Lookup 用它做二分查找）",
        "extern const uint16_t n2_font_codes[N2FONT_COUNT];",
        "// 字模表：每字 24 字节 = 12 行 x 2 字节，每行高 12 位有效，bit15 是最左列",
        "extern const uint8_t n2_font_bitmaps[][24];",
        "",
        "// 查字模：命中返回该字的 24 字节字模，未收录返回 0",
        "const uint8_t *N2Font_Lookup(uint16_t code);",
        "",
        "#endif",
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def write_source(path, entries, font_path, cell, scale, threshold):
    """entries: [(char, code, 24 字节列表)]"""
    font_name = os.path.basename(font_path)
    lines = [
        "// n2_font.c —— 12x12 中文点阵字库（由 nano2/tools/make_cjk_font.py 生成，勿手工修改）",
        "//",
        "// 字体：%s（%dpx 超采样 %dx → %dx%d 灰度 → 阈值 %d）"
        % (font_name, cell * scale, scale, cell, cell, threshold),
        "// 每字 24 字节 = 12 行 x 2 字节；每行高 12 位有效，bit15 是最左列",
        "",
        '#include "n2_font.h"',
        "",
        "const uint16_t n2_font_codes[N2FONT_COUNT] = {",
    ]
    for i in range(0, len(entries), 8):
        chunk = entries[i:i + 8]
        lines.append("    " + ", ".join("0x%04X" % e[1] for e in chunk) + ",")
    lines += [
        "};",
        "",
        "const uint8_t n2_font_bitmaps[N2FONT_COUNT][24] = {",
    ]
    for ch, code, data in entries:
        lines.append("    /* %s U+%04X */" % (ch, code))
        for i in range(0, 24, 8):
            head = "    {" if i == 0 else "     "
            tail = "}," if i + 8 == 24 else ","
            lines.append(head + c_bytes(data[i:i + 8]) + tail)
    lines += [
        "};",
        "",
        "const uint8_t *N2Font_Lookup(uint16_t code)",
        "{",
        "    uint16_t lo = 0;",
        "    uint16_t hi = N2FONT_COUNT;",
        "",
        "    while (lo < hi) {",
        "        uint16_t mid = (uint16_t)(lo + (hi - lo) / 2);   /* 防溢出的二分 */",
        "        if (code == n2_font_codes[mid]) return n2_font_bitmaps[mid];",
        "        if (code < n2_font_codes[mid]) hi = mid;",
        "        else lo = (uint16_t)(mid + 1);",
        "    }",
        "    return 0;                                            /* 没收录 */",
        "}",
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def write_preview(path, entries, cell=12, zoom=4, per_row=12):
    """导出字模预览图（白底黑字，方便人工核对有没有缺笔画）。"""
    n = len(entries)
    cols = min(per_row, n)
    rows = (n + cols - 1) // cols
    pad = 2
    img = Image.new("L", (cols * (cell + pad) + pad, rows * (cell + pad) + pad), 255)
    for i, (ch, code, data) in enumerate(entries):
        cx = pad + (i % cols) * (cell + pad)
        cy = pad + (i // cols) * (cell + pad)
        for y in range(cell):
            value = (data[y * 2] << 8) | data[y * 2 + 1]
            for x in range(cell):
                if value & (1 << (15 - x)):
                    img.putpixel((cx + x, cy + y), 0)
    img = img.resize((img.width * zoom, img.height * zoom), Image.Resampling.NEAREST)
    img.save(path)
    return path


def main():
    ap = argparse.ArgumentParser(
        description="生成 12x12 中文点阵字库（n2_font.c / n2_font.h）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--charset", action="append",
                    help="字符集：txt 文件路径，或直接给一串字符（可给多次，取并集）")
    ap.add_argument("--from-corpus", action="append",
                    help="从 jsonl 语料的 q/a 字段取字符（可给多次，取并集）")
    ap.add_argument("--all", action="store_true", help="连 ASCII 一起收录（默认只收非 ASCII）")
    ap.add_argument("--with-default", action="store_true",
                    help="除 --charset/--from-corpus 外，再并入内置占位字符集")
    ap.add_argument("--font", help="字体路径（默认等线 Deng.ttf，回退微软雅黑）")
    ap.add_argument("--cell", type=int, default=12, help="点阵边长，默认 12")
    ap.add_argument("--scale", type=int, default=2,
                    help="渲染倍数，默认 2（24px 起画）；必须整数倍缩到 12x12")
    ap.add_argument("--threshold", type=int, default=128, help="二值化阈值，默认 128")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="二值化前的灰度增益，默认 1.0；等线偏细，给 1.3~1.4 笔画更完整")
    ap.add_argument("--min-ink", type=int, default=2,
                    help="每字最少墨点数；不足则单字降低阈值重试，默认 2")
    ap.add_argument("--out-c", default=os.path.join(FW, "n2_font.c"))
    ap.add_argument("--out-h", default=os.path.join(FW, "n2_font.h"))
    ap.add_argument("--preview", help="额外导出字模预览 PNG 的路径")
    args = ap.parse_args()

    if args.cell != 12:
        print("注意：固件按 12x12 编解码，--cell 只影响预览")
    if args.scale < 1:
        die("--scale 至少为 1")

    font_path = pick_font(args.font)
    print("字体    : %s" % font_path)

    chars, source_desc = collect_chars(args)
    print("字符集  : %d 个字（来源：%s）" % (len(chars), source_desc))

    ras = Rasterizer(font_path, cell=args.cell, scale=args.scale,
                     threshold=args.threshold, min_ink=args.min_ink, gain=args.gain)
    print("字框    : %s（%dx%d 像素 @%dpx，整数倍 %d:1 面积平均）"
          % (ras.ref, ras.ref[2] - ras.ref[0], ras.ref[3] - ras.ref[1], ras.px, args.scale))

    entries = []
    fallback = []
    blank = []
    for ch in chars:
        rows, thr, ink = ras.bitmap(ch)
        data = pack_rows(rows)
        entries.append((ch, ord(ch), data))
        if ink == 0:
            blank.append(ch)
        elif thr != args.threshold:
            fallback.append((ch, thr, ink))

    n = len(entries)
    glyph_bytes = n * 24
    code_bytes = n * 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out_c)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_h)), exist_ok=True)
    write_header(args.out_h, font_path, n, args.cell, args.scale, args.threshold, source_desc)
    write_source(args.out_c, entries, font_path, args.cell, args.scale, args.threshold)

    print()
    print("字符数       : %d" % n)
    print("每字字节数   : 24（12 行 x 2 字节）")
    print("字模总字节   : %d (%.2f KiB)" % (glyph_bytes, glyph_bytes / 1024.0))
    print("码表总字节   : %d" % code_bytes)
    print("字库合计     : %d (%.2f KiB)"
          % (glyph_bytes + code_bytes, (glyph_bytes + code_bytes) / 1024.0))
    print("写入         : %s（%d 字节）" % (args.out_c, os.path.getsize(args.out_c)))
    print("             : %s（%d 字节）" % (args.out_h, os.path.getsize(args.out_h)))

    if fallback:
        print()
        print("阈值下调的字（12x12 下全角标点太小，按 %d 画不出来）：" % args.threshold)
        for ch, thr, ink in fallback:
            print("  '%s' U+%04X → 阈值 %d，%d 个墨点" % (ch, ord(ch), thr, ink))
    if blank:
        print()
        print("警告：以下字在字体里没有笔画（缺字）：%s" % "".join(blank))

    if args.preview:
        out = write_preview(args.preview, entries)
        print("预览图       : %s" % out)


if __name__ == "__main__":
    main()
