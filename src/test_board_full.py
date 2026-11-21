# -*- coding: utf-8 -*-
"""真机全量验收：把 acceptance_host.py 的 33 条用例通过串口在板子上跑一遍。

用法： python test_board_full.py COM13
"""
import re
import sys
import time

import serial

from acceptance_host import CASES


def read_idle(ser, idle=5.0, max_wait=120.0):
    buf = bytearray()
    t0 = time.time()
    last = time.time()
    while True:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
            last = time.time()
        else:
            time.sleep(0.03)
            if time.time() - last >= idle:
                break
        if time.time() - t0 > max_wait:
            break
    return bytes(buf)


def parse(raw):
    text = raw.decode("ascii", "replace")
    m = re.search(r"nano2> (.*?)\r?\n", text, re.S)
    ans = m.group(1).strip() if m else "<no answer>"
    t = re.search(r"\[(\d+\.\ds) (\d+)B\]", text)
    return ans, ("%s %sB" % t.groups() if t else "?")


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM13"
    ser = serial.Serial(port, 115200, timeout=0.1)
    time.sleep(0.4)
    ser.reset_input_buffer()
    ser.write(b"\r\n")          # 触发一次空行，清掉残留
    read_idle(ser, idle=0.5, max_wait=2)

    ok = 0
    by_cat = {}
    rows = []
    for cat, q, want in CASES:
        ser.write(q.encode("ascii") + b"\r\n")
        ser.flush()
        got, stat = parse(read_idle(ser, idle=5.0))
        hit = got == want
        ok += hit
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][0] += hit
        by_cat[cat][1] += 1
        rows.append((cat, q, want, got, stat, hit))
        print(("OK  " if hit else "FAIL") + " | %-30s | %-8s | %r" % (q, stat, got), flush=True)
    ser.close()

    print("\n=== 真机成绩 %d/%d ===" % (ok, len(CASES)))
    for cat, (a, b) in by_cat.items():
        print("  %-6s %d/%d" % (cat, a, b))

    with open("验收报告_真机.md", "w", encoding="utf-8") as f:
        f.write("# nano2 真机验收（C8T6 + SSD1306 + CH340）\n\n")
        f.write("> 通过串口（115200 8N1）在板子上逐条提问，记录固件实际输出与耗时。\n\n")
        f.write("| 类别 | 通过 |\n|---|---|\n")
        for cat, (a, b) in by_cat.items():
            f.write("| %s | %d/%d |\n" % (cat, a, b))
        f.write("\n**总计 %d/%d**\n\n" % (ok, len(CASES)))
        f.write("| 类别 | 问题 | 期望 | 板上实际输出 | 耗时/字节 | 结果 |\n|---|---|---|---|---|---|\n")
        for cat, q, want, got, stat, hit in rows:
            f.write("| %s | `%s` | `%s` | `%s` | %s | %s |\n"
                    % (cat, q, want, got, stat, "OK" if hit else "**FAIL**"))
    print("wrote 验收报告_真机.md")


if __name__ == "__main__":
    main()
