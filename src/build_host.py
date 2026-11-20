# -*- coding: utf-8 -*-
"""把固件里的 n2_model.c + n2_weights.c 编译成 PC 程序（宿主验证用）。

需要 msys64 的 gcc；默认路径 D:\\software_installation\\msys64\\ucrt64\\bin\\gcc.exe
用法： python build_host.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GCC_CANDIDATES = [
    r"D:\software_installation\msys64\ucrt64\bin\gcc.exe",
    r"C:\msys64\ucrt64\bin\gcc.exe",
    "gcc",
]


def main():
    gcc = None
    for c in GCC_CANDIDATES:
        if c == "gcc" or os.path.exists(c):
            gcc = c
            break
    if not gcc:
        sys.exit("找不到 gcc（需要 msys64 的 ucrt64 gcc）")

    host = os.path.join(HERE, "host_test")
    out = os.path.join(host, "host_n2.exe")
    cmd = [gcc, "-O2", "-std=c99", "-o", out,
           os.path.join(host, "main.c"),
           os.path.join(HERE, "fw", "src", "n2_model.c"),
           os.path.join(HERE, "fw", "src", "n2_weights.c"),
           "-I", host,
           "-I", os.path.join(HERE, "fw", "src"),
           "-lm"]
    print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit("编译失败")
    print("built:", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
