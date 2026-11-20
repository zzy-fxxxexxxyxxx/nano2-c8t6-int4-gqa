# -*- coding: utf-8 -*-
"""USART1 问答客户端（115200 8N1）。

用法：
  python talk.py COM5                       # 交互：敲一行回车发一条
  python talk.py COM5 --ask 你好 --ask 你是谁 --idle 2.5
  python talk.py --list                     # 列出串口

--ask 模式下：发一条问题，读到串口安静 idle 秒就认为回答结束，继续下一条。
"""
import argparse
import sys
import time

import serial
from serial.tools import list_ports

BAUD = 115200


def list_serial():
    for p in list_ports.comports():
        print("%-8s %s" % (p.device, p.description))


def read_until_idle(ser, idle=2.0, max_wait=120.0):
    """读到 idle 秒没有新字节为止。"""
    buf = bytearray()
    t_last = time.time()
    t0 = time.time()
    while True:
        waiting = ser.in_waiting
        if waiting:
            buf += ser.read(waiting)
            t_last = time.time()
        else:
            time.sleep(0.05)
            if time.time() - t_last >= idle:
                break
        if time.time() - t0 > max_wait:
            break
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?")
    ap.add_argument("--ask", action="append", default=[])
    ap.add_argument("--idle", type=float, default=2.5)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--raw", action="store_true", help="不回显、不排版，直接打印字节")
    args = ap.parse_args()

    if args.list or not args.port:
        list_serial()
        return

    ser = serial.Serial(args.port, BAUD, timeout=0.1)
    try:
        ser.reset_input_buffer()
        time.sleep(0.3)
        banner = read_until_idle(ser, idle=0.6, max_wait=5)
        if banner and not args.raw:
            print("[开机输出] " + banner.decode("utf-8", "replace").strip())

        questions = args.ask if args.ask else None
        if questions:
            for q in questions:
                print("\n>>> " + q)
                ser.write(q.encode("utf-8") + b"\r\n")
                ser.flush()
                t0 = time.time()
                resp = read_until_idle(ser, idle=args.idle)
                dt = time.time() - t0
                text = resp.decode("utf-8", "replace")
                print(text.rstrip())
                print("--- %.1fs, %d bytes" % (dt, len(resp)))
        else:
            print("输入问题回车发送（Ctrl-C 退出）：")
            while True:
                try:
                    line = input("问> ")
                except (EOFError, KeyboardInterrupt):
                    break
                if not line:
                    continue
                ser.write(line.encode("utf-8") + b"\r\n")
                ser.flush()
                t0 = time.time()
                resp = read_until_idle(ser, idle=args.idle)
                print(resp.decode("utf-8", "replace").rstrip())
                print("--- %.1fs, %d bytes" % (time.time() - t0, len(resp)))
    finally:
        ser.close()


if __name__ == "__main__":
    main()
