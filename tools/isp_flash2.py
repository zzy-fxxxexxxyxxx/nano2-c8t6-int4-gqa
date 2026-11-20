# -*- coding: utf-8 -*-
"""STM32F1 串口 ISP 烧录（极简稳健版）：握手 → 全片擦除 → 写 → 跳转。

不依赖 GET/GET_ID（那两个只是读信息，容易在握手后被打断），只做必要步骤。

用法：
    python isp_flash2.py COM13 <bin> [wait_seconds]
启动后会持续轻敲 0x7F 等待 bootloader —— 这段时间你按一下板子复位键即可。
"""
import os
import sys
import time

import serial

ACK, CMD_ERASE, CMD_WRITE, CMD_GO = 0x79, 0x43, 0x31, 0x21


def xor(*bs):
    v = 0
    for b in bs:
        v ^= b
    return v


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM13"
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "nano2", "fw", "build", "Debug", "nano2.bin")
    wait_s = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0

    ser = serial.Serial(port, 115200, timeout=0.25)
    ser.dtr = False
    ser.rts = False
    time.sleep(0.3)
    ser.reset_input_buffer()

    def ack(what, timeout=2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            b = ser.read(1)
            if b:
                if b[0] == ACK:
                    return
                raise IOError("%s: got 0x%02X (expect 0x79)" % (what, b[0]))
        raise IOError("%s: no response" % what)

    print("waiting for bootloader on %s (press reset now, up to %.0fs) ..." % (port, wait_s), flush=True)
    t0 = time.time()
    ok = False
    while time.time() - t0 < wait_s:
        ser.write(b"\x7f")
        ser.flush()
        b = ser.read(1)
        if b and b[0] == ACK:
            ok = True
            break
        time.sleep(0.05)
    if not ok:
        print("NO_ACK: bootloader 没响应（检查 BOOT0=1 和接线）", flush=True)
        return 2
    print("handshake OK after %.1fs" % (time.time() - t0), flush=True)
    time.sleep(0.2)
    ser.reset_input_buffer()

    data = open(path, "rb").read()
    print("chip erase ...", flush=True)
    ser.write(bytes([CMD_ERASE, 0xBC]))
    ser.flush()
    ack("ERASE cmd")
    ser.write(bytes([0xFF, 0x00]))   # 0xFF = 全片擦除
    ser.flush()
    ack("ERASE global", timeout=20.0)
    print("erased", flush=True)

    print("writing %d bytes @0x08000000 ..." % len(data), flush=True)
    t1 = time.time()
    for off in range(0, len(data), 256):
        chunk = data[off:off + 256]
        addr = 0x08000000 + off
        frame = bytes([CMD_WRITE, (addr >> 24) & 0xFF, (addr >> 16) & 0xFF,
                       (addr >> 8) & 0xFF, addr & 0xFF])
        ser.write(frame + bytes([xor(*frame)]))
        ser.flush()
        ack("WRITE addr")
        body = bytes([len(chunk) - 1]) + chunk
        ser.write(body + bytes([xor(*body)]))
        ser.flush()
        ack("WRITE data")
        if off % 8192 == 0:
            sys.stdout.write("\r  %5.1f%%" % (100.0 * off / len(data)))
            sys.stdout.flush()
    print("\r  100.0%%  (%.1fs)" % (time.time() - t1), flush=True)

    frame = bytes([CMD_GO, 0x08, 0x00, 0x00, 0x00])
    ser.write(frame + bytes([xor(*frame)]))
    ser.flush()
    try:
        ack("GO", timeout=1.0)
    except IOError:
        pass
    ser.close()
    print("FLASH_DONE —— 现在把 BOOT0 拨回 0，按一下复位，板子就跑新固件了", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
