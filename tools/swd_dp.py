# -*- coding: utf-8 -*-
"""直接读 SWD 的 DP 寄存器，判断调试电源/系统电源是否被目标确认（AP 访问失败最常见的原因）。

DP 寄存器：
  0x00 IDCODE
  0x04 CTRL/STAT  bit31 CSYSPWRUPACK, bit29 CDBGPWRUPACK,
                  bit30 CSYSPWRUPREQ, bit28 CDBGPWRUPREQ
"""
import sys
import time

from pyocd.core.helpers import ConnectHelper


def main():
    probes = ConnectHelper.get_all_connected_probes()
    if not probes:
        sys.exit("no debugger found")
    probe = probes[0]
    probe.open()
    print("probe:", probe.product_name, probe.unique_id)
    try:
        print("IDCODE      = 0x%08X" % probe.read_dp(0x00))
        stat = probe.read_dp(0x04)
        print("CTRL/STAT   = 0x%08X" % stat)
        print("  CDBGPWRUPREQ=%d CDBGPWRUPACK=%d CSYSPWRUPREQ=%d CSYSPWRUPACK=%d"
              % ((stat >> 28) & 1, (stat >> 29) & 1, (stat >> 30) & 1, (stat >> 31) & 1))
        probe.write_dp(0x00, 0x1E)   # ABORT: clear sticky error
        print("after abort CTRL/STAT = 0x%08X" % probe.read_dp(0x04))
        probe.write_dp(0x04, (1 << 30) | (1 << 28))
        time.sleep(0.05)
        stat2 = probe.read_dp(0x04)
        print("after pwr-req CTRL/STAT = 0x%08X" % stat2)
        print("  CDBGPWRUPACK=%d CSYSPWRUPACK=%d" % ((stat2 >> 29) & 1, (stat2 >> 31) & 1))
        try:
            probe.write_dp(0x08, 0x00000000)  # SELECT: APSEL=0, APBANKSEL=0
            idr = probe.read_ap(0xFC)
            print("AP0 IDR     = 0x%08X" % idr)
        except Exception as e:
            print("AP0 IDR read failed:", e)
    finally:
        probe.close()


if __name__ == "__main__":
    main()
