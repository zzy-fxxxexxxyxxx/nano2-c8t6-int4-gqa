# -*- coding: utf-8 -*-
"""nano2 固件构建（等价 EIDE Debug 目标，逐文件编译）。

用法： python build_n2.py [--clean] [--gcc <toolchain bin>]
产物： build/Debug/nano2.{elf,hex,bin,map}
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "build", "Debug")
OBJ = os.path.join(OUT, "obj")

EXCLUDED_HAL = {
    "stm32f10x_bkp.c", "stm32f10x_can.c", "stm32f10x_cec.c", "stm32f10x_crc.c",
    "stm32f10x_dac.c", "stm32f10x_dbgmcu.c", "stm32f10x_dma.c", "stm32f10x_flash.c",
    "stm32f10x_fsmc.c", "stm32f10x_i2c.c", "stm32f10x_iwdg.c", "stm32f10x_pwr.c",
    "stm32f10x_rtc.c", "stm32f10x_sdio.c", "stm32f10x_wwdg.c",
}

INCLUDES = [
    "src",
    "lib/cmsis",
    "hal/STM32F10x_StdPeriph_Driver",
    "hal/STM32F10x_StdPeriph_Driver/inc",
    "hal/STM32F10x_StdPeriph_Driver/src",
]

CFLAGS = [
    "-mcpu=cortex-m3", "-mthumb", "-mfloat-abi=soft",
    "-Os", "-g", "-Wall", "-ffunction-sections", "-fdata-sections",
    "-DUSE_STDPERIPH_DRIVER", "-DSTM32F10X_MD",
] + [t for inc in INCLUDES for t in ("-I", inc)]

LDFLAGS = [
    "-T", "stm32f1x_64KB_flash.ld",
    "--specs=nosys.specs", "--specs=nano.specs",
    "-Wl,--gc-sections", "-Wl,--print-memory-usage",
    # newlib 的 expf 会拖进 ARM EHABI 展开链（约 4KB），本工程用不到异常
    "-Wl,--defsym=__aeabi_unwind_cpp_pr0=0",
]

TOOLCHAIN = os.path.abspath(os.path.join(ROOT, "..", "..", "tools", "gcc",
                                         "xpack-arm-none-eabi-gcc-10.2.1-1.1", "bin"))


def tool(name, gcc_dir):
    p = os.path.join(gcc_dir, name + ".exe")
    if os.path.exists(p):
        return p
    w = shutil.which(name)
    if w:
        return w
    sys.exit("找不到 " + name)


def sources():
    out = []
    for f in sorted(os.listdir(os.path.join(ROOT, "src"))):
        if f.endswith(".c"):
            out.append(os.path.join("src", f))
    hsrc = os.path.join(ROOT, "hal", "STM32F10x_StdPeriph_Driver", "src")
    for f in sorted(os.listdir(hsrc)):
        if f.endswith(".c") and f not in EXCLUDED_HAL:
            out.append(os.path.join("hal", "STM32F10x_StdPeriph_Driver", "src", f))
    out.append(os.path.join("hal", "STM32F10x_StdPeriph_Driver", "system_stm32f10x.c"))
    out.append(os.path.join("src", "startup_stm32f10x_md.s"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--gcc", default=TOOLCHAIN)
    args = ap.parse_args()

    cc = tool("arm-none-eabi-gcc", args.gcc)
    objcopy = tool("arm-none-eabi-objcopy", args.gcc)
    size = tool("arm-none-eabi-size", args.gcc)

    if args.clean and os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OBJ, exist_ok=True)

    objs = []
    for src in sources():
        obj = os.path.join(OBJ, os.path.splitext(os.path.basename(src))[0] + ".o")
        objs.append(obj)
        if os.path.exists(obj) and os.path.getmtime(obj) >= os.path.getmtime(os.path.join(ROOT, src)):
            continue
        r = subprocess.run([cc] + CFLAGS + ["-c", src, "-o", obj], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            sys.exit("编译失败: " + src)
        if r.stderr.strip():
            print("[" + os.path.basename(src) + "]\n" + r.stderr)

    elf = os.path.join(OUT, "nano2.elf")
    mapf = os.path.join(OUT, "nano2.map").replace("\\", "/")
    cmd = [cc, "-mcpu=cortex-m3", "-mthumb", "-mfloat-abi=soft"] + LDFLAGS
    cmd += ["-Wl,-Map=" + mapf, "-o", elf] + objs + ["-lm"]
    print("link ...")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        subprocess.call([size] + objs)
        sys.exit("链接失败")

    for ext, fmt in (("hex", "ihex"), ("bin", "binary")):
        subprocess.check_call([objcopy, "-O", fmt, elf, os.path.join(OUT, "nano2." + ext)])
    subprocess.call([size, elf])
    used = os.path.getsize(os.path.join(OUT, "nano2.bin"))
    print("bin = %d bytes (%.1f KiB), flash 65536, free %d" % (used, used / 1024.0, 65536 - used))


if __name__ == "__main__":
    main()
