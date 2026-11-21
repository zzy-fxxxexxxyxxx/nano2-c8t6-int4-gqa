# -*- coding: utf-8 -*-
"""把训练好的 ckpt 导出成固件用的 C 数组（int4 打包 + f32 伴随）。

布局约定（与固件 n2_model.c 一一对应）：
  - 所有矩阵按 **行主序 [out][in]** 存放（即 PyTorch 权重的转置）
  - int4 权重：每 2 个值打包进 1 字节，低半字节在前；二补码，范围 [-7, 7]
  - 每个输出行（out 通道）一个 f32 scale：w = scale * q
  - pos_emb: int8 + 每行一个 scale
  - bias: int16 + 每张量一个 f32 scale（省 Flash）
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import CFG, Nano2, GELU_LUT, afz  # noqa: E402

OUT = os.path.join(HERE, "fw", "src")


def q4(w):
    """w: [rows, out_channels] 的转置版（[out][in]）→ int4 + 每行 scale。"""
    with torch.no_grad():
        amax = w.abs().amax(dim=1).clamp_min(1e-8)
        s = amax / 7.0
        q = afz(w / s[:, None]).clamp(-7, 7).to(torch.int8)
    return q, s.to(torch.float32)


def q8(w):
    with torch.no_grad():
        amax = w.abs().amax(dim=1).clamp_min(1e-8)
        s = amax / 127.0
        q = afz(w / s[:, None]).clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def pack4(q):
    flat = (q.reshape(-1).to(torch.int32) & 0xF)
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int32)])
    return (flat[0::2] | (flat[1::2] << 4)).to(torch.uint8)


def c_u8(name, arr, per_line=16):
    b = bytes(arr.tolist())
    lines = []
    for i in range(0, len(b), per_line):
        lines.append("    " + ",".join("0x%02X" % x for x in b[i:i + per_line]) + ",")
    return "const uint8_t %s[%d] = {\n%s\n};\n" % (name, len(b), "\n".join(lines))


def c_i8(name, arr, per_line=16):
    b = arr.to(torch.int8).tolist()
    lines = []
    for i in range(0, len(b), per_line):
        lines.append("    " + ",".join("%d" % x for x in b[i:i + per_line]) + ",")
    return "const int8_t %s[%d] = {\n%s\n};\n" % (name, len(b), "\n".join(lines))


def fnum(v):
    """C 浮点字面量：保证有小数点/指数，再补 f 后缀（否则 0f 非法）。"""
    if v == 0.0:
        return "0.0f"
    s = "%.9g" % v
    if not any(ch in s for ch in ".eE"):
        s += ".0"
    return s + "f"


def c_f32(name, vals, per_line=6):
    if isinstance(vals, torch.Tensor):
        vals = vals.flatten().tolist()
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append("    " + ",".join(fnum(v) for v in vals[i:i + per_line]) + ",")
    return "const float %s[%d] = {\n%s\n};\n" % (name, len(vals), "\n".join(lines))


def quant_i16(vals):
    """返回 (int16 列表, 每张量 f32 scale)。"""
    v = torch.tensor(vals, dtype=torch.float32)
    amax = float(v.abs().max())
    if amax < 1e-8:
        amax = 1e-8
    sc = amax / 32767.0
    q = torch.clamp(torch.round(v / sc), -32767, 32767).to(torch.int16).tolist()
    return q, sc


def c_i16(name, q, per_line=8):
    body = []
    for i in range(0, len(q), per_line):
        body.append("    " + ",".join("%d" % x for x in q[i:i + per_line]) + ",")
    return "const int16_t %s[%d] = {\n%s\n};\n" % (name, len(q), "\n".join(body))


def main():
    ck = torch.load(os.path.join(HERE, "ckpt.pt"), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = Nano2(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    os.makedirs(OUT, exist_ok=True)

    L, D, NH, HD, G, FFN, V, SEQ = (cfg["layers"], cfg["d"], cfg["heads"], cfg["head_dim"],
                                    cfg["kv_groups"], cfg["ffn"], cfg["vocab"], cfg["seq"])
    KVD = HD * G
    hdr = os.path.join(OUT, "n2_weights.h")
    src = os.path.join(OUT, "n2_weights.c")
    decls, defs = [], []

    def add_mat(name, w_out_in):
        """w_out_in 已经是 [out][in] 行主序（与 PyTorch nn.Linear 的 weight.t() 一致）。"""
        w = w_out_in.detach().contiguous()
        q, s = q4(w)
        packed = pack4(q)
        rows, cols = w.shape
        defs.append(c_u8("n2_%s_q" % name, packed))
        defs.append(c_f32("n2_%s_s" % name, s))
        decls.append("extern const uint8_t n2_%s_q[%d];   /* %dx%d int4 */" % (name, len(packed), rows, cols))
        decls.append("extern const float   n2_%s_s[%d];" % (name, rows))

    # token embedding（同时当 lm_head）：PyTorch 里就是 [vocab, d] = [out][in]
    add_mat("tok_emb", model.tok_emb.detach())
    # 位置编码用 RoPE：只需导出每种频率（head_dim/2 个），固件按位置实时算 cos/sin
    import math as _m
    half = HD // 2
    freqs = [float(10000.0 ** (-i / half)) for i in range(half)]
    defs.append(c_f32("n2_rope_freq", freqs))
    decls.append("extern const float n2_rope_freq[%d];   /* RoPE 频率 */" % half)

    for i, b in enumerate(model.blocks):
        add_mat("l%d_wq" % i, b.wq.detach().t())   # [in,d] -> [out,in]
        add_mat("l%d_wk" % i, b.wk.detach().t())
        add_mat("l%d_wv" % i, b.wv.detach().t())
        add_mat("l%d_wo" % i, b.wo.detach().t())
        add_mat("l%d_w1" % i, b.w1.detach().t())
        add_mat("l%d_w2" % i, b.w2.detach().t())
        for nm, t in [("ln1_w", b.ln1_w), ("ln2_w", b.ln2_w),
                      ("bq", b.bq), ("bk", b.bk), ("bv", b.bv), ("bo", b.bo),
                      ("b1", b.b1), ("b2", b.b2)]:
            vals = t.detach().flatten().tolist()
            bname = "n2_l%d_%s" % (i, nm)
            if nm.startswith("b"):
                q, sc = quant_i16(vals)
                defs.append(c_i16(bname, q))
                decls.append("extern const int16_t %s[%d];" % (bname, len(q)))
                decls.append("#define %s_s (%s)   /* int16 bias 的还原标度 */" % (bname, fnum(sc)))
            else:
                defs.append(c_f32(bname, vals))
                decls.append("extern const float %s[%d];" % (bname, len(vals)))

    defs.append(c_f32("n2_ln_f_w", model.ln_f_w.detach().tolist()))
    decls.append("extern const float n2_ln_f_w[%d];" % D)

    defs.append(c_f32("n2_gelu_lut", GELU_LUT.tolist()))
    decls.append("extern const float n2_gelu_lut[161];")

    head = """// 自动生成，勿手改！由 nano2/export_c.py 从 ckpt.pt 导出
#ifndef N2_WEIGHTS_H
#define N2_WEIGHTS_H

#include <stdint.h>

#define N2_VOCAB   %d
#define N2_D       %d
#define N2_NH      %d
#define N2_HD      %d
#define N2_G       %d
#define N2_KVD     %d
#define N2_LAYERS  %d
#define N2_FFN     %d
#define N2_SEQ     %d
#define N2_WIN     %d
#define N2_EOS     %d
#define N2_LUT_N   161
#define N2_LUT_LO  (-6.0f)
#define N2_LUT_HI  (6.0f)
#define N2_LUT_STEP ((N2_LUT_HI - N2_LUT_LO) / (N2_LUT_N - 1))

""" % (V, D, NH, HD, G, KVD, L, FFN, SEQ, cfg["window"], cfg["eos"])
    head += "\n".join(decls) + "\n\n#endif // N2_WEIGHTS_H\n"
    with open(hdr, "w", encoding="utf-8") as f:
        f.write(head)
    with open(src, "w", encoding="utf-8") as f:
        f.write("// 自动生成，勿手改！\n#include \"n2_weights.h\"\n\n")
        f.write("\n".join(defs))
    print("wrote", hdr, os.path.getsize(hdr))
    print("wrote", src, os.path.getsize(src),
          "(%.1f KB 文本, 其中 int4 权重 %d B)" % (os.path.getsize(src) / 1024.0,
                                                sum(len(x) for x in defs) and 0))


if __name__ == "__main__":
    main()
