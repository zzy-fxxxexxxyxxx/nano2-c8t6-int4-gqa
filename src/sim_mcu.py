# -*- coding: utf-8 -*-
"""nano2 固件数值模拟器：用 numpy 复刻 n2_model.c 的整数路径，用来在烧板子前验证一致性。

与训练侧的区别（也是固件真实行为）：
  - 权重 int4 + 每行 f32 scale，int32 累加后一次性乘回两个标度
  - 激活 int8 逐 token 动态量化，away-from-zero 舍入
  - K/V 以 int8 + 逐位置 scale 存环形窗口（N2_WIN），外加 position 0 的全局槽
  - GELU 用 161 点 LUT + 线性插值
"""
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import CFG, Nano2, GELU_LUT, afz  # noqa: E402

F32 = np.float32


def quant_i8(x):
    amax = float(np.max(np.abs(x)))
    if amax < 1e-12:
        amax = 1e-12
    s = F32(amax / 127.0)
    v = x / s
    q = np.trunc(np.abs(v) + 0.5).astype(np.int32)
    q = np.clip(q, 0, 127) * np.sign(v).astype(np.int32)
    return q.astype(np.int8), s


def quant_w4(w):
    """w: [out][in] float -> int4 q [out][in] + scale [out]"""
    amax = np.maximum(np.abs(w).max(axis=1), 1e-8)
    s = (amax / 7.0).astype(F32)
    v = w / s[:, None]
    q = np.trunc(np.abs(v) + 0.5).astype(np.int32)
    q = np.clip(q, 0, 7) * np.sign(v).astype(np.int32)
    return q.astype(np.int8), s


def quant_w8(w):
    amax = np.maximum(np.abs(w).max(axis=1), 1e-8)
    s = (amax / 127.0).astype(F32)
    v = w / s[:, None]
    q = np.trunc(np.abs(v) + 0.5).astype(np.int32)
    q = np.clip(q, 0, 127) * np.sign(v).astype(np.int32)
    return q.astype(np.int8), s


def gemm(xq, sx, wq, ws, bias):
    """int32 累加 -> f32 反量化 + bias"""
    acc = (xq.astype(np.int32)[None, :] * wq.astype(np.int32)).sum(axis=1)
    return (acc.astype(F32) * sx * ws + bias.astype(F32)).astype(F32)


def rope(x, pos, freq):
    """x: [heads, hd]；half-split 旋转，与固件 rope_apply 一致。"""
    half = x.shape[-1] // 2
    ang = F32(pos) * freq[:half]
    c = np.cos(ang).astype(F32)
    s = np.sin(ang).astype(F32)
    x1 = x[..., :half].copy()
    x2 = x[..., half:].copy()
    out = x.copy()
    out[..., :half] = x1 * c - x2 * s
    out[..., half:] = x1 * s + x2 * c
    return out


def rms_norm(x, w, eps=1e-5):
    """RMSNorm：不减均值、无 bias。"""
    ss = F32((x * x).mean())
    rstd = F32(1.0 / math.sqrt(float(ss) + eps))
    return (x * rstd * w).astype(F32)


def gelu_lut(x):
    lo, hi = -6.0, 6.0
    n = 161
    step = (hi - lo) / (n - 1)
    lut = GELU_LUT.numpy()
    y = np.empty_like(x)
    for i, v in enumerate(x):
        if v <= lo:
            y[i] = lut[0]
        elif v >= hi:
            y[i] = lut[n - 1] + (v - hi)
        else:
            t = (v - lo) / step
            j = int(math.floor(t))
            f = t - j
            y[i] = lut[j] + (lut[j + 1] - lut[j]) * f
    return y.astype(F32)


class McuNano:
    def __init__(self, model: Nano2):
        cfg = model.cfg
        self.cfg = cfg
        self.D, self.NH, self.HD, self.G = cfg["d"], cfg["heads"], cfg["head_dim"], cfg["kv_groups"]
        self.KVD, self.FFN, self.V, self.SEQ, self.WIN = (self.HD * self.G, cfg["ffn"],
                                                          cfg["vocab"], cfg["seq"], cfg["window"])
        sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
        self.tok_q, self.tok_s = quant_w4(sd["tok_emb"])
        half = self.HD // 2
        self.rope_w = np.array([10000.0 ** (-i / half) for i in range(half)], dtype=F32)
        self.blocks = []
        for i in range(cfg["layers"]):
            p = "blocks.%d." % i
            W = {}
            for nm in ("wq", "wk", "wv", "wo", "w1", "w2"):
                W[nm] = quant_w4(sd[p + nm].T)  # [out][in]
            for nm in ("ln1_w", "ln2_w", "bq", "bk", "bv", "bo", "b1", "b2"):
                W[nm] = sd[p + nm].astype(F32)
            self.blocks.append(W)
        self.lnf_w = sd["ln_f_w"].astype(F32)
        self.reset()

    def reset(self):
        L, W, K = self.cfg["layers"], self.WIN, self.KVD
        self.kc = np.zeros((L, W, K), np.int8)
        self.vc = np.zeros((L, W, K), np.int8)
        self.kcs = np.zeros((L, W), F32)
        self.vcs = np.zeros((L, W), F32)
        self.kg = np.zeros((L, K), np.int8)
        self.vg = np.zeros((L, K), np.int8)
        self.kgs = np.zeros(L, F32)
        self.vgs = np.zeros(L, F32)
        self.t = 0
        self.x = np.zeros(self.D, F32)

    def step(self, byte):
        D, NH, HD, G, KVD, FFN = self.D, self.NH, self.HD, self.G, self.KVD, self.FFN
        t = self.t
        pos = min(t, self.SEQ - 1)
        x = self.tok_q[byte].astype(F32) * self.tok_s[byte]
        for L, W in enumerate(self.blocks):
            h = rms_norm(x, W["ln1_w"])
            hq, sx = quant_i8(h)
            q = gemm(hq, sx, W["wq"][0], W["wq"][1], W["bq"])
            k = gemm(hq, sx, W["wk"][0], W["wk"][1], W["bk"])
            v = gemm(hq, sx, W["wv"][0], W["wv"][1], W["bv"])
            q = rope(q.reshape(NH, HD), pos, self.rope_w).reshape(-1)
            k = rope(k.reshape(G, HD), pos, self.rope_w).reshape(-1)
            qi, sq = quant_i8(q)
            ki, sk = quant_i8(k)
            vi, sv = quant_i8(v)
            slot = t % self.WIN
            self.kc[L, slot] = ki
            self.vc[L, slot] = vi
            self.kcs[L, slot] = sk
            self.vcs[L, slot] = sv
            if t == 0:
                self.kg[L] = ki
                self.vg[L] = vi
                self.kgs[L] = sk
                self.vgs[L] = sv
            # 注意力：窗口内 + 全局首 token
            idx = [i for i in range(max(0, t - self.WIN + 1), t + 1)]
            use_global = t >= self.WIN
            att_scale = F32(1.0 / math.sqrt(HD))
            out = np.zeros(D, F32)
            for hh in range(NH):
                g = hh // (NH // G)
                qh = qi[hh * HD:(hh + 1) * HD].astype(np.int32)
                keys, vals, kss, vss = [], [], [], []
                for i in idx:
                    sl = i % self.WIN
                    keys.append(self.kc[L, sl]); vals.append(self.vc[L, sl])
                    kss.append(self.kcs[L, sl]); vss.append(self.vcs[L, sl])
                if use_global:
                    keys.append(self.kg[L]); vals.append(self.vg[L])
                    kss.append(self.kgs[L]); vss.append(self.vgs[L])
                sc = []
                for kk, skk in zip(keys, kss):
                    kh = kk[g * HD:(g + 1) * HD].astype(np.int32)
                    acc = int((qh * kh).sum())
                    sc.append(F32(F32(acc) * sq * skk * att_scale))
                sc = np.array(sc, F32)
                sc = sc - sc.max()
                e = np.exp(sc.astype(F32))
                p = e / e.sum()
                acc = np.zeros(HD, F32)
                for pi, vv, svv in zip(p, vals, vss):
                    vh = vv[g * HD:(g + 1) * HD].astype(F32)
                    acc += F32(pi * svv) * vh
                out[hh * HD:(hh + 1) * HD] = acc
            oq, so = quant_i8(out)
            x = x + gemm(oq, so, W["wo"][0], W["wo"][1], W["bo"])
            h2 = rms_norm(x, W["ln2_w"])
            h2q, s2 = quant_i8(h2)
            a = gemm(h2q, s2, W["w1"][0], W["w1"][1], W["b1"])
            a = gelu_lut(a)
            aq, sa = quant_i8(a)
            x = x + gemm(aq, sa, W["w2"][0], W["w2"][1], W["b2"])
        h = rms_norm(x, self.lnf_w)
        hq, sx = quant_i8(h)
        acc = (self.tok_q.astype(np.int32) @ hq.astype(np.int32)).astype(F32)
        logits = acc * sx * self.tok_s
        self.x = x
        self.t += 1
        return int(np.argmax(logits))

    def generate(self, prompt_bytes, max_new=48):
        self.reset()
        nxt = None
        for b in prompt_bytes:
            nxt = self.step(b)
        out = []
        for _ in range(max_new):
            if nxt == self.cfg["eos"]:
                break
            out.append(nxt)
            nxt = self.step(nxt)
        return bytes(out)


def main():
    ck = torch.load(os.path.join(HERE, "ckpt.pt"), map_location="cpu", weights_only=False)
    model = Nano2(ck["cfg"])
    model.load_state_dict(ck["model"])
    model.eval()
    sim = McuNano(model)
    prompts = [
        "who is mark lin", "who is nina hou", "who is owen yu", "who is paul li",
        "who is my senior", "hi", "who are you", "what is 3+4", "what is 7+8",
    ]
    agree = 0
    for p in prompts:
        pre = ("q:" + p + "\na:").encode()
        a_t = bytes(model.generate(list(pre), max_new=48)).decode("ascii", "replace")
        a_s = sim.generate(list(pre), max_new=48).decode("ascii", "replace")
        same = a_t.strip() == a_s.strip()
        agree += same
        print(("  MATCH " if same else "  DIFF  ") + "%-22s torch=%-42r mcu=%r" % (p, a_t, a_s))
    print("agreement: %d/%d" % (agree, len(prompts)))


if __name__ == "__main__":
    main()
