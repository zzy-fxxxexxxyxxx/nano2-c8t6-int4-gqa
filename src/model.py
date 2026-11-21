# -*- coding: utf-8 -*-
"""nano2 模型：3 层 GQA + 滑窗因果注意力 + int4 QAT（STE 伪量化）。

所有伪量化都**逐项对齐固件实现**：
  权重：每输出行 scale = max|w|/7，q = sign*floor(|w|/scale+0.5) 夹到 [-7,7]
  激活：逐 token scale = max|x|/127，舍入同上
  GELU：161 点 LUT + 线性插值（和 C 版逐行对应）
  LN/softmax：f32
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = dict(
    vocab=128, d=48, layers=3, heads=4, head_dim=12, kv_groups=2,
    ffn=160, seq=96, window=72, eos=127, ln_eps=1e-5,
)
LUT_N, LUT_LO, LUT_HI = 161, -6.0, 6.0
LUT_STEP = (LUT_HI - LUT_LO) / (LUT_N - 1)


def afz(x):
    """away-from-zero rounding，与 C 里 (int)(fabs(v)+0.5) * sign 一致。"""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def fake_w4(w):
    """int4 伪量化（每输出行 = dim 0 一个 scale），STE 反传。"""
    with torch.no_grad():
        amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / 7.0
        q = afz(w / scale).clamp(-7.0, 7.0)
        wq = q * scale
    return w + (wq - w).detach()


def fake_a8(x):
    """int8 逐 token 动态伪量化（最后一维），STE 反传。"""
    with torch.no_grad():
        amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = amax / 127.0
        q = afz(x / scale).clamp(-127.0, 127.0)
        xq = q * scale
    return x + (xq - x).detach()


def gelu_table():
    xs = [LUT_LO + i * LUT_STEP for i in range(LUT_N)]
    return [0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0))) for x in xs]


GELU_LUT = torch.tensor(gelu_table(), dtype=torch.float32)


def gelu_lut(x):
    """与固件 gelu_fast() 完全一致：越界外推 + 段内线性插值。"""
    lut = GELU_LUT
    if not hasattr(gelu_lut, "_luts"):
        gelu_lut._luts = {}
    key = (x.device, x.dtype)
    lut = gelu_lut._luts.get(key)
    if lut is None:
        lut = GELU_LUT.to(device=x.device, dtype=x.dtype)
        gelu_lut._luts[key] = lut
    n = LUT_N
    out = torch.empty_like(x)
    lo_mask = x <= LUT_LO
    hi_mask = x >= LUT_HI
    mid = ~(lo_mask | hi_mask)
    out = torch.where(lo_mask, lut[0].expand_as(x), out)
    out = torch.where(hi_mask, lut[n - 1] + (x - LUT_HI), out)
    xm = x[mid]
    t = (xm - LUT_LO) / LUT_STEP
    i = torch.floor(t).clamp(0, n - 2).long()
    f = t - i.to(t.dtype)
    out = out.clone()
    out[mid] = lut[i] + (lut[i + 1] - lut[i]) * f
    return out


def rms_norm(x, w, eps=1e-5):
    """RMSNorm：不减均值、无 bias（LLaMA 风格），比 LayerNorm 少一遍减法。"""
    var = (x * x).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    return x * rstd * w


def rope_freqs(head_dim, base=10000.0, device=None):
    half = head_dim // 2
    return base ** (-torch.arange(half, device=device, dtype=torch.float32) / half)


def rope_apply(x, pos, base=10000.0):
    """x: [B, H, T, hd]；pos: [T]（绝对位置）。half-split 旋转。"""
    hd = x.shape[-1]
    half = hd // 2
    freqs = rope_freqs(hd, base, x.device)
    ang = pos.to(torch.float32)[:, None] * freqs[None, :]      # [T, half]
    cos = ang.cos()[None, None, :, :]
    sin = ang.sin()[None, None, :, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, hd, nh, g, ffn = cfg["d"], cfg["head_dim"], cfg["heads"], cfg["kv_groups"], cfg["ffn"]
        kv = hd * g
        self.nh, self.hd, self.g, self.kv = nh, hd, g, kv
        self.ln1_w = nn.Parameter(torch.ones(d))
        self.wq = nn.Parameter(torch.randn(d, hd * nh) * 0.06)
        self.wk = nn.Parameter(torch.randn(d, kv) * 0.06)
        self.wv = nn.Parameter(torch.randn(d, kv) * 0.06)
        self.bq = nn.Parameter(torch.zeros(hd * nh))
        self.bk = nn.Parameter(torch.zeros(kv))
        self.bv = nn.Parameter(torch.zeros(kv))
        self.wo = nn.Parameter(torch.randn(hd * nh, d) * 0.06)
        self.bo = nn.Parameter(torch.zeros(d))
        self.ln2_w = nn.Parameter(torch.ones(d))
        self.w1 = nn.Parameter(torch.randn(d, ffn) * 0.06)
        self.b1 = nn.Parameter(torch.zeros(ffn))
        self.w2 = nn.Parameter(torch.randn(ffn, d) * 0.06)
        self.b2 = nn.Parameter(torch.zeros(d))

    def forward(self, x, mask, pos):
        B, T, D = x.shape
        nh, hd, g = self.nh, self.hd, self.g
        h = rms_norm(x, self.ln1_w)
        hq = fake_a8(h)
        q = (hq @ fake_w4(self.wq) + self.bq).view(B, T, nh, hd).transpose(1, 2)   # [B,nh,T,hd]
        k = (hq @ fake_w4(self.wk) + self.bk).view(B, T, g, hd).transpose(1, 2)    # [B,g,T,hd]
        v = (hq @ fake_w4(self.wv) + self.bv).view(B, T, g, hd).transpose(1, 2)
        q = rope_apply(q, pos)          # RoPE：q/k 施加旋转位置编码（v 不动）
        k = rope_apply(k, pos)
        # GQA：每个 KV 组服务 nh/g 个 Q 头
        rep = nh // g
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        # 存储到 KV cache 前的 int8 伪量化（部署端 KV 就是 int8）
        k = fake_a8(k)
        v = fake_a8(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att.float(), dim=-1)
        o = att @ v
        o = o.transpose(1, 2).reshape(B, T, nh * hd)
        o = fake_a8(o)
        x = x + (o @ fake_w4(self.wo) + self.bo)
        h2 = rms_norm(x, self.ln2_w)
        h2 = fake_a8(h2)
        h2 = gelu_lut(h2 @ fake_w4(self.w1) + self.b1)
        h2 = fake_a8(h2)
        x = x + (h2 @ fake_w4(self.w2) + self.b2)
        return x


class Nano2(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        d, seq, vocab = cfg["d"], cfg["seq"], cfg["vocab"]
        self.tok_emb = nn.Parameter(torch.randn(vocab, d) * 0.06)
        # 位置编码改成 RoPE（旋转位置编码），不再需要可学习的 pos_emb 表
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["layers"])])
        self.ln_f_w = nn.Parameter(torch.ones(d))

    def build_mask(self, T, device):
        """causal + 滑窗(window) + 全局首 token；True = 屏蔽。"""
        i = torch.arange(T, device=device).view(T, 1)
        j = torch.arange(T, device=device).view(1, T)
        allowed = (j <= i) & ((i - j) < self.cfg["window"])
        allowed = allowed | (j == 0)
        return (~allowed).view(1, 1, T, T)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).clamp_max(self.cfg["seq"] - 1)
        x = fake_w4(self.tok_emb)[ids]
        mask = self.build_mask(T, ids.device)
        for blk in self.blocks:
            x = blk(x, mask, pos)
        h = rms_norm(x, self.ln_f_w)
        hq = fake_a8(h)
        # 权重绑定：tok_emb 的行视角当 lm_head
        logits = hq @ fake_w4(self.tok_emb).t()
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=48, eos=127):
        ids = list(prompt_ids)
        out = []
        for _ in range(max_new):
            x = torch.tensor([ids[-self.cfg["seq"]:]], dtype=torch.long)
            logits = self.forward(x)
            nxt = int(torch.argmax(logits[0, -1]).item())
            if nxt == eos:
                break
            out.append(nxt)
            ids.append(nxt)
        return out


def param_count(model):
    n = sum(p.numel() for p in model.parameters())
    int4 = model.tok_emb.numel() + sum(
        b.wq.numel() + b.wk.numel() + b.wv.numel() + b.wo.numel() + b.w1.numel() + b.w2.numel()
        for b in model.blocks)
    f32 = n - int4
    return n, int4, 0, f32
