# -*- coding: utf-8 -*-
"""nano2 训练：CPU 上训 int4 QAT 的 GQA 小模型。

用法：
    python train.py --epochs 200 --bs 64 --lr 3e-3
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import CFG, Nano2, param_count  # noqa: E402

DATA = os.path.join(HERE, "data_v3")
EOS = CFG["eos"]
FACTS = [
    ("谁是甲同学？", "甲同学是我爸最好的朋友。"),
    ("谁是乙同学？", "乙同学是American。"),
    ("谁是丙同学？", "丙同学是gay。"),
    ("谁是丁同学？", "丁同学是ai lab股神。"),
    ("谁是戊同学？", "戊同学是交大phd。"),
    ("谁是己同学？", "己同学是我爸爸的舍友。"),
    ("谁是庚同学？", "庚同学是我师兄。"),
    ("谁是辛同学？", "辛同学是我师兄。"),
    ("谁是壬同学？", "壬同学是我同级硕士生。"),
    ("丁同学真的是ai lab股神？", "对。"),
]
EXTRA = [
    ("who is alex lin", "alex lin is my dad."),
    ("who is dana li", "dana li is the best teacher in lands."),
    ("hi", "hello!"),
    ("who are you", "i am egg-nano."),
    ("what is 3+4", "7."),
    ("what is 7+8", "15."),
    ("which is bigger 3 or 9", "9 is bigger."),
    ("how is the weather", "i do not know."),
    ("你是谁？", "我是egg-nano。"),
    ("丁同学真的是ai lab股神？", "对。"),
]


def load_split(name):
    path = os.path.join(DATA, name)
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def encode(rec):
    ids = list(rec["text"].encode("utf-8")) + [EOS]
    na = len(("q:" + rec["q"] + "\na:").encode("utf-8"))
    return ids, na


def make_batch(recs, idxs):
    seqs, masks = [], []
    for i in idxs:
        ids, na = encode(recs[i])
        inputs = ids[:-1]
        tgts = ids[1:]
        m = [1 if (k + 1) >= na else 0 for k in range(len(tgts))]
        seqs.append(inputs)
        masks.append((tgts, m))
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), 0, dtype=torch.long)
    y = torch.full((len(seqs), L), -100, dtype=torch.long)
    for r, s in enumerate(seqs):
        x[r, :len(s)] = torch.tensor(s)
        t, m = masks[r]
        for k, (tok, keep) in enumerate(zip(t, m)):
            if keep:
                y[r, k] = tok
    return x, y


@torch.no_grad()
def greedy(model, prompt, max_new=48):
    ids = list(("q:" + prompt + "\na:").encode("utf-8"))
    out = model.generate(ids, max_new=max_new)
    return bytes(out).decode("utf-8", "replace")


@torch.no_grad()
def evaluate(model, verbose=False):
    model.eval()
    hits = 0
    rows = []
    for q, a in FACTS:
        got = greedy(model, q)
        ok = got.strip() == a
        hits += ok
        rows.append((q, a, got, ok))
    extra_hits = 0
    extra_rows = []
    for q, a in EXTRA:
        got = greedy(model, q)
        ok = got.strip() == a
        extra_hits += ok
        extra_rows.append((q, a, got, ok))
    if verbose:
        for q, a, got, ok in rows + extra_rows:
            print(("  OK  " if ok else "  FAIL") + " | %-32s -> %-40s (want %s)" % (q, got, a))
    model.train()
    return hits, len(FACTS), extra_hits, len(EXTRA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--every", type=int, default=20)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    train = load_split("train.jsonl")

    def weight_of(rec):
        """事实题/关系链最重要，闲聊次之，数学最占量所以权重最低。"""
        a = rec["a"]
        if ("my dad" in a or "my senior" in a or "best teacher in lands" in a
                or a.startswith(("alex lin", "blair hou", "casey yu", "dana li"))
                or "my mom" in a or "my brother" in a or "my friend" in a):
            return 8
        if a.strip().rstrip(".").isdigit():
            return 1
        return 3

    weighted = []
    for i, r in enumerate(train):
        weighted.extend([i] * weight_of(r))
    print("train records:", len(train), "weighted:", len(weighted))
    model = Nano2(CFG)
    n, n4, npos, nf = param_count(model)
    print("params total=%d  int4=%d  pos_int8=%d  f32=%d" % (n, n4, npos, nf))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    steps_per_epoch = max(1, len(weighted) // args.bs)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(50, total_steps // 10)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    t0 = time.time()
    step = 0
    for ep in range(1, args.epochs + 1):
        order = weighted[:]
        random.shuffle(order)
        tot, cnt = 0.0, 0
        for s in range(steps_per_epoch):
            idxs = order[s * args.bs:(s + 1) * args.bs]
            if not idxs:
                continue
            x, y = make_batch(train, idxs)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, CFG["vocab"]), y.reshape(-1),
                                   ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.step()
            step += 1
            tot += float(loss.detach())
            cnt += 1
        if ep % args.every == 0 or ep == 1:
            hf, nf_, he, ne = evaluate(model)
            print("ep %3d  loss %.4f  facts %d/%d  extra %d/%d  lr %.2e  %.1fs"
                  % (ep, tot / max(cnt, 1), hf, nf_, he, ne, lr_at(step), time.time() - t0),
                  flush=True)
            torch.save({"model": model.state_dict(), "cfg": CFG}, os.path.join(HERE, "ckpt.pt"))
        if ep == args.epochs:
            hf, nf_, he, ne = evaluate(model, verbose=True)
            torch.save({"model": model.state_dict(), "cfg": CFG}, os.path.join(HERE, "ckpt.pt"))
            print("saved ckpt.pt  facts %d/%d extra %d/%d" % (hf, nf_, he, ne))


if __name__ == "__main__":
    main()
