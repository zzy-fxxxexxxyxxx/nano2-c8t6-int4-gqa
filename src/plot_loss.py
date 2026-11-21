# -*- coding: utf-8 -*-
"""从 train log 里解析 loss 曲线，画成 PNG（用于报告）。"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "train_v3.log"
out = sys.argv[2] if len(sys.argv) > 2 else "loss_curve.png"

eps, losses, vloss, facts, extras = [], [], [], [], []
pat = re.compile(r"ep\s+(\d+)\s+loss\s+([\d.]+)(?:\s+val_loss\s+([\d.]+))?\s+facts\s+(\d+)/(\d+)\s+extra\s+(\d+)/(\d+)")
raw = open(path, "rb").read()
for enc in ("utf-8", "utf-16", "utf-16-le", "gbk"):
    try:
        text = raw.decode(enc)
        break
    except UnicodeDecodeError:
        continue
else:
    text = raw.decode("utf-8", "replace")

for line in text.splitlines():
    m = pat.search(line)
    if m:
        eps.append(int(m.group(1)))
        losses.append(float(m.group(2)))
        vloss.append(float(m.group(3)) if m.group(3) else float("nan"))
        facts.append(int(m.group(4)) / max(1, int(m.group(5))))
        extras.append(int(m.group(6)) / max(1, int(m.group(7))))

if not eps:
    sys.exit("no data parsed from " + path)

fig, ax1 = plt.subplots(figsize=(8, 4.2), dpi=140)
ax1.set_xlabel("epoch")
ax1.set_ylabel("cross-entropy loss (answer tokens)")
ax1.plot(eps, losses, color="#1f77b4", lw=1.6, label="train loss")
if any(x == x for x in vloss):
    ax1.plot(eps, vloss, color="#ff7f0e", lw=1.4, ls="--", label="val loss")
ax1.set_yscale("log")
ax1.grid(True, which="both", ls=":", alpha=0.4)

ax2 = ax1.twinx()
ax2.set_ylabel("exact-match rate")
ax2.plot(eps, facts, color="#d62728", lw=1.3, ls="-", label="facts (greedy)")
ax2.plot(eps, extras, color="#2ca02c", lw=1.3, ls="--", label="extra set (greedy)")
ax2.set_ylim(-0.05, 1.05)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
plt.title("nano2 v3 training (int4 QAT, GQA, sliding-window attention)", fontsize=10)
plt.tight_layout()
plt.savefig(out)
print("wrote", out, "points:", len(eps))
