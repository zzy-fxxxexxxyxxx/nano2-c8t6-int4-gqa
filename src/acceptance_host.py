# -*- coding: utf-8 -*-
"""宿主验收：直接运行固件里那份 n2_model.c（gcc 编成 PC 程序），比对答案。

这是比 sim_mcu 更硬的证据——跑的是要烧进板子的同一份 C 源码和同一份 n2_weights.c，
只把 stm32f10x.h 换成空 shim。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "host_test", "host_n2.exe")

def _identity():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "identity.txt")
    try:
        return open(p, encoding="utf-8-sig").read().strip().lstrip("\ufeff") or "nano-egg"
    except OSError:
        return "nano-egg"


IDENT = _identity()

AMBIGUOUS = {}

CASES = [
    # 用户指定的九个人（英文/拼音问法）
    ("必备事实", "who is alex chen", "alex chen is my dad's best friend."),
    ("必备事实", "who is blair ouyang", "blair ouyang is american."),
    ("必备事实", "who is casey deng", "casey deng is gay."),
    ("必备事实", "who is dana luo", "dana luo is the ai lab stock god."),
    ("必备事实", "who is erin gao", "erin gao is a sjtu phd."),
    ("必备事实", "who is finn tong", "finn tong is my dad's roommate."),
    ("必备事实", "who is gina tong", "gina tong is my senior."),
    ("必备事实", "who is hank xu", "hank xu is my senior."),
    ("必备事实", "who is iris wang", "iris wang is my master classmate."),
    ("必备事实", "dana luo is the ai lab stock god?", "yes."),   # 两个答案都算对，见 AMBIGUOUS
    # 换说法
    ("换说法", "tell me about alex chen", "alex chen is my dad's best friend."),
    ("换说法", "do you know dana luo", "dana luo is the ai lab stock god."),
    ("换说法", "who is iris wang?", "iris wang is my master classmate."),
    # 旧的四条 + 关系链
    ("旧事实", "who is mark lin", "mark lin is my dad."),
    ("旧事实", "who is nina hou", "nina hou is my dad's senior."),
    ("旧事实", "who is owen yu", "owen yu is my senior's senior."),
    ("旧事实", "who is paul li", "paul li is the best teacher in lands."),
    ("关系链", "who is my dad's senior", "nina hou is my dad's senior."),
    ("关系链", "who is my senior's senior", "owen yu is my senior's senior."),
    ("关系链", "who is my senior", "nina hou is my senior."),
    # 日常对话
    ("日常对话", "hi", "hello!"),
    ("日常对话", "hello", "hi there!"),
    ("日常对话", "good morning", "good morning!"),
    ("日常对话", "who are you", "i am %s." % IDENT),
    ("日常对话", "what is your name", "my name is %s." % IDENT),
    ("日常对话", "how are you", "i am fine, thanks."),
    ("日常对话", "i am tired", "take a rest."),
    ("日常对话", "do you like cats", "yes, i like cats."),
    ("日常对话", "bye", "bye!"),
    ("日常对话", "thanks", "you are welcome."),
    # 数学
    ("数学", "what is 3+4", "7."),
    ("数学", "what is 7+8", "15."),
    ("数学", "what is 9+9", "18."),
    ("数学", "what is 5-2", "3."),
    ("数学", "what is 3*4", "12."),
    ("数学", "what is 9*5", "45."),
    ("数学", "which is bigger 3 or 9", "9 is bigger."),
    ("数学", "is 2 bigger than 5", "no."),
    ("数学", "count 7 8 9", "10."),
    ("数学", "1 2 3 4 what comes next", "5."),
    # 边界
    ("边界拒绝", "how is the weather", "i do not know."),
    ("边界拒绝", "what time is it", "i do not know."),
    ("边界拒绝", "who is the president", "i do not know."),
    ("边界拒绝", "what is 100 times 100", "too hard for me."),
]


AMBIGUOUS = {"dana luo is the ai lab stock god?":
             ("yes.", "maybe the ai lab kurume too.")}


def run_batch(prompts):
    inp = "\n".join(prompts) + "\n"
    r = subprocess.run([EXE, "--batch"], input=inp.encode("utf-8"), capture_output=True)
    out = r.stdout.decode("utf-8", "replace")
    answers = {}
    cur = None
    for line in out.splitlines():
        if line.startswith("PROMPT:"):
            cur = line[len("PROMPT:"):]
        elif line.startswith("ANSWER:"):
            answers[cur] = line[len("ANSWER:"):].strip()
        elif line == "END":
            cur = None
    return answers


def main():
    if not os.path.exists(EXE):
        sys.exit("先编译宿主版本: cd host_test && gcc -O2 -std=c99 -o host_n2.exe main.c ../fw/src/n2_model.c ../fw/src/n2_weights.c -I . -I ../fw/src -lm")
    ans = run_batch([q for _, q, _ in CASES])
    ok = 0
    by_cat = {}
    rows = []
    for cat, q, want in CASES:
        got = ans.get(q, "<missing>")
        hit = (got == want) or (q in AMBIGUOUS and got in AMBIGUOUS[q])
        ok += hit
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][0] += hit
        by_cat[cat][1] += 1
        rows.append((cat, q, want, got, hit))
    print("host (真实 C 源码) 验收: %d/%d" % (ok, len(CASES)))
    for cat, (a, b) in by_cat.items():
        print("  %-6s %d/%d" % (cat, a, b))
    bad = [r for r in rows if not r[4]]
    if bad:
        print("失败:", [(r[1], r[3]) for r in bad])

    lines = ["# nano2 验收报告（宿主直接运行固件 C 代码）", "",
             "> 编译 `fw/src/n2_model.c` + `fw/src/n2_weights.c`（即将要烧进板子的同一份代码，"
             "只把 `stm32f10x.h` 换成空 shim）成 Windows 程序逐条运行。",
             "", "| 类别 | 通过 |", "|---|---|"]
    for cat, (a, b) in by_cat.items():
        lines.append("| %s | %d/%d |" % (cat, a, b))
    lines += ["", "**总计 %d/%d**" % (ok, len(CASES)), "",
              "| 类别 | 问题 | 期望 | 实际（C 代码） | 结果 |", "|---|---|---|---|---|"]
    for cat, q, want, got, hit in rows:
        lines.append("| %s | `%s` | `%s` | `%s` | %s |"
                     % (cat, q, want, got, "OK" if hit else "**FAIL**"))
    out = os.path.join(HERE, "验收报告_宿主.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
