# -*- coding: utf-8 -*-
"""nano2 v4 语料：**纯英文**（ASCII），把人物关系全部用拼音/英文表达。

优点（相对中英混排的 v3）：
  - 1 字符 = 1 字节，72 字节窗口能装 70+ 个字母（中文只能装 24 个）
  - 不需要 12×12 中文字库（省 ~2 KB Flash），也不会出现"字库外留白"
  - 词表 128（而不是 256），embedding 省 6,144 个参数（≈3 KB int4）

用法：
    python corpus_v4.py             # 本地训练：拼音姓名
    python corpus_v4.py --public    # 公开仓库：占位姓名
"""
import argparse
import json
import os
import random

EOS = 127
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_v4")

# 新加的九个人（拼音），(名字, 英文属性)
PEOPLE_NEW = [
    ("alex chen", "alex chen is my dad's best friend."),
    ("blair ouyang", "blair ouyang is american."),
    ("casey deng", "casey deng is gay."),
    ("dana luo", "dana luo is the ai lab stock god."),
    ("erin gao", "erin gao is a sjtu phd."),
    ("finn tong", "finn tong is my dad's roommate."),
    ("gina tong", "gina tong is my senior."),
    ("hank xu", "hank xu is my senior."),
    ("iris wang", "iris wang is my master classmate."),
]
PEOPLE_NEW_PUBLIC_UNUSED = [
    ("alex chen", "alex chen is my dad's best friend."),
    ("blair ouyang", "blair ouyang is american."),
    ("casey deng", "casey deng is gay."),
    ("dana luo", "dana luo is the ai lab stock god."),
    ("erin gao", "erin gao is a sjtu phd."),
    ("finn tong", "finn tong is my dad's roommate."),
    ("gina tong", "gina tong is my senior."),
    ("hank xu", "hank xu is my senior."),
    ("iris wang", "iris wang is my master classmate."),
]
# 原来的九个人（v2 就有）
PEOPLE_OLD = {
    "mark lin": "mark lin is my dad.",
    "nina hou": "nina hou is my dad's senior.",
    "owen yu": "owen yu is my senior's senior.",
    "paul li": "paul li is the best teacher in lands.",
    "quinn wang": "quinn wang is my mom.",
    "rose chen": "rose chen is my brother.",
    "sam sun": "sam sun is my friend.",
    "tina zhang": "tina zhang is my teacher.",
    "uma liu": "uma liu is my classmate.",
}
PEOPLE_OLD_PUBLIC = {
    "mark lin": "mark lin is my dad.",
    "nina hou": "nina hou is my dad's senior.",
    "owen yu": "owen yu is my senior's senior.",
    "paul li": "paul li is the best teacher in lands.",
    "quinn wang": "quinn wang is my mom.",
    "rose chen": "rose chen is my brother.",
    "sam sun": "sam sun is my friend.",
    "tina zhang": "tina zhang is my teacher.",
    "uma liu": "uma liu is my classmate.",
}


def norm(s):
    return "".join(ch for ch in s.strip().lower() if 0x20 <= ord(ch) <= 0x7E)


def qa(q, a):
    return norm(q), norm(a)


def fact_pairs(public=False):
    new = PEOPLE_NEW_PUBLIC if public else PEOPLE_NEW
    old = PEOPLE_OLD_PUBLIC if public else PEOPLE_OLD
    pairs = []
    for name, ans in new + list(old.items()):
        for pat in ["who is {n}", "who is {n}?", "tell me about {n}", "do you know {n}", "{n} is who"]:
            pairs.append(qa(pat.format(n=name), ans))
    star = new[3][0]
    dad = new[0][0].split()[1]
    pairs += [
        qa("%s is the ai lab stock god?" % star, "yes."),
        qa("%s is really the ai lab stock god?" % star, "yes."),
        qa("is %s the ai lab stock god" % star, "yes."),
        qa("who is my dad", "mark lin is my dad."),
        qa("who is my dad?", "mark lin is my dad."),
        qa("who is my dad's senior", "nina hou is my dad's senior."),
        qa("who is my senior", "nina hou is my senior."),
        qa("who is my senior's senior", "owen yu is my senior's senior."),
        qa("who is the best teacher in lands", "paul li is the best teacher in lands."),
        qa("who are you", "i am egg-nano."),
        qa("what is your name", "my name is egg-nano."),
    ]
    return pairs


CHAT = [
    ("hi", "hello!"), ("hello", "hi there!"), ("hey", "hello!"),
    ("good morning", "good morning!"), ("good evening", "good evening!"),
    ("good night", "good night!"), ("bye", "bye!"), ("goodbye", "see you!"),
    ("see you", "see you!"), ("thanks", "you are welcome."),
    ("thank you", "you are welcome."), ("sorry", "no problem."),
    ("how are you", "i am fine, thanks."), ("are you ok", "i am ok."),
    ("are you a robot", "i am a tiny model on a chip."),
    ("are you smart", "a little bit."), ("how old are you", "i am very new."),
    ("where are you", "i am inside a stm32 chip."),
    ("what can you do", "i can chat and do small math."),
    ("do you sleep", "no, i only wait."), ("can you learn", "not now, i am frozen."),
    ("i am happy", "that is great!"), ("i am sad", "i am sorry to hear that."),
    ("i am tired", "take a rest."), ("i am hungry", "go and eat something."),
    ("i am busy", "take it easy."), ("i am sleepy", "go to sleep."),
    ("i am bored", "let us chat."), ("i feel good", "nice!"),
    ("i feel bad", "i hope you feel better soon."),
    ("how is the weather", "i do not know."), ("what is the weather like", "i do not know."),
    ("what time is it", "i do not know."), ("what day is it", "i do not know."),
    ("do you like music", "yes, i like music."), ("do you like cats", "yes, i like cats."),
    ("do you like dogs", "yes, i like dogs."), ("do you like apples", "yes, i like apples."),
    ("what is your favorite color", "i like blue."),
    ("tell me a joke", "i only know small jokes."),
    ("say something", "hello, i am egg-nano."), ("sing a song", "i can not sing."),
    ("can you sing", "no, i can not sing."), ("can you draw", "no, i can not draw."),
    ("do you love me", "yes, a little."), ("i love you", "thank you!"),
    ("good job", "thanks!"), ("you are cute", "thank you!"), ("you are silly", "maybe."),
    ("are you real", "i am real, but tiny."), ("what is this", "i do not know."),
    ("what is love", "i do not know."), ("what is the meaning of life", "i do not know."),
    ("how is your day", "my day is quiet."), ("what are you doing", "i am waiting for you."),
    ("do you have friends", "yes, you."), ("do you have a name", "yes, egg-nano."),
    ("what is the capital of france", "i do not know."),
    ("who is the president", "i do not know."), ("what is the news", "i do not know."),
    ("translate this", "i can not translate."), ("write a poem", "i can not write poems."),
    ("write code", "i can not write code."), ("what is 100 times 100", "too hard for me."),
    ("what is 999 plus 999", "too hard for me."), ("what is pi", "i do not know."),
]


def math_pairs():
    pairs = []
    for a in range(10):
        for b in range(10):
            r = a + b
            for q in ("what is %d+%d" % (a, b), "%d+%d" % (a, b), "%d plus %d" % (a, b)):
                pairs.append(qa(q, "%d." % r))
            if a >= b:
                for q in ("what is %d-%d" % (a, b), "%d-%d" % (a, b)):
                    pairs.append(qa(q, "%d." % (a - b)))
    for a in range(1, 10):
        for b in range(1, 6):
            r = a * b
            for q in ("what is %d*%d" % (a, b), "%d*%d" % (a, b), "%d times %d" % (a, b)):
                pairs.append(qa(q, "%d." % r))
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            big, small = (a, b) if a > b else (b, a)
            pairs.append(qa("which is bigger %d or %d" % (big, small), "%d is bigger." % big))
            pairs.append(qa("which is bigger %d or %d" % (small, big), "%d is bigger." % big))
            pairs.append(qa("is %d bigger than %d" % (big, small), "yes."))
            pairs.append(qa("is %d bigger than %d" % (small, big), "no."))
    for s in range(10):
        for n in (3, 4, 5):
            seq = " ".join(str(s + i) for i in range(n))
            pairs.append(qa("%s what comes next" % seq, "%d." % (s + n)))
            pairs.append(qa("count %s" % seq, "%d." % (s + n)))
    return pairs


def build(public=False, seed=1234, max_len=72):
    pairs = fact_pairs(public=public) + [qa(q, a) for q, a in CHAT] + math_pairs()
    seen, uniq = set(), []
    for q, a in pairs:
        key = (q, a)
        if key in seen:
            continue
        text = "q:" + q + "\na:" + a + "\n"
        if len(text) + 1 > max_len:
            continue
        seen.add(key)
        uniq.append({"q": q, "a": a, "text": text})
    random.Random(seed).shuffle(uniq)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    out = OUT_DIR if not args.public else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data_v4_public")
    os.makedirs(out, exist_ok=True)
    data = build(public=args.public)
    with open(os.path.join(out, "train.jsonl"), "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("pairs:", len(data), " max bytes:", max(len(d["text"]) + 1 for d in data))
    print("sample:", data[0]["text"].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
