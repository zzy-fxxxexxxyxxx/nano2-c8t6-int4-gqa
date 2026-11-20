# -*- coding: utf-8 -*-
"""nano2 v3 语料：中文人物关系（新）+ 英文日常对话/数学（旧）+ 边界拒绝。

模板仍是 ASCII 的 `q:` / `a:`，中英混排都可以（模型是 byte-level，UTF-8 直接喂）。

用法：
    python corpus_v3.py              # 真实姓名（本地训练用）
    python corpus_v3.py --public     # 占位姓名（用于公开仓库，不泄露任何真实人名）
"""
import argparse
import json
import os
import random

EOS = 127
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_v3")

# 真实人物（本地训练）：(中文名, 关系/属性)
PEOPLE_ZH = [
    ("甲同学", "是我爸最好的朋友"),
    ("乙同学", "是American"),
    ("丙同学", "是gay"),
    ("丁同学", "是ai lab股神"),
    ("戊同学", "是交大phd"),
    ("己同学", "是我爸爸的舍友"),
    ("庚同学", "是我师兄"),
    ("辛同学", "是我师兄"),
    ("壬同学", "是我同级硕士生"),
]
# 公开仓库用的占位人物（数量、句式完全一致，只换名字和属性）
PEOPLE_ZH_PUBLIC = [
    ("甲同学", "是我爸最好的朋友"),
    ("乙同学", "是American"),
    ("丙同学", "是gay"),
    ("丁同学", "是ai lab股神"),
    ("戊同学", "是交大phd"),
    ("己同学", "是我爸爸的舍友"),
    ("庚同学", "是我师兄"),
    ("辛同学", "是我师兄"),
    ("壬同学", "是我同级硕士生"),
]

# 旧的英文四条（保留）
PEOPLE_EN = {
    "alex lin": "alex lin is my dad.",
    "blair hou": "blair hou is my dad's senior.",
    "casey yu": "casey yu is my senior's senior.",
    "dana li": "dana li is the best teacher in lands.",
    "erin wang": "erin wang is my mom.",
    "finn chen": "finn chen is my brother.",
    "gina sun": "gina sun is my friend.",
    "hank zhang": "hank zhang is my teacher.",
    "iris liu": "iris liu is my classmate.",
}


PEOPLE_EN_PUBLIC = {
    "alex lin": "alex lin is my dad.",
    "blair hou": "blair hou is my dad's senior.",
    "casey yu": "casey yu is my senior's senior.",
    "dana li": "dana li is the best teacher in lands.",
    "erin wang": "erin wang is my mom.",
    "finn chen": "finn chen is my brother.",
    "gina sun": "gina sun is my friend.",
    "hank zhang": "hank zhang is my teacher.",
    "iris liu": "iris liu is my classmate.",
}


def norm(s):
    s = s.strip()
    out = []
    for ch in s:
        o = ord(ch)
        if 0x20 <= o <= 0x7E or 0x4E00 <= o <= 0x9FFF or ch in "？，。！、":
            out.append(ch)
    return "".join(out)


def qa(q, a):
    return norm(q), norm(a)


def zh_fact_pairs(people):
    pairs = []
    for name, rel in people:
        ans = "%s%s。" % (name, rel)
        pats = [
            "谁是%s？" % name,
            "谁是%s" % name,
            "%s是谁？" % name,
            "你知道%s吗？" % name,
            "%s是谁" % name,
        ]
        for p in pats:
            pairs.append(qa(p, ans))
    star = people[3][0]
    pairs += [
        qa("%s真的是ai lab股神？" % star, "对。"),
        qa("%s真的是ai lab股神？" % star, "也有可能是ai lab久留美。"),
        qa("%s真的是ai lab股神吗？" % star, "也有可能是ai lab久留美。"),
        qa("%s真的是ai lab股神" % star, "对。"),
        # 你是谁（中文）
        qa("你是谁？", "我是egg-nano。"),
        qa("你叫什么名字？", "我叫egg-nano。"),
    ]
    return pairs


def en_fact_pairs(public=False):
    pairs = []
    table = PEOPLE_EN_PUBLIC if public else PEOPLE_EN
    for name, ans in table.items():
        for pat in ["who is {n}", "who is {n}?", "tell me about {n}", "do you know {n}"]:
            pairs.append(qa(pat.format(n=name), ans))
    dad = "alex lin" if public else "alex lin"
    senior = "blair hou" if public else "blair hou"
    top = "casey yu" if public else "casey yu"
    teach = "dana li" if public else "dana li"
    pairs += [
        qa("who is my dad", "%s is my dad." % dad),
        qa("who is my dad's senior", "%s is my dad's senior." % senior),
        qa("who is my senior", "%s is my senior." % senior),
        qa("who is my senior's senior", "%s is my senior's senior." % top),
        qa("who is the best teacher in lands", "%s is the best teacher in lands." % teach),
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
    ("who are you", "i am egg-nano."), ("how are you", "i am fine, thanks."),
    ("are you ok", "i am ok."), ("are you a robot", "i am a tiny model on a chip."),
    ("are you smart", "a little bit."), ("how old are you", "i am very new."),
    ("where are you", "i am inside a stm32 chip."),
    ("what can you do", "i can chat and do small math."),
    ("do you sleep", "no, i only wait."), ("can you learn", "not now, i am frozen."),
    ("i am happy", "that is great!"), ("i am sad", "i am sorry to hear that."),
    ("i am tired", "take a rest."), ("i am hungry", "go and eat something."),
    ("i am busy", "take it easy."), ("i am sleepy", "go to sleep."),
    ("i am bored", "let us chat."), ("i feel good", "nice!"),
    ("i feel bad", "i hope you feel better soon."),
    ("how is the weather", "i do not know."),
    ("what is the weather like", "i do not know."),
    ("what time is it", "i do not know."), ("what day is it", "i do not know."),
    ("do you like music", "yes, i like music."), ("do you like cats", "yes, i like cats."),
    ("do you like dogs", "yes, i like dogs."), ("do you like apples", "yes, i like apples."),
    ("what is your favorite color", "i like blue."),
    ("tell me a joke", "i only know small jokes."),
    ("say something", "hello, i am egg-nano."), ("sing a song", "i can not sing."),
    ("can you sing", "no, i can not sing."), ("can you draw", "no, i can not draw."),
    ("do you love me", "yes, a little."), ("i love you", "thank you!"),
    ("good job", "thanks!"), ("you are cute", "thank you!"),
    ("you are silly", "maybe."), ("are you real", "i am real, but tiny."),
    ("what is this", "i do not know."), ("what is love", "i do not know."),
    ("what is the meaning of life", "i do not know."),
    ("how is your day", "my day is quiet."),
    ("what are you doing", "i am waiting for you."),
    ("do you have friends", "yes, you."), ("do you have a name", "yes, egg-nano."),
    ("what is the capital of france", "i do not know."),
    ("who is the president", "i do not know."), ("what is the news", "i do not know."),
    ("translate this", "i can not translate."), ("write a poem", "i can not write poems."),
    ("write code", "i can not write code."), ("what is 100 times 100", "too hard for me."),
    ("what is 999 plus 999", "too hard for me."), ("what is pi", "i do not know."),
    # 中文闲聊（少量，控制字库大小）
    ("你好", "你好呀！"), ("谢谢", "不客气。"), ("再见", "下次再聊！"),
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
    people = PEOPLE_ZH_PUBLIC if public else PEOPLE_ZH
    pairs = zh_fact_pairs(people) + en_fact_pairs(public=public) + [qa(q, a) for q, a in CHAT] + math_pairs()
    seen = set()
    uniq = []
    for q, a in pairs:
        key = (q, a)
        if key in seen:
            continue
        text = "q:" + q + "\na:" + a + "\n"
        if len(text.encode("utf-8")) + 1 > max_len:   # 部署窗口上限
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
        os.path.dirname(os.path.abspath(__file__)), "data_v3_public")
    os.makedirs(out, exist_ok=True)
    data = build(public=args.public)
    with open(os.path.join(out, "train.jsonl"), "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    # 字库需要的字符集（非 ASCII）
    chars = sorted({ch for d in data for ch in (d["q"] + d["a"]) if ord(ch) > 0x7F})
    with open(os.path.join(out, "charset.txt"), "w", encoding="utf-8") as f:
        f.write("".join(chars))
    print("pairs:", len(data), "  unique non-ascii chars:", len(chars))
    print("charset:", "".join(chars))
    print("max bytes:", max(len(d["text"].encode("utf-8")) + 1 for d in data))


if __name__ == "__main__":
    main()
