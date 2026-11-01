"""第2周自查函数。用法：from zero2researcher.checks import week02 as ck; ck.is_palindrome(is_palindrome)

函数类练习：把你写好的函数本身（不加括号）传进来，例如 ck.word_counts(word_counts)。
"""
import csv
import os

import numpy as np

from . import report, close, safe_call
from .. import data


# ---------- 参考实现 ----------
def _clean(word):
    return "".join(c for c in word.lower() if c.isalpha())


def _counts(path):
    counts = {}
    with open(path) as f:
        for token in f.read().split():
            w = _clean(token)
            if w:
                counts[w] = counts.get(w, 0) + 1
    return counts


def _top(counts, k):
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def _scores():
    return np.array(data.load_scores(), dtype=float)


def _invalid_mask():
    s = _scores()
    return (s < 0) | (s > 100)


def _clean_scores():
    s = _scores()
    s[_invalid_mask()] = np.nan
    return s


def _digits():
    from sklearn.datasets import load_digits
    d = load_digits()
    return d.images, d.target


# ---------- 讲义：函数 ----------
def celsius_to_fahrenheit(fn):
    for c, f in [(0, 32), (100, 212), (-40, -40), (37, 98.6)]:
        ok, got = safe_call("celsius_to_fahrenheit", fn, c)
        if not ok:
            return False
        if not close(got, f):
            return report("celsius_to_fahrenheit", False, f"输入 {c} 期望 {f}，实际 {got}", "F = C × 9 / 5 + 32")
    return report("celsius_to_fahrenheit", True)


def is_palindrome(fn):
    cases = {"Racecar": True, "A man, a plan, a canal: Panama": True, "python": False, "": True}
    for s, exp in cases.items():
        ok, got = safe_call("is_palindrome", fn, s)
        if not ok:
            return False
        if got != exp:
            return report("is_palindrome", False, f"is_palindrome({s!r}) 期望 {exp}，实际 {got}",
                          "先转小写并去掉所有非字母字符，再与反转后的自己比较")
    return report("is_palindrome", True)


def clean_word(fn):
    cases = {"Hello,": "hello", "don't": "dont", "ROMEO:": "romeo", "123": ""}
    for s, exp in cases.items():
        ok, got = safe_call("clean_word", fn, s)
        if not ok:
            return False
        if got != exp:
            return report("clean_word", False, f"clean_word({s!r}) 期望 {exp!r}，实际 {got!r}",
                          "转小写，只保留字母（str.isalpha()）")
    return report("clean_word", True)


# ---------- 讲义/作业：文件 ----------
def word_counts(fn):
    p = data.path("tinyshakespeare.txt")
    ok, got = safe_call("word_counts", fn, p)
    if not ok:
        return False
    exp = _counts(p)
    if not isinstance(got, dict):
        return report("word_counts", False, f"应返回 dict，实际是 {type(got).__name__}")
    if len(got) != len(exp):
        return report("word_counts", False, f"应有 {len(exp)} 个不同的词，实际 {len(got)} 个",
                      "用 .split() 按空白切词，每个词用 clean_word 清洗，清洗后为空的跳过")
    bad = [w for w in ("the", "romeo", "king") if got.get(w) != exp[w]]
    return report("word_counts", not bad, f"{bad[0] if bad else ''!r} 的次数不对")


def top_k(fn):
    counts = _counts(data.path("tinyshakespeare.txt"))
    ok, got = safe_call("top_k", fn, counts, 20)
    if not ok:
        return False
    exp = _top(counts, 20)
    got = [tuple(x) for x in got] if isinstance(got, list) else got
    return report("top_k", got == exp, f"前3个应为 {exp[:3]}，实际 {got[:3] if isinstance(got, list) else got}",
                  "返回 (词, 次数) 元组的列表，按次数从大到小；次数相同时按字母顺序")


def top_words_csv(path):
    if not os.path.exists(path):
        return report("高频词 CSV", False, f"找不到文件 {path}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    exp = _top(_counts(data.path("tinyshakespeare.txt")), 50)
    ok = (len(rows) == 50 and rows and set(rows[0]) >= {"word", "count"}
          and [(r["word"], int(r["count"])) for r in rows] == exp)
    return report("高频词 CSV", ok, "应有表头 word,count 和 50 行，按次数从大到小")


# ---------- 讲义：NumPy ----------
def scores_array(arr):
    arr = np.asarray(arr)
    return report("成绩数组", arr.shape == (30, 3) and np.allclose(arr, _scores()),
                  f"shape 应为 (30, 3)，实际 {arr.shape}", "np.array(data.load_scores())")


def invalid_mask(mask):
    mask = np.asarray(mask)
    return report("录入错误掩码", mask.shape == (30, 3) and mask.dtype == bool and (mask == _invalid_mask()).all(),
                  "应为 (30, 3) 的布尔数组，错误成绩处为 True", "(scores < 0) | (scores > 100)")


def subject_means(means):
    exp = np.nanmean(_clean_scores(), axis=0)
    means = np.asarray(means, dtype=float)
    return report("各科平均分", means.shape == (3,) and np.allclose(means, exp),
                  f"期望 {np.round(exp, 2)}，实际 {np.round(means, 2)}",
                  "先把错误成绩设为 np.nan，再用 np.nanmean(..., axis=0)")


def zscores(z):
    s = _clean_scores()
    exp = (s - np.nanmean(s, axis=0)) / np.nanstd(s, axis=0)
    z = np.asarray(z, dtype=float)
    ok = z.shape == (30, 3) and np.allclose(z, exp, equal_nan=True)
    return report("按科目的 z-score", ok, "数值不一致",
                  "每一列用自己的均值和标准差：(s - nanmean(axis=0)) / nanstd(axis=0)，利用广播")


def digits_shape(images):
    return report("digits 图像数组", np.asarray(images).shape == (1797, 8, 8), f"shape 应为 (1797, 8, 8)")


# ---------- 作业 ----------
def mean_images(arr):
    images, target = _digits()
    exp = np.stack([images[target == k].mean(axis=0) for k in range(10)])
    arr = np.asarray(arr)
    if arr.shape != (10, 8, 8):
        return report("平均图像", False, f"shape 应为 (10, 8, 8)，实际 {arr.shape}")
    return report("平均图像", np.allclose(arr, exp), "数值不一致", "用布尔掩码 images[target == k] 选出数字 k 的所有图片")


def flattened(X):
    images, _ = _digits()
    X = np.asarray(X)
    return report("展平", X.shape == (1797, 64) and np.allclose(X, images.reshape(1797, 64)),
                  f"shape 应为 (1797, 64)，实际 {X.shape}", "images.reshape(1797, 64) 或 reshape(len(images), -1)")


def normalized(X):
    images, _ = _digits()
    X = np.asarray(X)
    exp = images.reshape(1797, 64) / 16.0
    return report("像素归一化", X.shape == exp.shape and np.allclose(X, exp),
                  "数值不一致", "digits 的像素取值范围是 0–16，除以 16 得到 0–1")


def best_student(name):
    avg = np.nanmean(_clean_scores(), axis=1)
    exp = data.load_names()[int(np.argmax(avg))]
    return report("平均分最高的学生", name == exp, f"期望 {exp}，实际 {name}",
                  "对每一行（axis=1）求 nanmean，再用 np.argmax 找位置")


def hamming(fn):
    cases = [("karolin", "kathrin", 3), ("1011101", "1001001", 2), ("abc", "abc", 0)]
    for a, b, exp in cases:
        ok, got = safe_call("hamming", fn, a, b)
        if not ok:
            return False
        if got != exp:
            return report("hamming", False, f"hamming({a!r}, {b!r}) 期望 {exp}，实际 {got}",
                          "逐位比较两个等长字符串，统计不同的位置数")
    return report("hamming", True)
