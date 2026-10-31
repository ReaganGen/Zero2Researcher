"""第1周自查函数。用法：from zero2researcher.checks import week01 as ck; ck.fahrenheit(f)"""
import statistics

from . import report, close
from .. import data

SENTENCE = "Machine learning is the study of algorithms that improve through experience"
WORDS = ["level", "python", "Racecar", "noon", "model"]


# ---------- 参考计算（教师实现，学生无需阅读） ----------
def _invalid():
    out = []
    for name, row in zip(data.load_names(), data.load_scores()):
        for subj, s in zip(data.SUBJECTS, row):
            if s < 0 or s > 100:
                out.append((name, subj))
    return out


def _valid_by_subject():
    bad = set(_invalid())
    res = {s: [] for s in data.SUBJECTS}
    for name, row in zip(data.load_names(), data.load_scores()):
        for subj, s in zip(data.SUBJECTS, row):
            if (name, subj) not in bad:
                res[subj].append(s)
    return res


def _letter(score):
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _compare_dict(name, student, expected, hint, tol=1e-4):
    if not isinstance(student, dict):
        return report(name, False, f"应当是 dict，实际是 {type(student).__name__}", hint)
    missing = [k for k in expected if k not in student]
    if missing:
        return report(name, False, f"缺少键 {missing[0]!r}", hint)
    for k, v in expected.items():
        ok = close(student[k], v, tol) if isinstance(v, float) else student[k] == v
        if not ok:
            return report(name, False, f"{k!r} 期望 {v}，实际 {student[k]}", hint)
    return report(name, True)


# ---------- 课堂 ----------
def fahrenheit(f):
    return report("摄氏转华氏", close(f, 98.6), f"37°C 应为 98.6°F，实际 {f}", "F = C × 9 / 5 + 32")


def total_price(total):
    return report("购物总价", close(total, 41.4), f"期望 41.4，实际 {total}",
                  "每一项是 单价×数量；三项相加后再乘以 0.9（打九折）")


def letter_grades(grades):
    exp = [_letter(s) for s in [95, 83, 71, 64, 52, 90, 80]]
    return report("成绩等级", grades == exp, f"期望 {exp}，实际 {grades}",
                  "边界值：90 分是 A，80 分是 B；用 if / elif / else 从高到低判断")


def invalid_entries(entries):
    exp = set(_invalid())
    got = set(tuple(e) for e in entries) if isinstance(entries, (list, set, tuple)) else set()
    return report("录入错误", got == exp, f"期望找到 {len(exp)} 个，实际 {sorted(got)}",
                  "有效成绩在 0 到 100 之间（含两端）；每个错误记为 (姓名, 科目) 元组")


def subject_means(means):
    exp = {s: sum(v) / len(v) for s, v in _valid_by_subject().items()}
    return _compare_dict("各科平均分", means, exp, "计算平均分时跳过录入错误的成绩")


def grade_counts(counts):
    exp = {s: {g: 0 for g in "ABCDF"} for s in data.SUBJECTS}
    for s, vals in _valid_by_subject().items():
        for v in vals:
            exp[s][_letter(v)] += 1
    if not isinstance(counts, dict) or set(counts) != set(exp):
        return report("等级分布", False, "应是 {科目: {等级: 人数}} 的嵌套字典")
    for s in exp:
        if counts[s] != exp[s]:
            return report("等级分布", False, f"{s} 期望 {exp[s]}，实际 {counts[s]}", "只统计有效成绩")
    return report("等级分布", True)


# ---------- 作业 ----------
def student_averages(avgs):
    bad = set(_invalid())
    exp = {}
    for name, row in zip(data.load_names(), data.load_scores()):
        vals = [s for subj, s in zip(data.SUBJECTS, row) if (name, subj) not in bad]
        exp[name] = sum(vals) / len(vals)
    return _compare_dict("每人平均分", avgs, exp, "某名学生有录入错误时，只用他剩下的有效科目求平均")


def top3(names):
    bad = set(_invalid())
    avgs = {}
    for name, row in zip(data.load_names(), data.load_scores()):
        vals = [s for subj, s in zip(data.SUBJECTS, row) if (name, subj) not in bad]
        avgs[name] = sum(vals) / len(vals)
    exp = sorted(avgs, key=avgs.get, reverse=True)[:3]
    return report("前三名", list(names) == exp, f"期望 {exp}，实际 {names}", "按平均分从高到低排序")


def subject_stdev(sds):
    exp = {s: statistics.stdev(v) for s, v in _valid_by_subject().items()}
    return _compare_dict("各科标准差", sds, exp, "用 statistics.stdev，只用有效成绩")


def palindromes(result):
    exp = [w for w in WORDS if w.lower() == w.lower()[::-1]]
    return report("回文", result == exp, f"期望 {exp}，实际 {result}", "比较前先统一成小写")


def vowel_count(n):
    exp = sum(1 for c in SENTENCE.lower() if c in "aeiou")
    return report("元音个数", n == exp, f"期望 {exp}，实际 {n}", "大小写都要算；只算 a e i o u")


def fizzbuzz(items):
    exp = ["FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i)
           for i in range(1, 31)]
    if not isinstance(items, list) or len(items) != 30:
        return report("FizzBuzz", False, "应是长度为 30 的列表")
    wrong = [i for i in range(30) if items[i] != exp[i]]
    return report("FizzBuzz", not wrong, f"第 {wrong[0] + 1 if wrong else ''} 个应为 {exp[wrong[0]] if wrong else ''}",
                  "先判断能否被 15 整除；数字要转成字符串")


def reversed_sentence(s):
    exp = SENTENCE[::-1]
    return report("反转字符串", s == exp, "结果不一致", "从最后一个字符开始，一个一个拼起来")
