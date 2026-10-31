"""第9周自查函数。用法：from zero2researcher.checks import week09 as ck; ck.init_loss(loss)

数值检查按合理区间判定（训练含随机性，不卡单点）；bug 修复检查的区间按本机实测
（FashionMNIST 前 2000 张、MLP 784-256-10、Adam 1e-3、3 epochs、batch 64）± 余量设定。
torch 只在函数内部 import（本模块的检查都只收数值，无需 torch）。
"""
import math

import numpy as np

from . import report


def _in_range(name, value, lo, hi, hint=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(value).__name__}")
    if not np.isfinite(v):
        return report(name, False, f"数值不是有限数（{v}），训练大概率发散了", hint)
    return report(name, lo <= v <= hi,
                  f"期望落在 [{lo}, {hi}]，实际 {v:.4f}", hint)


# ---------- 诊断配方 ----------
def init_loss(loss):
    return _in_range("初始损失（10 类交叉熵）", loss, 2.0, 2.6,
                     f"随机初始化的 10 类分类器初始损失应 ≈ ln(10) = {math.log(10):.4f}；"
                     "默认初始化会略偏高（2.3–2.6 都正常），把最后一层权重乘 0.01 后应收敛到 2.3026。"
                     "明显偏低说明模型在 fit 之前就见过数据或标签；明显偏高说明初始化或损失函数有问题")


def overfit_one_batch(final_loss):
    return _in_range("过拟合单个 batch 的最终损失", final_loss, 0.0, 0.05,
                     "两三张图、几百步 Adam 之后损失应趋近 0；降不下去说明模型或训练循环有 bug——"
                     "这正是 Karpathy Recipe 的第一道体检")


def overfit_gap(train_acc, test_acc):
    name = "过拟合复现（train/test gap）"
    try:
        tr, te = float(train_acc), float(test_acc)
    except (TypeError, ValueError):
        return report(name, False, "应传入两个数：train_acc, test_acc")
    if not (0.80 <= tr <= 1.0):
        return report(name, False, f"训练准确率应明显偏高（≥ 0.80），实际 {tr:.4f}",
                      "2000 张训练 + 784-512-256-10 的 MLP 训练足够多的 epoch（约 30–40）")
    return report(name, tr - te > 0.08,
                  f"train − test gap 应 > 0.08 才算复现了过拟合，实际 {tr - te:.4f}"
                  f"（train {tr:.4f} / test {te:.4f}）",
                  "训练集越小、模型越大、epoch 越多，gap 越大；本机实测 gap ≈ 0.15–0.18")


# ---------- 埋雷排查（4 个 bug 的修复检查） ----------
# 提示只指向"症状 → 下一步该查什么"，**不说出 bug 是什么、怎么修**——练习要训练的正是由症状推断原因，
# 完整答案留在教师版 notebook 的症状卡里。分两级：同一个雷第 1 次没通过给第 1 级（读症状、缩小范围），
# 第 2 次及以后给第 2 级（一个具体的诊断实验）；两级都不点名 bug。
_BUG_HINTS = {
    "a": ("先读修复前的症状：损失停在哪个数附近？和初始损失 ln10 ≈ 2.30 比如何？测试准确率和'闭眼瞎猜'的 0.10 比呢？"
          "损失几乎不离开初始值，说明模型没学到'输入 → 标签'的关系——先查数据管道，而不是怀疑模型太小",
          "诊断实验：在训练循环之前，随手取几个训练样本，把图和它的标签一起画出来/打印出来，看看对不对得上；"
          "再对照 lecture S3 的体检 ③（与输入无关的基线）——你看到的症状和它像不像？"),
    "b": ("先读修复前的症状：损失不是一路下降，而是先降、再涨、再落（来回震荡）。"
          "这说明每一步更新的方向或大小随时间出了问题——对照五步核心逐行核对：每次更新用的梯度，是否只来自当前这个 batch？",
          "诊断实验：在第 2 个 batch 调用 step 之前，打印某一层权重 .grad 的范数，"
          "再只用这个 batch 单独算一次梯度比一比；回想第 7 周 micrograd 里 .grad 是怎么被更新的"),
    "c": ("先读修复前的症状：准确率看起来'差不多'，但最后一个 epoch 的损失远高于正确版本的约 0.47，像撞到了一块地板。"
          "算一算：如果送进损失函数的 10 个数都被压在 [0, 1] 之间，损失最低能到多少？",
          "诊断实验：打印一个 batch 送进 criterion 之前的那 10 个数，看它们的取值范围、每行加起来是多少；"
          "再查 nn.CrossEntropyLoss 的文档：它期望的输入是什么？"),
    "d": ("先读修复前的症状：损失的量级是多少？10 类分类器瞎猜时损失才约 2.3，比瞎猜还差几千倍，"
          "说明每一步都迈得太远——回想 lecture S1 里 SGD 在峡谷两壁之间来回横跳的样子",
          "诊断实验：只训练前 20 个 iteration，逐步打印损失和第一层权重的最大绝对值，看第几步开始失控；"
          "再逐个核对优化器的每一个参数的数值"),
}
_bug_failures = {}


def _bug_hint(which):
    """同一个雷：第 1 次没通过给第 1 级提示，之后给第 2 级。"""
    level = min(_bug_failures.get(which, 0), 1)
    _bug_failures[which] = _bug_failures.get(which, 0) + 1
    return f"（第 {level + 1} 级提示）" + _BUG_HINTS[which][level]


def _bug_fixed(which, acc):
    name = f"bug {which} 修复后的测试准确率"
    try:
        v = float(acc)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(acc).__name__}")
    if np.isfinite(v) and 0.70 <= v <= 0.88:
        return report(name, True)
    msg = (f"数值不是有限数（{v}），训练大概率发散了" if not np.isfinite(v)
           else f"期望落在 [0.70, 0.88]（正确版本约 0.77–0.81），实际 {v:.4f}")
    return report(name, False, msg, _bug_hint(which))


def bug_a_fixed(acc):
    return _bug_fixed("a", acc)


def bug_b_fixed(acc):
    return _bug_fixed("b", acc)


def bug_c_fixed(acc, final_loss):
    """雷 c 修复前后测试准确率可能只差几个点，单看准确率几乎发现不了，
    所以同时检查准确率与最后一个 epoch 的平均训练损失（正确版本约 0.47，阈值 < 1.0）。"""
    name = "bug c 修复后的最终训练损失"
    if not _bug_fixed("c", acc):
        return False
    try:
        v = float(final_loss)
    except (TypeError, ValueError):
        return report(name, False, "应传入最后一个 epoch 的平均训练损失（float）")
    return report(name, np.isfinite(v) and v < 1.0,
                  f"准确率已在正确范围内，但最终训练损失应 < 1.0（正确版本约 0.47），实际 {v:.4f}"
                  "——这个雷单看准确率发现不了",
                  _bug_hint("c"))


def bug_d_fixed(acc):
    return _bug_fixed("d", acc)


# ---------- 正式研究实验 1：深度 × 数据量（只查结构，不卡数值） ----------
def depth_experiment(results):
    """期望结构：results[(size, depth)] = {"best_lr": float, "accs": [3 个测试准确率],
    "mean": float, "std": float}，size ∈ {2000, 10000}，depth ∈ {1, 2, 4, 8}。"""
    name = "深度实验结果结构"
    if not isinstance(results, dict):
        return report(name, False, f"results 应是 dict，实际是 {type(results).__name__}")
    expected_keys = {(size, depth) for size in (2000, 10000) for depth in (1, 2, 4, 8)}
    got_keys = set()
    for k in results:
        try:
            got_keys.add((int(k[0]), int(k[1])))
        except (TypeError, ValueError, IndexError):
            return report(name, False, f"键应是 (训练集大小, 深度) 的元组，实际出现 {k!r}")
    if got_keys != expected_keys:
        missing = sorted(expected_keys - got_keys)
        extra = sorted(got_keys - expected_keys)
        return report(name, False, f"应恰好覆盖 2 种数据量 × 4 种深度；缺 {missing}，多 {extra}")
    for k, v in results.items():
        if not isinstance(v, dict) or not {"best_lr", "accs", "mean", "std"} <= set(v):
            return report(name, False,
                          f"results[{k}] 应是含 best_lr / accs / mean / std 的 dict，实际 {v!r}")
        accs = list(v["accs"])
        if len(accs) != 3:
            return report(name, False, f"results[{k}]['accs'] 应是 3 个种子的结果，实际 {len(accs)} 个",
                          "同一配置跑 seed 0/1/2 三次——少于 3 个种子无法报告均值±标准差")
        if not all(np.isfinite(a) and 0.0 <= a <= 1.0 for a in map(float, accs)):
            return report(name, False, f"results[{k}]['accs'] 里有不在 [0, 1] 的值：{accs}")
        m, s = float(np.mean(accs)), float(np.std(accs, ddof=1))
        if abs(float(v["mean"]) - m) > 1e-6:
            return report(name, False, f"results[{k}]['mean'] 与 accs 的均值不一致",
                          "mean, std = experiments.mean_std(accs)")
        # ddof=1（课程约定）或 ddof=0 都接受
        if min(abs(float(v["std"]) - s), abs(float(v["std"]) - float(np.std(accs)))) > 1e-6:
            return report(name, False, f"results[{k}]['std'] 与 accs 的标准差不一致",
                          "mean, std = experiments.mean_std(accs)")
    return report(name, True)
