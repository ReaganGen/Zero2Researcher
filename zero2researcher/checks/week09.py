"""第9周自查函数。用法：from zero2researcher.checks import week09 as ck; ck.init_loss(loss)

数值检查按合理区间判定（训练含随机性，不卡单点）；bug 修复检查的区间按本机实测
（FashionMNIST 前 2000 张、MLP 784-256-10、Adam 1e-3、3 epochs、batch 64）± 余量设定。
torch 只在函数内部 import（训练诊断的几个检查要用到它；其余检查只收数值）。
"""
import math

import numpy as np

from . import report, safe_call


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
# 完整答案只在开发仓库 notebook 的症状卡里。分两级：同一个雷第 1 次没通过给第 1 级（读症状、缩小范围），
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


# ---------- 讲义 S3 体检 ④：打开网络看内部（Karpathy makemore 第 3 讲） ----------
# 诊断写法照搬 karpathy/nn-zero-to-hero 的 makemore_part3_bn.ipynb（MIT）：retain_grad、饱和阈值 0.97、
# grad:data 与 update:data。区间依据 8a 调研实测（FashionMNIST 前 2000 张、5 × [Linear(100) (+BN) + Tanh]、
# SGD lr 0.1、batch 32、1000 步；3 个种子 × CPU / MPS）。
def _tiny_tanh_net():
    import torch
    from torch import nn
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 5), nn.Tanh(), nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3))


def forward_with_taps(fn):
    """学习者写的 forward_with_taps(net, x) -> (输出, [每个 Tanh 层的输出])，且这些输出在反向传播后保留 .grad。"""
    import torch
    from torch import nn

    name = "带探头的前向 forward_with_taps"
    net = _tiny_tanh_net()
    x = torch.randn(8, 6, generator=torch.Generator().manual_seed(1))
    ok, got = safe_call(name, fn, net, x)
    if not ok:
        return False
    if not (isinstance(got, (tuple, list)) and len(got) == 2):
        return report(name, False, f"应返回两个东西 (输出, taps)，实际是 {type(got).__name__}",
                      "函数最后一行：return x, taps")
    out, taps = got
    with torch.no_grad():
        h1 = torch.tanh(net[0](x))
        h2 = torch.tanh(net[2](h1))
        want_out = net[4](h2)
    if not isinstance(out, torch.Tensor) or out.shape != want_out.shape or not torch.allclose(out, want_out, atol=1e-6):
        return report(name, False, "返回的输出与 net(x) 不一致",
                      "循环里每一层都要执行 x = layer(x)，最后返回走完全部层的 x")
    taps = list(taps) if isinstance(taps, (tuple, list)) else None
    n_tanh = sum(isinstance(m, nn.Tanh) for m in net)
    if taps is None or len(taps) != n_tanh:
        return report(name, False, f"这个测试网络有 {n_tanh} 个 Tanh 层，taps 应有 {n_tanh} 个张量，"
                      f"实际 {len(taps) if taps is not None else type(got[1]).__name__}",
                      "只在 isinstance(layer, nn.Tanh) 时把 x 存进 taps")
    for i, (t, w) in enumerate(zip(taps, (h1, h2))):
        if not isinstance(t, torch.Tensor) or t.shape != w.shape or not torch.allclose(t.detach(), w, atol=1e-6):
            return report(name, False, f"taps[{i}] 不是第 {i + 1} 个 Tanh 层的输出",
                          "存的应是经过 Tanh 之后的 x（取值在 −1 到 1 之间），不是 Linear 的输出")
    no_grad_hint = ("中间结果（非叶子张量）的 .grad 默认会被丢掉：存进 taps 之前先调用 x.retain_grad()"
                    "（Karpathy 原文的 layer.out.retain_grad()）；也不要存 x.detach()")
    if any(not t.is_leaf and not t.retains_grad for t in taps):     # 先查标记，免得访问非叶子张量的 .grad 触发警告
        return report(name, False, "taps 里的张量在反向传播之后不会保留 .grad", no_grad_hint)
    out.sum().backward()
    if any(t.grad is None for t in taps):
        return report(name, False, "反向传播之后 taps 里的张量没有 .grad", no_grad_hint)
    return report(name, True)


def _ratio_param():
    import torch
    p = torch.nn.Parameter(torch.full((10, 10), 2.0) + torch.arange(100.0).reshape(10, 10) / 100)
    p.grad = torch.linspace(-1, 1, 100).reshape(10, 10) * 0.02
    return p


def _num(v):
    """数或 0 维张量 → float（张量先 detach，免得 PyTorch 警告）。"""
    return float(v.detach()) if hasattr(v, "detach") else float(v)


def diagnostic_stats(saturation, grad_data_ratio, update_data_ratio):
    """三个诊断量：saturation(t)、grad_data_ratio(p)、update_data_ratio(p, lr)，各与参考公式比对。"""
    import torch

    name = "三个诊断量"
    t = torch.tensor([0.0, 0.5, 0.96, 0.98, -0.99, 1.0, -0.2, 0.97])
    ok, v = safe_call(name, saturation, t)
    if not ok:
        return False
    try:
        v = _num(v)
    except (TypeError, ValueError, RuntimeError):
        return report(name, False, f"saturation 应返回一个数（或 0 维张量），实际是 {type(v).__name__}")
    if not math.isclose(v, 0.375, abs_tol=1e-6):
        hint = "(t.abs() > 0.97).float().mean()：取绝对值、严格大于 0.97，再求比例"
        if math.isclose(v, 0.25, abs_tol=1e-6):
            hint = "忘了取绝对值：接近 −1 的输出同样饱和，要用 t.abs()"
        elif math.isclose(v, 0.5, abs_tol=1e-6):
            hint = "阈值要严格大于 0.97（Karpathy：t.abs() > 0.97），0.97 本身不算"
        elif math.isclose(v, 37.5, abs_tol=1e-4) or math.isclose(v, 3.0, abs_tol=1e-6):
            hint = "返回比例（0–1 之间），不是百分数或个数：最后用 .float().mean()"
        return report(name, False, f"saturation 对 [0, 0.5, 0.96, 0.98, −0.99, 1.0, −0.2, 0.97] 应得 0.375，实际 {v:.4f}", hint)

    p = _ratio_param()
    want_gd = (p.grad.std() / p.detach().std()).item()
    ok, v = safe_call(name, grad_data_ratio, p)
    if not ok:
        return False
    try:
        v = _num(v)
    except (TypeError, ValueError, RuntimeError):
        return report(name, False, f"grad_data_ratio 应返回一个数（或 0 维张量），实际是 {type(v).__name__}")
    if not math.isclose(v, want_gd, rel_tol=1e-4):
        return report(name, False, f"grad_data_ratio 应为 {want_gd:.4g}，实际 {v:.4g}",
                      "梯度的标准差除以参数本身的标准差：p.grad.std() / p.std()（都用标准差，不用均值）")

    lr = 0.1
    want_ud = math.log10((lr * p.grad).std().item() / p.detach().std().item())
    ok, v = safe_call(name, update_data_ratio, p, lr)
    if not ok:
        return False
    try:
        v = _num(v)
    except (TypeError, ValueError, RuntimeError):
        return report(name, False, f"update_data_ratio 应返回一个数（或 0 维张量），实际是 {type(v).__name__}")
    if not math.isclose(v, want_ud, abs_tol=1e-4):
        hint = "Karpathy 的写法：((lr * p.grad).std() / p.std()).log10()"
        if math.isclose(v, want_ud - math.log10(lr), abs_tol=1e-4):
            hint = "忘了乘学习率：分子是这一步的更新量 lr × 梯度，不是梯度本身"
        elif math.isclose(v, 10 ** want_ud, rel_tol=1e-4):
            hint = "忘了取 log10：诊断图的纵轴是 log10(update:data)，参照线在 −3"
        elif math.isclose(v, want_ud * math.log(10), abs_tol=1e-4):
            hint = "要用以 10 为底的对数：.log10()，不是自然对数 .log()"
        return report(name, False, f"update_data_ratio(p, lr=0.1) 应为 {want_ud:.4f}，实际 {v:.4f}", hint)
    return report(name, True)


TANH_CONFIGS = ("gain 5/3", "gain 3", "BN + gain 3")


def tanh_diagnostics(diag):
    """diag[名字] = {"sat_init": [5 个 tanh 层的饱和比例], "sat_end": [...], "ud": (步数, 权重矩阵数) 的 log10(update:data)}，
    名字为 "gain 5/3"、"gain 3"、"BN + gain 3"。

    本机实测（3 个种子 × CPU / MPS）：第 2–5 层初始饱和平均 6.8–7.3% / 39.7–41.0% / 2.9–3.0%；BN 版各层
    （初始与结束）≤ 3.7%；隐层（W2–W5）最后 100 步的 log10(update:data) −2.66 ~ −2.75 / −2.72 ~ −2.77 /
    −3.00 ~ −3.04。区间都留了 2 倍以上的余量。"""
    name = "体检 ④ 的诊断数字"
    if not isinstance(diag, dict) or not all(k in diag for k in TANH_CONFIGS):
        return report(name, False, f"diag 应是含 {list(TANH_CONFIGS)} 三个键的 dict")
    sat, ud = {}, {}
    for k in TANH_CONFIGS:
        d = diag[k]
        try:
            sat[k] = (np.asarray(d["sat_init"], dtype=float), np.asarray(d["sat_end"], dtype=float))
            ud[k] = np.asarray(d["ud"], dtype=float)
        except (KeyError, TypeError, ValueError):
            return report(name, False, f"diag[{k!r}] 应含 sat_init、sat_end、ud 三项")
        if any(len(v) != 5 for v in sat[k]):
            return report(name, False, f"diag[{k!r}] 的 sat_init / sat_end 应各有 5 个数（5 个 tanh 层）")
        if any(np.any((v < 0) | (v > 1)) for v in sat[k]):
            return report(name, False, f"diag[{k!r}] 的饱和比例应是 0–1 之间的小数，不是百分数")
        if ud[k].ndim != 2 or ud[k].shape[0] < 100 or ud[k].shape[1] < 3 or not np.all(np.isfinite(ud[k])):
            return report(name, False, f"diag[{k!r}]['ud'] 应是 (步数 ≥ 100, 权重矩阵数) 的有限数数组，"
                          f"实际形状 {ud[k].shape}")
    good, bad = sat["gain 5/3"][0][1:].mean(), sat["gain 3"][0][1:].mean()
    bn = max(sat["BN + gain 3"][0].max(), sat["BN + gain 3"][1].max())
    if not 0.02 <= good <= 0.15:
        return report(name, False, f"gain 5/3 时第 2–5 层的初始饱和比例应在 2%–15%（本机约 7%），实际 {good:.1%}",
                      "检查初始化：权重标准差 = gain / sqrt(fan_in)；饱和的判据是 |t| > 0.97")
    if not bad > 0.25:
        return report(name, False, f"gain 3 时第 2–5 层应大面积饱和（> 25%，本机约 40%），实际 {bad:.1%}",
                      "gain 3 的网络应该真的用了 3 倍于 1/sqrt(fan_in) 的标准差")
    if not bn < 0.08:
        return report(name, False, f"加了 BatchNorm 后各层的饱和比例应 < 8%（本机约 3%），实际最大 {bn:.1%}",
                      "BatchNorm1d 要放在 Linear 和 Tanh 之间")
    hid = {k: v[-100:, 1:-1].mean() for k, v in ud.items()}      # 只看隐层权重（去掉第一个与最后一个）
    if not -3.3 <= hid["gain 5/3"] <= -2.2:
        return report(name, False, f"gain 5/3 时隐层 log10(update:data) 最后 100 步的均值应在 [−3.3, −2.2]"
                      f"（本机约 −2.7），实际 {hid['gain 5/3']:.2f}",
                      "分子是 lr × 梯度的标准差，分母是参数本身的标准差，最后取 log10")
    if not hid["BN + gain 3"] < hid["gain 5/3"] - 0.1:
        return report(name, False, f"BN + gain 3 的隐层 update:data 应明显低于 gain 5/3（本机约 −3.0 vs −2.7），"
                      f"实际 {hid['BN + gain 3']:.2f} vs {hid['gain 5/3']:.2f}",
                      "BatchNorm 让前向不再受权重尺度影响；但权重大了，同样的学习率带来的相对更新就小了")
    return report(name, True)


# ---------- 讲义 S4：网格搜索 vs 随机搜索 ----------
def log_uniform(fn):
    """学习者写的 log_uniform(rng, low, high, n)：在 [low, high] 上按对数尺度均匀采样 n 个数。"""
    name = "对数均匀采样 log_uniform"
    ok, x = safe_call(name, fn, np.random.default_rng(0), 1e-4, 1e-1, 20000)
    if not ok:
        return False
    try:
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return report(name, False, f"应返回 n 个数的数组，实际是 {type(x).__name__}")
    if x.shape != (20000,):
        return report(name, False, f"n = 20000 时应返回 20000 个数，实际形状 {x.shape}", "rng.uniform(..., size=n)")
    if not np.all((x >= 1e-4) & (x <= 1e-1)):
        return report(name, False, f"样本应都落在 [1e-4, 1e-1] 内，实际范围 [{x.min():.3g}, {x.max():.3g}]",
                      "在 log10 尺度上取端点：rng.uniform(np.log10(low), np.log10(high))，再取 10 的幂")
    med = float(np.median(x))
    if med > 0.02:
        return report(name, False, f"中位数 {med:.3g} 太大：这是线性均匀采样（中位数约 0.05），九成样本挤在 1e-2 到 1e-1 之间",
                      "先在 log10 尺度上均匀采样，再取 10 的幂：10 ** rng.uniform(np.log10(low), np.log10(high), size=n)")
    m = float(np.log10(x).mean())
    return report(name, abs(m + 2.5) < 0.05,
                  f"log10 之后的均值应约为 −2.5（区间 [−4, −1] 的中点），实际 {m:.3f}",
                  "log10 之后应在 [log10(low), log10(high)] 上均匀分布")


def best_so_far(fn):
    """学习者写的 best_so_far(vals)：第 k 个元素是前 k 次试验的最大值。"""
    name = "迄今最好曲线 best_so_far"
    cases = [([0.3, 0.5, 0.4, 0.7, 0.6], [0.3, 0.5, 0.5, 0.7, 0.7]),
             ([0.8, 0.1, 0.2], [0.8, 0.8, 0.8])]
    for vals, want in cases:
        ok, out = safe_call(name, fn, list(vals))
        if not ok:
            return False
        try:
            out = np.asarray(out, dtype=float)
        except (TypeError, ValueError):
            return report(name, False, f"应返回与输入等长的数组，实际是 {type(out).__name__}")
        if out.shape != (len(vals),) or not np.allclose(out, want):
            return report(name, False, f"对 {vals} 应得 {want}，实际 {np.round(out, 4).tolist()}",
                          "累积最大值：np.maximum.accumulate(vals)")
    return report(name, True)


def search_results(grid_vals, random_bests):
    """grid_vals：4 × 4 网格 16 次试验的验证准确率；random_bests：每次随机搜索（16 次试验）的最好验证准确率。

    本机实测：网格最好 0.7975（CPU 与 MPS 相同）；5 次随机搜索的最好值 0.7935–0.8045。
    **不检查"随机胜过网格"**：二维时差距小于种子波动，20 次重复里有 15% 会输，跨平台还可能翻转。"""
    name = "网格 vs 随机搜索的结果"
    try:
        g = np.asarray(grid_vals, dtype=float).ravel()
        r = np.asarray(random_bests, dtype=float).ravel()
    except (TypeError, ValueError):
        return report(name, False, "应传入两个数列：grid_vals（16 个）与 random_bests（每次随机搜索的最好值）")
    if g.size != 16:
        return report(name, False, f"grid_vals 应有 16 个（4 × 4 网格），实际 {g.size} 个")
    if r.size < 3:
        return report(name, False, f"随机搜索至少重复 3 次（换采样种子），实际 {r.size} 次",
                      "只跑一次随机搜索看不出它自己的波动有多大")
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(r))):
        return report(name, False, "结果里有非有限数（nan / inf）")
    if not 0.77 <= g.max() <= 0.83:
        return report(name, False, f"网格的最好验证准确率应在 [0.77, 0.83]（本机 0.7975），实际 {g.max():.4f}",
                      "trial 里要用验证集 X_val；3 个 epoch、Adam；weight_decay 传进 train_epochs 了吗？")
    return report(name, bool(np.all((0.75 <= r) & (r <= 0.84))),
                  f"每次随机搜索的最好验证准确率应在 [0.75, 0.84]（本机 0.79–0.81），实际 {np.round(r, 4).tolist()}",
                  "学习率与权重衰减都要按对数尺度在给定范围里采样")


# ---------- 练习 E5：看诊断图下诊断 II ----------
# 三次"神秘训练"的设置放在这里，练习里先看图、下诊断，再揭晓（参考答案只在开发仓库）。
# A 比参照深一倍（10 层，题面已写明），另一处不同是初始化（PyTorch 默认）；B、C 各只有一处（学习率）不同。
_MYSTERY = {
    "A": {"depth": 10, "gain": None, "batchnorm": False, "lr": 0.1},
    "B": {"depth": 5, "gain": 5 / 3, "batchnorm": False, "lr": 1.0},
    "C": {"depth": 5, "gain": 5 / 3, "batchnorm": False, "lr": 0.003},
}


def mystery_settings(which):
    """练习 E5 的神秘训练设置：返回 {"depth", "gain", "batchnorm", "lr"} 的副本（gain=None 表示 PyTorch 默认初始化）。"""
    key = str(which).strip().upper()
    if key not in _MYSTERY:
        raise ValueError(f"只有神秘训练 A、B、C，实际传入 {which!r}")
    return dict(_MYSTERY[key])


def mystery_fixed(grad_ratio, val_acc):
    """修好神秘训练 A 之后：初始化时第 1 层与最后一层激活梯度的标准差之比，以及 1000 步后的验证准确率。

    本机实测（10 层、1000 步，种子 0/1/2）：修好前比值约 0.006、验证 0.58–0.66；gain 5/3 比值约 2.1、验证 0.78–0.80；
    BatchNorm 比值 2.8–3.2、验证 0.77–0.82；gain 1 / gain 2 比值 0.42 / 3.7、验证约 0.78；gain 3 比值 19、验证 0.69。"""
    name = "修好神秘训练 A"
    try:
        r, a = float(grad_ratio), float(val_acc)
    except (TypeError, ValueError):
        return report(name, False, "应传入两个数：grad_ratio（第 1 层 / 最后一层）、val_acc")
    if not (np.isfinite(r) and np.isfinite(a)):
        return report(name, False, f"数值不是有限数（{r}, {a}）")
    if not 0.25 <= r <= 5:
        return report(name, False, f"第 1 层与最后一层激活梯度的标准差之比应在 [0.25, 5]（各层'长得差不多'），实际 {r:.3g}",
                      "比值远小于 1：梯度往回传时逐层萎缩——回想体检 ④ 里哪种初始化让各层的梯度分布相近；"
                      "比值远大于 1：梯度被放大，初始化可能又太大了")
    return report(name, a > 0.74, f"各层梯度已经平衡，但验证准确率应 > 0.74，实际 {a:.4f}",
                  "只改那一处设置（层数保持 10 层）；学习率保持 0.1、训练 1000 步")
