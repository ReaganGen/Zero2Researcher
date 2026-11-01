"""第17周自查函数。用法：from zero2researcher.checks import week17 as ck; ck.alpha_bar(alpha_bar)

函数练习把函数本身传进来（不加括号），例如 ck.cfg_combine(cfg_combine)；其余检查直接传入你算出的结果。

本周的约定：
- DDPM（Ho, Jain & Abbeel 2020, arXiv:2006.11239）：x = √ᾱ·x_data + √(1−ᾱ)·ε，**时间 0 是数据**。
  原文离散调度 β 从 1e-4 线性增到 0.02、T = 1000；课程代码用它的连续极限
  ᾱ(τ) = exp(−(0.1·τ + 9.95·τ²))，τ ∈ [0, 1]（τ = i/T），与离散的 ∏(1 − β_i) 处处相差 < 1e-3。
- Flow Matching（第 16 周）：x_t = t·x_data + (1−t)·ε，**t = 0 是噪声**。两套约定的时间方向相反。
- Classifier-free guidance 用 Ho & Salimans 2022（arXiv:2207.12598）原文约定：
  guided = (1 + w)·cond − w·uncond，**w = 0 就是普通的条件模型（不加引导）**。
  MIT 6.S184 讲义用 (1 − w)·uncond + w·cond，两者的关系是 w_MIT = 1 + w_Ho。
训练结果按合理区间检查，区间依据本机实测 ± 余量。torch 只在函数内部 import。
"""
import math

import numpy as np

from . import report, safe_call

STEPS = (2, 5, 20, 100)
METHODS = ("FM", "DDPM")
WS = (0, 1, 3)
N_SEEDS = 3


def _ref_alpha_bar(tau):
    return (-(0.1 * tau + 9.95 * tau ** 2)).exp()


# ---------- ᾱ 调度 ----------
def alpha_bar(fn):
    """fn(tau) -> ᾱ(τ) = exp(−(0.1·τ + 9.95·τ²))，逐元素作用于张量 tau（τ = 0 数据，τ = 1 几乎纯噪声）。

    同时与原文的离散调度 ∏_{s≤i}(1 − β_s)（β 从 1e-4 线性到 0.02，T = 1000）在 τ = i/T 处比较。"""
    import torch

    name = "ᾱ 调度 alpha_bar"
    tau = torch.linspace(0, 1, 101)
    ok, ab = safe_call(name, fn, tau)
    if not ok:
        return False
    if not isinstance(ab, torch.Tensor) or tuple(ab.shape) != (101,):
        return report(name, False, f"输入 101 个时间，应返回同形状 (101,) 的张量，实际 {getattr(ab, 'shape', ab)!r}",
                      "用 torch 运算逐元素计算，不要用 math.exp")
    ab = ab.double()
    if not torch.isfinite(ab).all():
        return report(name, False, "结果里有 NaN/inf")
    ref = _ref_alpha_bar(tau.double())
    if torch.allclose(ab, 1 - ref, atol=1e-4):       # 先于 ᾱ(0) 的检查：1 − ᾱ 在 τ = 0 处也是 0
        return report(name, False, f"τ = 0.5 时应为 {float(ref[50]):.5f}，实际 {float(ab[50]):.5f}",
                      "你返回的是 1 − ᾱ（噪声方差），不是 ᾱ")
    if abs(float(ab[0]) - 1.0) > 1e-6:
        hint = "τ = 0 是数据，此时 ᾱ = exp(0) = 1"
        if float(ab[0]) < 1e-3:
            hint = "时间方向反了：DDPM 约定 τ = 0 是数据（ᾱ = 1）、τ = 1 是噪声（ᾱ ≈ 0）"
        return report(name, False, f"ᾱ(0) 应为 1，实际 {float(ab[0]):.4g}", hint)
    if not bool((ab[1:] < ab[:-1]).all()):
        return report(name, False, "ᾱ 应随 τ 单调递减（越往后噪声越多）")
    if not torch.allclose(ab, ref, rtol=1e-3, atol=1e-7):
        hint = "ᾱ(τ) = exp(−∫₀^τ β(s) ds)，β(s) = 0.1 + 19.9·s，积分得 0.1·τ + 9.95·τ²"
        if torch.allclose(ab, (-(0.1 * tau + 19.9 * tau ** 2)).exp().double(), rtol=1e-3):
            hint = "τ² 前的系数应是 19.9/2 = 9.95（对 β(s) = 0.1 + 19.9·s 积分时别忘了 1/2）"
        return report(name, False, f"τ = 0.5 时应为 {float(ref[50]):.5f}，实际 {float(ab[50]):.5f}", hint)
    beta = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)
    discrete = torch.cumprod(1 - beta, 0)
    ok, ab_i = safe_call(name, fn, torch.arange(1, 1001, dtype=torch.float32) / 1000)
    if not ok:
        return False
    gap = float((ab_i.double() - discrete).abs().max())
    return report(name, gap < 2e-3, f"与原文离散调度 ∏(1 − β_i) 的最大差 {gap:.4f}，应 < 0.002")


def half_time(data_coef, t_matched):
    """E1b：τ = 0.5 时的数据系数 √ᾱ(0.5)，以及与它噪声水平相同的 FM 时间 t = generative.ddpm_to_fm_time(ᾱ(0.5))。

    参考值在这里由公式现算，不写进 notebook——题目要求先估计、再用代码算。"""
    name = "E1b：τ = 0.5 的数据系数与噪声水平相同的 FM 时间"
    try:
        c, t = float(data_coef), float(t_matched)
    except (TypeError, ValueError):
        return report(name, False, "应传入两个数：data_coef 与 t_matched", "用 float(...) 把张量转成数")
    if not (math.isfinite(c) and math.isfinite(t)):
        return report(name, False, f"两个数都应是有限数，实际 {c}、{t}")
    ab = math.exp(-(0.1 * 0.5 + 9.95 * 0.5 ** 2))
    want_c = math.sqrt(ab)

    def fm_time(a):                                  # = generative.ddpm_to_fm_time(a)
        return math.sqrt(a) / (math.sqrt(a) + math.sqrt(1 - a))

    want_t = fm_time(ab)
    if abs(c - want_c) > 1e-3:
        hint = "数据系数是 √ᾱ(0.5)：先算 alpha_bar(0.5)，再开根号"
        if abs(c - ab) < 1e-3:
            hint = "你给的是 ᾱ(0.5) 本身；数据系数是 √ᾱ，别忘了开根号"
        elif abs(c - math.sqrt(1 - ab)) < 1e-3:
            hint = "你给的是噪声系数 √(1 − ᾱ)；数据系数是 √ᾱ"
        return report(name, False, f"data_coef = {c:.4f} 不对", hint)
    if abs(t - want_t) > 1e-3:
        hint = "t_matched = generative.ddpm_to_fm_time(alpha_bar(0.5))——传进去的是 ᾱ"
        if abs(t - 0.5) < 1e-3:
            hint = ("t = 0.5 正是把 τ = 0.5 当成 ᾱ 传进去的结果（也恰好等于'直接翻转' 1 − τ）；"
                    "ddpm_to_fm_time 要的是 ᾱ(0.5)，不是 τ")
        elif abs(t - fm_time(want_c)) < 1e-3:
            hint = "你把 √ᾱ（数据系数）传了进去；ddpm_to_fm_time 要的是 ᾱ 本身"
        return report(name, False, f"t_matched = {t:.4f} 不对", hint)
    return report(name, True)


# ---------- 前向加噪 ----------
def ddpm_forward(fn):
    """fn(x0, alpha_bar) -> (x_t, eps)：内部采样 eps ~ N(0, I)，返回 x_t = √ᾱ·x0 + √(1−ᾱ)·eps 与所用的 eps。

    alpha_bar 是能与 x0 广播的张量（例如 (B, 1)）。检查公式、统计量（均值 √ᾱ·x0、方差 1−ᾱ）与广播。"""
    import torch

    name = "DDPM 前向加噪 ddpm_forward"
    torch.manual_seed(0)
    B = 40000
    x0 = torch.full((B, 2), 2.0)
    ab = torch.full((B, 1), 0.25)
    ok, out = safe_call(name, fn, x0, ab)
    if not ok:
        return False
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        return report(name, False, "应返回两个张量 (x_t, eps)", "return x_t, eps")
    xt, eps = out
    if tuple(getattr(xt, "shape", ())) != (B, 2) or tuple(getattr(eps, "shape", ())) != (B, 2):
        return report(name, False, f"x_t 与 eps 都应与 x0 同形状 ({B}, 2)")
    if not torch.allclose(xt, 0.5 * x0 + math.sqrt(0.75) * eps, atol=1e-5):
        hint = "x_t = √ᾱ·x0 + √(1−ᾱ)·eps"
        if torch.allclose(xt, 0.25 * x0 + 0.75 * eps, atol=1e-5):
            hint = "系数要开根号：√ᾱ 与 √(1−ᾱ)（保证 x_t 的方差在数据方差为 1 时保持为 1）"
        elif torch.allclose(xt, math.sqrt(0.75) * x0 + 0.5 * eps, atol=1e-5):
            hint = "数据与噪声的系数对调了：数据乘 √ᾱ，噪声乘 √(1−ᾱ)"
        return report(name, False, "返回的 x_t 与 eps 不满足 x_t = √ᾱ·x0 + √(1−ᾱ)·eps", hint)
    m, s = float(eps.mean()), float(eps.std())
    if abs(m) > 0.02 or abs(s - 1) > 0.02:
        return report(name, False, f"eps 应是标准正态噪声（均值 0、标准差 1），实际均值 {m:.3f}、标准差 {s:.3f}",
                      "eps = torch.randn_like(x0)")
    mt, st = float(xt.mean()), float(xt.std())
    if abs(mt - 1.0) > 0.02 or abs(st - math.sqrt(0.75)) > 0.02:
        return report(name, False, f"ᾱ = 0.25、x0 = 2 时 x_t 的均值应 ≈ 1.0、标准差 ≈ {math.sqrt(0.75):.3f}，"
                      f"实际 {mt:.3f}、{st:.3f}")
    abs_ = torch.tensor([[1.0], [0.5], [1e-4]])
    ok, out = safe_call(name, fn, torch.ones(3, 2), abs_)
    if not ok:
        return False
    xt, eps = out
    if not torch.allclose(xt[0], torch.ones(2), atol=1e-6):
        return report(name, False, "ᾱ = 1 时 x_t 应恰好等于 x0（还没加噪）", "每一行用自己的 ᾱ：形状 (B, 1) 会广播")
    return report(name, True)


# ---------- 参数化换算 ----------
def param_conversion(eps_to_data_ddpm, velocity_to_eps_fm, velocity_to_data_fm):
    """三个手写的换算函数，与工具包 generative.convert_prediction 对账：

    - eps_to_data_ddpm(eps, x_t, alpha_bar) -> data = (x_t − √(1−ᾱ)·eps)/√ᾱ   （DDPM 约定）
    - velocity_to_eps_fm(u, x_t, t) -> eps = x_t − t·u                        （FM 约定，u = data − eps）
    - velocity_to_data_fm(u, x_t, t) -> data = x_t + (1 − t)·u                （FM 约定）"""
    import torch
    from zero2researcher import generative

    name = "参数化换算（与工具包对账）"
    torch.manual_seed(0)
    data, eps = torch.randn(500, 2) * 2 + 1, torch.randn(500, 2)
    t = torch.rand(500, 1) * 0.98 + 0.01
    ab = generative.fm_to_ddpm_time(t)
    x_fm = t * data + (1 - t) * eps
    x_vp = ab.sqrt() * data + (1 - ab).sqrt() * eps
    u = data - eps
    cases = [
        ("eps_to_data_ddpm", eps_to_data_ddpm, (eps, x_vp, ab),
         generative.convert_prediction(eps, x_vp, "eps", "data", alpha_bar=ab), data,
         "data = (x_t − √(1−ᾱ)·eps) / √ᾱ"),
        ("velocity_to_eps_fm", velocity_to_eps_fm, (u, x_fm, t),
         generative.convert_prediction(u, x_fm, "velocity", "eps", t_fm=t), eps,
         "x_t = t·data + (1−t)·eps 且 u = data − eps，消去 data：eps = x_t − t·u"),
        ("velocity_to_data_fm", velocity_to_data_fm, (u, x_fm, t),
         generative.convert_prediction(u, x_fm, "velocity", "data", t_fm=t), data,
         "消去 eps：data = x_t + (1 − t)·u"),
    ]
    for fname, f, args, toolkit, truth, hint in cases:
        ok, got = safe_call(f"{name}：{fname}", f, *args)
        if not ok:
            return False
        if not isinstance(got, torch.Tensor) or got.shape != truth.shape:
            return report(name, False, f"{fname} 应返回形状 {tuple(truth.shape)} 的张量，实际 {getattr(got, 'shape', got)!r}")
        if not torch.allclose(toolkit, truth, atol=1e-3):   # 工具包本身的自检
            return report(name, False, "工具包换算与真值不符——请把这条信息报到课程仓库的 Issues")
        err = float((got - toolkit).abs().max())
        if err > 1e-3:
            return report(name, False, f"{fname} 与工具包结果的最大差 {err:.4f}", hint)
    return report(name, True)


# ---------- classifier-free guidance ----------
def cfg_combine(fn):
    """fn(pred_cond, pred_uncond, w) -> (1 + w)·pred_cond − w·pred_uncond（Ho & Salimans 约定）。"""
    import torch

    name = "CFG 组合公式 cfg_combine"
    torch.manual_seed(0)
    c, u = torch.randn(8, 2), torch.randn(8, 2)
    # 先查 w = 1：w = 0 时"MIT 约定"与"cond / uncond 对调"给出同一个结果（都是 uncond），分不出是哪种错误
    for w in (1.0, 3.0, 0.5, 0.0):
        ok, g = safe_call(name, fn, c, u, w)
        if not ok:
            return False
        if not isinstance(g, torch.Tensor) or g.shape != c.shape:
            return report(name, False, f"应返回与预测同形状 {tuple(c.shape)} 的张量")
        want = (1 + w) * c - w * u
        if not torch.allclose(g, want, atol=1e-6):
            hint = "guided = (1 + w)·cond − w·uncond"
            if torch.allclose(g, (1 - w) * u + w * c, atol=1e-6):
                hint = ("你用的是 MIT 讲义的约定 (1 − w)·uncond + w·cond（w = 1 才是不加引导）。"
                        "本课程统一用 Ho & Salimans 原文约定：w = 0 不加引导，w_MIT = 1 + w_Ho")
            elif torch.allclose(g, (1 + w) * u - w * c, atol=1e-6):
                hint = "cond 与 uncond 对调了：放大的是条件预测，减去的是无条件预测"
            msg = f"w = {w:g} 时结果不对" + ("（w = 0 应原样返回条件预测）" if w == 0 else "")
            return report(name, False, msg, hint)
    return report(name, True)


# ---------- 研究实验 4：结构检查 ----------
def fm_vs_ddpm_results(results):
    """期望结构：results[数据集名][方法][采样步数] = {"ed": [3 个种子], "precision": [3 个种子]}，
    方法恰为 {"FM", "DDPM"}，步数恰为 {2, 5, 20, 100}，至少一个数据集。
    另检查一条健全性：每种方法 100 步的能量距离都小于 2 步。"""
    name = "FM vs DDPM 受控对比的结果结构"
    if not isinstance(results, dict) or not results:
        return report(name, False, f"results 应是非空 dict（键 = 数据集名），实际 {type(results).__name__}")
    for ds, by_method in results.items():
        if not isinstance(by_method, dict) or set(by_method) != set(METHODS):
            got = sorted(by_method) if isinstance(by_method, dict) else by_method
            return report(name, False, f"results[{ds!r}] 的键应恰为 {METHODS}，实际 {got!r}")
        for method, by_steps in by_method.items():
            where = f"results[{ds!r}][{method!r}]"
            if not isinstance(by_steps, dict):
                return report(name, False, f"{where} 应是 {{步数: {{'ed': [...], 'precision': [...]}}}}")
            try:
                got = sorted(int(k) for k in by_steps)
            except (TypeError, ValueError):
                return report(name, False, f"{where} 的键应是采样步数（整数），实际 {list(by_steps)!r}")
            if got != sorted(STEPS):
                return report(name, False, f"{where} 应恰好包含步数 {STEPS}，实际 {got}")
            means = {}
            for k, v in by_steps.items():
                if not isinstance(v, dict) or not {"ed", "precision"} <= set(v):
                    return report(name, False, f"{where}[{k}] 应是含 'ed' 与 'precision' 两个列表的 dict")
                for key in ("ed", "precision"):
                    vals = [float(x) for x in v[key]]
                    if len(vals) != N_SEEDS:
                        return report(name, False, f"{where}[{k}]['{key}'] 应有 {N_SEEDS} 个种子的结果，实际 {len(vals)} 个",
                                      "FM 与 DDPM 用同样的 3 个种子（同一种子 → 同样的初始化与数据流）")
                    if not all(np.isfinite(vals)):
                        return report(name, False, f"{where}[{k}]['{key}'] 里有非有限数：{vals}",
                                      "DDPM 祖先采样在 2 步时会把误差放大几十倍，但不应出现 NaN——检查 ᾱ 是否写对、是否除以了 0")
                if not all(0.0 <= float(p) <= 1.0 for p in v["precision"]):
                    return report(name, False, f"{where}[{k}]['precision'] 应在 [0, 1] 内")
                means[int(k)] = float(np.mean([float(x) for x in v["ed"]]))
            if not means[100] < means[2]:
                return report(name, False, f"{ds} / {method}：100 步的能量距离（{means[100]:.4f}）应小于 2 步（{means[2]:.4f}）",
                              "检查采样器：FM 从 t=0 积分到 1；DDPM 从 τ=1 走到 0，模型收到的是 DDPM 时间 τ")
    return report(name, True)


def guidance_tradeoff(results):
    """期望结构：results[w] = {"frac_target": 判为目标数字的比例, "class_entropy": 预测类别直方图的归一化熵,
    "diversity": 同类样本两两像素距离的均值}，w 恰为 {0, 1, 3}（Ho & Salimans 约定）。

    检查：数值合法；"判为目标的比例"随 w 不减（允许 0.03 的波动）；w = 0 时已明显高于随机猜的 0.1。"""
    name = "guidance 强度的权衡"
    if not isinstance(results, dict):
        return report(name, False, f"results 应是 dict（键 = w），实际 {type(results).__name__}")
    try:
        keys = sorted(float(k) for k in results)
    except (TypeError, ValueError):
        return report(name, False, f"键应是 guidance 强度 w 的数值，实际 {list(results)!r}")
    if keys != [float(w) for w in WS]:
        return report(name, False, f"应恰好包含 w ∈ {WS}（Ho 约定），实际 {keys}",
                      "如果你写的是 1、2、4，那是 MIT 约定（w_MIT = 1 + w_Ho）")
    rows = {float(k): v for k, v in results.items()}
    for w in sorted(rows):
        v = rows[w]
        if not isinstance(v, dict) or not {"frac_target", "class_entropy", "diversity"} <= set(v):
            return report(name, False, f"results[{w:g}] 应是含 frac_target / class_entropy / diversity 的 dict")
        try:
            f, e, d = float(v["frac_target"]), float(v["class_entropy"]), float(v["diversity"])
        except (TypeError, ValueError):
            return report(name, False, f"results[{w:g}] 里的三个值都应是数")
        if not (0 <= f <= 1 and 0 <= e <= 1 + 1e-9 and np.isfinite(d) and d > 0):
            return report(name, False, f"results[{w:g}] 数值不合法：frac_target {f}、class_entropy {e}、diversity {d}",
                          "frac_target 与 class_entropy 都在 [0, 1]；diversity 是正的像素距离")
    fr = [float(rows[float(w)]["frac_target"]) for w in WS]
    if fr[0] < 0.3:
        return report(name, False, f"w = 0（纯条件模型）时判为目标的比例只有 {fr[0]:.3f}，几乎和随机猜（0.1）一样",
                      "检查训练时是否真的把类别 y 传进了模型、null 类别是否只占 p_uncond 的比例、"
                      "采样时传的是目标类别而不是 null")
    ok = all(b >= a - 0.03 for a, b in zip(fr, fr[1:]))
    return report(name, ok, f"判为目标的比例应随 w 增大而不减，实际 w=0/1/3：{fr[0]:.3f} / {fr[1]:.3f} / {fr[2]:.3f}",
                  "检查组合公式 (1 + w)·cond − w·uncond，以及 uncond 是否用 null 类别（= num_classes）算出")


def noise_control(frac, entropy):
    """阴性对照：把**纯高斯噪声**（采样器的起点）直接交给分类器。

    frac：纯噪声中被判成"最常见那一类"的比例（= max(pred_counts) / 样本数）；
    entropy：纯噪声预测类别直方图的归一化熵（classifier_scores 的 class_entropy）。
    本机实测（14×14 分类器）：多数噪声被判成同一个数字（多为 8），frac ≈ 0.55–0.95、熵 ≈ 0.1–0.45，
    而真实数字的熵 ≈ 0.99。通过本检查说明你确实在纯噪声上做了对照——
    "分类器很有把握"不能证明样本是好数字。"""
    name = "纯噪声阴性对照"
    try:
        f, e = float(frac), float(entropy)
    except (TypeError, ValueError):
        return report(name, False, "应传入两个数：纯噪声被判成最常见类别的比例、预测类别直方图的归一化熵")
    if not (0 <= f <= 1 and 0 <= e <= 1 + 1e-9):
        return report(name, False, f"两个数都应在 [0, 1] 内，实际 {f}、{e}")
    if e > 0.8 or f < 0.25:
        return report(name, False, f"类别熵 {e:.3f}、最常见类别占 {f:.3f}——这更像真实数字或生成样本，不像纯噪声",
                      "阴性对照的输入是 torch.randn(...) 的纯噪声（与采样器的起点相同），不要传生成样本；"
                      "frac 用 max(pred_counts) / 样本数")
    return report(name, True)
