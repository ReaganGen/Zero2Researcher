"""第15周自查函数。用法：from zero2researcher.checks import week15 as ck; ck.reparameterize(reparameterize)

函数练习把函数本身传进来（不加括号），例如 ck.vae_kl(vae_kl)；其余检查直接传入你算出的数。
口径照 pytorch/examples/vae（BSD-3-Clause，https://github.com/pytorch/examples/tree/main/vae ）：
编码器输出 mu 与 logvar = log σ²；重参数化 z = mu + exp(0.5·logvar)·eps；
KL 项 = −½·Σ(1 + logvar − mu² − exp(logvar))（Kingma & Welling 2013 附录 B），对 batch 与潜维度全部求和。
数值区间依据本机实测（见各函数说明）± 余量。torch 只在函数内部 import。
"""
import math

import numpy as np

from . import report, safe_call

BETAS = (0.1, 1, 4)
N_SEEDS = 3


def _in_range(name, value, lo, hi, hint=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(value).__name__}")
    if not np.isfinite(v):
        return report(name, False, f"数值不是有限数（{v}），训练大概率发散了", hint)
    return report(name, lo <= v <= hi, f"期望落在 [{lo}, {hi}]，实际 {v:.4f}", hint)


# ---------- 重参数化、KL、插值 ----------
def reparameterize(fn):
    """fn(mu, logvar) -> z。检查形状、统计（均值≈mu、标准差≈exp(0.5·logvar)）、可求导、每次调用重新采样。"""
    import torch

    name = "重参数化 reparameterize"
    torch.manual_seed(0)
    n = 20000
    mu0 = torch.tensor([0.0, 2.0, -1.0])
    std0 = torch.tensor([1.0, 0.5, 2.0])
    mu = mu0.repeat(n, 1).requires_grad_()
    logvar = (2 * std0.log()).repeat(n, 1).requires_grad_()
    ok, z = safe_call(name, fn, mu, logvar)
    if not ok:
        return False
    if not isinstance(z, torch.Tensor) or tuple(z.shape) != (n, 3):
        return report(name, False, f"输入 mu、logvar 的形状是 ({n}, 3)，z 应同形状；实际 {getattr(z, 'shape', z)!r}")
    m, s = z.detach().mean(0), z.detach().std(0)
    if not torch.allclose(m, mu0, atol=0.06):
        return report(name, False, f"z 的样本均值应≈mu {mu0.tolist()}，实际 {m.numpy().round(3).tolist()}",
                      "z 应以 mu 为中心：在 mu 上加“标准差 × 标准正态噪声”，噪声的均值是 0")
    if not torch.allclose(s, std0, rtol=0.05):
        hint = "logvar 是 log σ²，所以 σ = exp(0.5·logvar)"
        if torch.allclose(s, std0 ** 2, rtol=0.05):
            hint = "你把 exp(logvar)（方差）当成了标准差：σ = exp(0.5·logvar)"
        return report(name, False, f"z 的样本标准差应≈{std0.tolist()}，实际 {s.numpy().round(3).tolist()}", hint)
    if not z.requires_grad:
        return report(name, False, "z 不能对 mu、logvar 求导",
                      "Normal(...).sample() 这类采样不可导。重参数化的要点：随机性只来自一个与参数无关的噪声，"
                      "z 用普通的加法和乘法由 mu、σ 和这个噪声算出来")
    z.sum().backward()
    if mu.grad is None or not torch.allclose(mu.grad, torch.ones_like(mu)):
        return report(name, False, "∂Σz/∂mu 应处处等于 1（z = mu + σ·eps 对 mu 的导数是 1）")
    if logvar.grad is None or torch.count_nonzero(logvar.grad) == 0:
        return report(name, False, "z 对 logvar 没有梯度", "σ 要由 logvar 用 torch 运算算出来，并参与 z 的计算（别 detach、别转成 numpy）")
    small_mu, small_lv = torch.zeros(4, 5), torch.zeros(4, 5)
    ok1, z1 = safe_call(name, fn, small_mu, small_lv)
    ok2, z2 = safe_call(name, fn, small_mu, small_lv)
    if not (ok1 and ok2):
        return False
    if torch.equal(z1, z2):
        return report(name, False, "两次调用得到完全相同的 z——每次都应重新抽一份噪声 eps",
                      "噪声要在函数内部每次调用时重新抽，而不是在函数外面算好一次反复用")
    return report(name, True)


def _ref_vae_kl(mu, logvar):
    return -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum()


def vae_kl(fn):
    """fn(mu, logvar) -> KL(N(mu, σ²) ‖ N(0, I))，对 batch 与潜维度全部求和（pytorch/examples 口径）。"""
    import torch

    name = "VAE 的 KL 项"
    torch.manual_seed(0)
    cases = [(torch.zeros(3, 4), torch.zeros(3, 4)),
             (torch.randn(8, 20), torch.randn(8, 20)),
             (torch.full((2, 5), 1.5), torch.full((2, 5), -2.0))]
    for mu, logvar in cases:
        ok, got = safe_call(name, fn, mu, logvar)
        if not ok:
            return False
        try:
            g = float(got)
        except (TypeError, ValueError):
            return report(name, False, f"应返回一个标量（0 维张量或数），实际形状 {getattr(got, 'shape', None)}",
                          "对所有元素求和：(...).sum()")
        ref = float(_ref_vae_kl(mu, logvar))
        if not math.isclose(g, ref, rel_tol=1e-5, abs_tol=1e-5):
            hint = "−0.5 · Σ(1 + logvar − mu² − exp(logvar))"
            if ref != 0 and math.isclose(g, ref / mu.numel(), rel_tol=1e-5):
                hint = "你对所有元素取了平均；pytorch/examples 的口径是对 batch 和潜维度全部求和（与 BCE 的 reduction='sum' 一致）"
            elif ref != 0 and mu.dim() == 2 and math.isclose(g, ref / mu.shape[0], rel_tol=1e-5):
                hint = ("你对潜维度求和、再对 batch 取了平均——这也是一种常见口径（每张图的 KL），"
                        "但本课与 pytorch/examples 一致，对 batch 也求和（BCE 用的是 reduction='sum'，两项口径要相同）")
            elif math.isclose(g, -ref, rel_tol=1e-5, abs_tol=1e-5):
                hint = "符号反了：KL ≥ 0，前面是 −0.5"
            return report(name, False, f"mu 形状 {tuple(mu.shape)} 的一组输入：期望 {ref:.5f}，实际 {g:.5f}", hint)
    mu = torch.randn(4, 3, requires_grad=True)
    logvar = torch.randn(4, 3, requires_grad=True)
    ok, out = safe_call(name, fn, mu, logvar)
    if not ok:
        return False
    if not getattr(out, "requires_grad", False):
        return report(name, False, "KL 项要能对 mu、logvar 求导（用 torch 运算，不要转成 numpy 或 .item()）")
    return report(name, True)


def interpolate(fn):
    """fn(z1, z2, n) -> (n, d) 张量：第 0 行 = z1，最后一行 = z2，中间等距线性插值。"""
    import torch

    name = "潜空间插值 interpolate"
    torch.manual_seed(0)
    z1, z2 = torch.randn(5), torch.randn(5)
    ok, out = safe_call(name, fn, z1, z2, 7)
    if not ok:
        return False
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != (7, 5):
        return report(name, False, f"z1、z2 形状 (5,)、n=7 时应返回 (7, 5) 的张量，实际 {getattr(out, 'shape', out)!r}",
                      "先造 n 个从 0 到 1 等距的插值系数，把它变成一列，再借助广播一次算出 n 行")
    out = out.detach().float()
    if not torch.allclose(out[0], z1, atol=1e-6) or not torch.allclose(out[-1], z2, atol=1e-6):
        return report(name, False, "第一行应等于 z1、最后一行应等于 z2", "t 从 0 走到 1，两端都要包含")
    ref = torch.stack([(1 - t) * z1 + t * z2 for t in torch.linspace(0, 1, 7)])
    if not torch.allclose(out, ref, atol=1e-5):
        return report(name, False, "中间各行不是等距的线性插值", "相邻两行之差应处处等于 (z2 − z1)/(n − 1)")
    return report(name, True)


# ---------- 训练结果的区间 ----------
def vae_test_loss(loss):
    """MNIST VAE（pytorch/examples 结构，z=20）在 1 万张训练子集上训 5 个 epoch 后的测试损失（BCE + KL，每张图）。"""
    return _in_range("MNIST VAE 测试损失（每张图）", loss, 118, 150,
                     "本机实测 10k 子集 × 5 epochs ≈ 132–133（全量 60k × 10 epochs ≈ 105.5–106，与 NVIDIA Base Command Manager 10 手册里"
                     "一次 pytorch/examples 默认配置的运行日志 105.87 对齐；pytorch/examples 仓库本身没有公布这个数）。"
                     "偏高很多：检查 BCE 是否 reduction='sum' 并除以图片张数、输入是否在 [0, 1]、解码器最后是否 sigmoid；"
                     "只有几或几十：多半是对像素取了平均而不是求和")


def knn_latent_accuracy(acc):
    """FashionMNIST VAE 的 μ 作特征、kNN（k=5）在测试集上的分类准确率。"""
    return _in_range("潜空间 kNN 准确率（FashionMNIST）", acc, 0.60, 0.88,
                     "本机实测 z=20、1 万张训练子集、3 epochs：β 越大越低，约 0.75–0.80（PCA-20 + kNN 约 0.81）。"
                     "接近 0.10 说明训练集的 μ 与标签对不上，或把测试集也拿去 fit 了；"
                     "用的应是编码器输出的 μ（不是采样出来的 z）")


def linear_ae_vs_pca(ae_err, pca_err):
    """线性自编码器的重建误差应接近、但不明显低于同维 PCA（Baldi & Hornik 1989）。"""
    name = "线性自编码器 vs PCA"
    try:
        a, p = float(ae_err), float(pca_err)
    except (TypeError, ValueError):
        return report(name, False, "应传入两个数：线性 AE 的重建误差、PCA 的重建误差")
    if not (np.isfinite(a) and np.isfinite(p)) or p <= 0:
        return report(name, False, f"误差应是正的有限数，实际 AE {a}、PCA {p}")
    if a < 0.98 * p:
        return report(name, False, f"线性 AE 的误差 {a:.5f} 明显低于 PCA {p:.5f}——这不应发生",
                      "PCA 是同维线性重建的最优解（Eckart–Young）。检查两者是否用了同一份数据、同一个误差口径"
                      "（每像素平方误差的平均）、同一个潜维度，以及 AE 里是不是混进了非线性激活")
    return report(name, a <= 1.3 * p,
                  f"线性 AE 的误差 {a:.5f} 是 PCA {p:.5f} 的 {a / p:.2f} 倍，还没训到位（期望 ≤ 1.3 倍）",
                  "多训几个 epoch，或调大学习率；线性 AE 的误差面没有坏的局部极小，训够了会贴近 PCA")


def beta_trend(results):
    """results[beta] = {"bce": [3 个种子], "kl": [...], "knn": [...]}，beta ∈ {0.1, 1, 4}。

    检查结构，并检查趋势：β 越大，KL/张的均值越小、BCE/张的均值越大。"""
    name = "β 扫描的结果与趋势"
    if not isinstance(results, dict):
        return report(name, False, f"results 应是 dict，实际是 {type(results).__name__}")
    try:
        keys = sorted(float(k) for k in results)
    except (TypeError, ValueError):
        return report(name, False, f"键应是 β 的数值，实际 {list(results)!r}")
    if keys != sorted(float(b) for b in BETAS):
        return report(name, False, f"应恰好包含 β ∈ {BETAS}，实际 {keys}")
    means = {}
    for b in sorted(results, key=float):
        v = results[b]
        if not isinstance(v, dict) or not {"bce", "kl", "knn"} <= set(v):
            return report(name, False, f"results[{b}] 应是含 bce / kl / knn 三个列表的 dict")
        for key in ("bce", "kl", "knn"):
            vals = list(v[key])
            if len(vals) != N_SEEDS:
                return report(name, False, f"results[{b}]['{key}'] 应有 {N_SEEDS} 个种子的结果，实际 {len(vals)} 个")
            if not all(np.isfinite(float(x)) for x in vals):
                return report(name, False, f"results[{b}]['{key}'] 里有非有限数：{vals}")
        if not all(0.0 <= float(x) <= 1.0 for x in v["knn"]):
            return report(name, False, f"results[{b}]['knn'] 应是 [0, 1] 内的准确率：{list(v['knn'])}")
        means[float(b)] = {k: float(np.mean([float(x) for x in v[k]])) for k in ("bce", "kl")}
    order = sorted(means)
    kl = [means[b]["kl"] for b in order]
    bce = [means[b]["bce"] for b in order]
    if not all(x > y for x, y in zip(kl, kl[1:])):
        return report(name, False, f"KL/张的均值应随 β 增大而单调减小，实际 {np.round(kl, 2).tolist()}",
                      "β 加大了 KL 惩罚；若趋势反了，检查训练损失是不是写成了 BCE + KL/β，或评估时的 KL 是否除以了张数")
    return report(name, all(x < y for x, y in zip(bce, bce[1:])),
                  f"BCE/张的均值应随 β 增大而单调增大，实际 {np.round(bce, 2).tolist()}",
                  "重建误差与 KL 此消彼长；若不单调，检查三个 β 是否用了相同的数据、epochs 与种子")
