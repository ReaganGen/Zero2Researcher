"""第15周自查函数。用法：from zero2researcher.checks import week15 as ck; ck.reparameterize(reparameterize)

函数练习把函数本身传进来（不加括号），例如 ck.vae_kl(vae_kl)；其余检查直接传入你算出的数。
口径照 pytorch/examples/vae（BSD-3-Clause，https://github.com/pytorch/examples/tree/main/vae ）：
编码器输出 mu 与 logvar = log σ²；重参数化 z = mu + exp(0.5·logvar)·eps；
KL 项 = −½·Σ(1 + logvar − mu² − exp(logvar))（Kingma & Welling 2013 附录 B），对 batch 与潜维度全部求和。
S3 末尾“似然作为评价指标”：MNIST 先随机二值化（Salakhutdinov & Murray 2008），VAE 与 MADE（Germain et al. 2015
式 (8)(9)(11)，自然顺序，784-500-784 + 直连）在同一份 0/1 数据、同一伯努利似然上比较；bits/dim = nats ÷ (784·ln 2)；
重要性加权估计照 Burda et al. 2016。
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


# ---------- S3 末尾：似然作为评价指标（二值化、bits/dim、重要性加权、MADE） ----------
LN2 = math.log(2)


def binarize(fn):
    """fn(X, seed) -> 与 X 同形状的 0/1 张量：把灰度当作“像素为 1 的概率”抽一次（随机二值化，
    Salakhutdinov & Murray 2008），同一个种子结果相同、换种子结果不同。

    检查：只含 0/1；灰度 0 / 1 的像素保持不变；每个像素为 1 的频率 ≈ 灰度（4000 行，最大偏差 ≤ 0.05）；可复现。
    专门识别阈值二值化（灰度 > 0.5 记为 1）与没把种子传给生成器。"""
    import torch

    name = "随机二值化 binarize"
    # 造测试数据的种子不能是 fn 会用的 0、1：否则 torch.bernoulli 抽到的均匀数恰好等于概率本身，结果全是 0
    X = torch.rand(4000, 50, generator=torch.Generator().manual_seed(2026))
    X[:, :5] = 0.0
    X[:, 5:10] = 1.0
    ok, B = safe_call(name, fn, X.clone(), 0)
    if not ok:
        return False
    if not isinstance(B, torch.Tensor) or tuple(B.shape) != tuple(X.shape):
        return report(name, False, f"应返回与输入同形状 {tuple(X.shape)} 的张量，实际 {getattr(B, 'shape', type(B).__name__)!r}")
    B = B.detach().float().cpu()
    if not torch.all((B == 0) | (B == 1)):
        return report(name, False, "结果里有 0 和 1 以外的值", "二值化后每个像素只能是 0 或 1：按灰度当概率抽样")
    if torch.equal(B, (X > 0.5).float()):
        return report(name, False, "结果和“灰度 > 0.5 记为 1”完全一样——这是阈值二值化，不是随机二值化",
                      "把灰度当作概率抽样（torch.bernoulli）。两种二值化下的似然相差约 20 nats；"
                      "MADE、IWAE 等论文用的是 Salakhutdinov & Murray 2008 的随机二值化")
    if B[:, :5].sum() != 0 or B[:, 5:10].min() != 1:
        return report(name, False, "灰度为 0 的像素应永远是 0、灰度为 1 的应永远是 1", "按灰度当概率抽样：概率 0 抽不到 1")
    gap = float((B[:, 10:].mean(0) - X[:, 10:].mean(0)).abs().max())
    if gap > 0.05:
        return report(name, False, f"每个像素为 1 的频率应≈它的灰度（4000 次里最大偏差 {gap:.3f}）",
                      "每个像素独立地以“它的灰度”为概率取 1")
    ok2, B2 = safe_call(name, fn, X.clone(), 0)
    ok3, B3 = safe_call(name, fn, X.clone(), 1)
    if not (ok2 and ok3):
        return False
    if not torch.equal(B, B2.detach().float().cpu()):
        return report(name, False, "同一个种子两次调用结果不同——二值化要只抽一次、可复现",
                      "用 torch.Generator().manual_seed(seed) 建一个生成器，传给 torch.bernoulli 的 generator 参数")
    if torch.equal(B, B3.detach().float().cpu()):
        return report(name, False, "换了种子结果却完全相同", "种子要真正传进生成器")
    return report(name, True)


def bits_per_dim(fn):
    """fn(nll_nats, n_dims=784) -> nll_nats / (n_dims · ln 2)：每张图的负对数似然（nats）换成每个维度多少比特。"""
    name = "nats → bits/dim 换算 bits_per_dim"
    for nll, d in ((88.40, 784), (100.0, 784), (3.0, 1)):
        ok, got = safe_call(name, fn, nll, d)
        if not ok:
            return False
        try:
            g = float(got)
        except (TypeError, ValueError):
            return report(name, False, f"应返回一个数，实际 {type(got).__name__}")
        want = nll / (d * LN2)
        if math.isclose(g, want, rel_tol=1e-6):
            continue
        hint = "bits/dim = nats ÷ (维数 × ln 2)：先换底（1 nat = 1/ln 2 ≈ 1.443 bit），再除以维数"
        if math.isclose(g, nll / d, rel_tol=1e-6):
            hint = "你只除了维数，还停留在 nats/dim：还要再除以 ln 2（自然对数 → 以 2 为底）"
        elif math.isclose(g, nll * LN2 / d, rel_tol=1e-6):
            hint = "ln 2 用反了：nats → bits 是除以 ln 2（1 nat ≈ 1.443 bit，比特数比 nats 大）"
        elif math.isclose(g, nll / LN2, rel_tol=1e-6):
            hint = "忘了除以维数（MNIST 每张图 784 个像素）"
        elif math.isclose(g, nll / (d * math.log(10)), rel_tol=1e-6):
            hint = "用了 ln 10；bit 对应以 2 为底的对数，应除以 ln 2"
        return report(name, False, f"nll = {nll} nats、维数 {d}：结果不对", hint)
    return report(name, True)


def vae_binarized_neg_elbo(value, X_eval):
    """二值化 MNIST（全量 6 万张训练、10 epochs，pytorch/examples 结构）测试集上的 −ELBO（nats/张）。
    X_eval 是评估用的数据，**必须全是 0/1**——否则灰度图上的 105.5–105.9（讲义 S2）也会落进区间。
    区间 [99, 106]：本机 CPU 3 个种子 102.5–102.7。"""
    import torch

    name = "二值化 MNIST 上 VAE 的 −ELBO"
    try:
        X = torch.as_tensor(X_eval).detach()
    except (TypeError, ValueError, RuntimeError):
        return report(name, False, "第二个参数应传入评估用的数据张量（B_test）")
    if not torch.all((X == 0) | (X == 1)):
        return report(name, False, "评估数据里有 0 和 1 以外的值——还在灰度图上评估",
                      "灰度图上的 BCE 不是似然；用 binarize(X_test, seed=1) 得到的 B_test 训练与评估")
    return _in_range(name, value, 99, 106,
                     "本机 CPU 3 个种子 102.5–102.7。约 85–90：多半用了阈值二值化（> 0.5）而不是随机二值化；"
                     "约 130：只用了 1 万张训练 5 个 epoch（要用全量 B_all、10 epochs）")


def log_mean_exp(fn):
    """fn(log_w) -> 每一列的 log((1/K)·Σ_k exp(log_w[k]))：log_w 的形状 (K, 图片数)，返回形状 (图片数,)。
    这是重要性加权估计的最后一步（Burda et al. 2016）：先平均权重、再取对数，全程在对数上算（logsumexp − log K）。

    在构造的数组上精确核对（与训练的随机性无关）：[[0, 0], [log 3, log 3]] → [log 2, log 2]；K = 3 的非对称例子；
    对数权重是 −800 量级时不能下溢；K = 1 时原样返回。专门识别"忘了减 log K""沿错了维度""对 log w 取平均（那是 ELBO）"
    "先 exp 再求和（下溢）""把 K 写死"。"""
    import torch

    name = "对数权重的平均 log_mean_exp"
    cases = [torch.tensor([[0.0, 0.0], [math.log(3), math.log(3)]]),
             torch.tensor([[0.0, 1.0], [math.log(2), 2.0], [math.log(3), 3.0]]),
             torch.tensor([[-800.0, -1000.0, -5.0], [-801.0, -1002.0, -5.0], [-799.5, -1001.0, -5.0], [-800.2, -999.0, -5.0]]),
             torch.tensor([[1.5, -2.0, 0.25]])]

    # float32 在 −800 附近的间距约 6e-5，容差取 1e-3；要识别的错误都差 0.1 以上
    def ref(lw):
        return torch.logsumexp(lw, dim=0) - math.log(lw.shape[0])

    for lw in cases:
        K, n = lw.shape
        before = lw.clone()
        ok, got = safe_call(name, fn, lw.clone())
        if not ok:
            return False
        try:
            g = torch.as_tensor(got).detach().double().cpu()
        except (TypeError, ValueError, RuntimeError):
            return report(name, False, f"应返回一个张量，实际是 {type(got).__name__}")
        want = ref(lw.double())
        if tuple(g.shape) != (n,):
            hint = "对第 0 维（K 个样本）求：torch.logsumexp(log_w, dim=0)，结果每张图一个数"
            if tuple(g.shape) == (K,):
                hint = "沿错了维度：第 0 维是 K 个样本、第 1 维是图片——要对第 0 维做 logsumexp（dim=0）"
            return report(name, False, f"log_w 的形状 ({K}, {n}) 时应返回形状 ({n},)，实际 {tuple(g.shape)}", hint)
        if torch.allclose(g, want, atol=1e-3, rtol=0):
            continue
        hint = "log((1/K)·Σ_k w_k) = torch.logsumexp(log_w, dim=0) − log K，K = log_w.shape[0]"
        if not torch.all(torch.isfinite(g)):
            hint = ("结果里有 −inf / NaN：对数权重是几百个 nats 的负数，先 exp 再求和会下溢成 0、取对数得 −∞。"
                    "全程在对数上算：torch.logsumexp(log_w, dim=0) − log K")
        elif K > 1 and torch.allclose(g, want + math.log(K), atol=1e-3, rtol=0):
            hint = "忘了减 log K：logsumexp 算的是 log Σ_k w_k，平均还要再除以 K，即减去 log K（K = log_w.shape[0]）"
        elif K > 1 and torch.allclose(g, lw.double().mean(0), atol=1e-3, rtol=0):
            hint = "你对 log w 取了平均——那是先取对数、再平均，也就是 ELBO；这里要先平均权重、再取对数：logsumexp − log K"
        elif K > 1 and n > 1 and K == n and torch.allclose(g, torch.logsumexp(lw.double(), dim=1) - math.log(n), atol=1e-3, rtol=0):
            hint = "沿错了维度：第 0 维是 K 个样本、第 1 维是图片——要对第 0 维做 logsumexp（dim=0）"
        elif K > 1 and any(torch.allclose(g, want + math.log(K) - math.log(c), atol=1e-3, rtol=0) for c in (10, 100, 1000)):
            hint = "减的是 log K，K 要从 log_w 的形状来（log_w.shape[0]），不要写死成某个数"
        elif torch.allclose(g, -want, atol=1e-3, rtol=0):
            hint = "符号反了：这里返回 log p(x) 的估计（对数权重的平均），负号留给调用它的 iw_nll"
        elif torch.allclose(g, lw.double().max(0).values - math.log(K), atol=1e-3, rtol=0):
            hint = "只取了最大的一项：logsumexp 要把所有 K 项都加进来（torch.logsumexp 会处理数值稳定）"
        if not torch.equal(lw, before):
            hint = "函数改动了传进来的 log_w；" + hint
        return report(name, False, f"log_w = {lw.tolist()}：期望 {[round(float(x), 4) for x in want]}，"
                      f"实际 {[round(float(x), 4) for x in g]}", hint)
    return report(name, True)


def iw_estimates(iw, neg_elbo):
    """iw = {1: …, 10: …, 100: …}：同一个 VAE 的 −log p(x) 重要性加权估计（nats/张）；neg_elbo：它的 −ELBO。

    检查（Burda et al. 2016 定理 1）：K = 1 ≈ −ELBO（差 ≤ 1，两者各抽一次 z）；K 越大估计越小（容差 0.2）；
    K = 100 比 −ELBO 小 3–7 nats，且 K = 10 与 K = 100 之差 ≤ 2.5（本机 CPU 缺口 4.76–4.8、MPS 4.77，K = 10→100 降 1.33；
    logsumexp 之后忘了减 log K 时缺口约 9.4、K = 10→100 降约 3.6——旧的上限 10 拦不住它）。"""
    name = "重要性加权估计（K = 1、10、100）"
    try:
        v = {int(k): float(x) for k, x in iw.items()}
        e = float(neg_elbo)
    except (TypeError, ValueError, AttributeError):
        return report(name, False, "iw 应是 {K: 估计值} 的 dict，neg_elbo 应是一个数")
    if sorted(v) != [1, 10, 100]:
        return report(name, False, f"应恰好包含 K = 1、10、100，实际 {sorted(v)}")
    if not all(math.isfinite(x) for x in [*v.values(), e]):
        return report(name, False, f"估计值里有 NaN / inf：{v}", "对数权重要先 logsumexp 再减 log K，不要先 exp 再求和（会下溢）")
    if abs(v[1] - e) > 1.0:
        return report(name, False, f"K = 1 的估计（{v[1]:.2f}）应≈ −ELBO（{e:.2f}）",
                      "K = 1 时“先平均权重再取对数”就是单个对数权重 = ELBO 的一次采样；"
                      "检查 log p(z) 与 log q(z|x) 的常数项与符号，以及返回的是负对数似然（正数）")
    if not (v[1] + 0.2 >= v[10] and v[10] + 0.2 >= v[100]):
        return report(name, False, f"估计应随 K 增大而减小（界变紧），实际 K=1/10/100：{v[1]:.2f} / {v[10]:.2f} / {v[100]:.2f}",
                      "先在 K 个对数权重上做 logsumexp，再减 log K（先用 ck.log_mean_exp 单独检查这一步）")
    gap = e - v[100]
    if gap > 7.0 or v[10] - v[100] > 2.5:
        return report(name, False, f"变紧得太多：K = 100 比 −ELBO 小 {gap:.2f} nats、K = 10→100 降了 {v[10] - v[100]:.2f}"
                      "（本机约 4.8 与 1.3）",
                      "logsumexp 之后要减 log K（K = 100 时差 log 100 ≈ 4.6 nats）——先用 ck.log_mean_exp 单独检查；"
                      "也检查 log q(z|x) 里用的是 logvar（= log σ²）")
    if gap < 3.0:
        return report(name, False, f"K = 100 应比 −ELBO 小 3–7 nats（本机约 4.8），实际只小 {gap:.2f}",
                      "几乎没有变紧：你可能对对数权重直接取了平均（那还是 ELBO），应先平均权重、再取对数")
    return report(name, True)


def made_masks(fn):
    """fn(n_in, n_hidden, rng) -> (mask_in (H, D), mask_out (D, H), mask_direct (D, D))：Germain et al. 2015
    式 (8)(9)(11)，自然顺序（输入按 1…D 编号）、m(k) ∈ {1, …, D−1} 由 rng 抽取。

    在 D = 12、H = 40 的小例子上检查：形状、只含 0/1、自回归性质（M_V·M_W 与直连掩码在 d′ ≥ d 处全为 0）、
    m(k) 的范围、与式 (8)(9) 逐项一致、直连是严格下三角、换 rng 掩码不同、同一个种子的 rng 掩码相同（可复现）。"""
    import torch

    name = "MADE 的掩码 made_masks"
    D, H = 12, 40
    ok, out = safe_call(name, fn, D, H, np.random.default_rng(0))
    if not ok:
        return False
    try:
        mi, mo, md = [torch.as_tensor(m).detach().float().cpu() for m in out]
    except (TypeError, ValueError):
        return report(name, False, "应返回三个掩码 (mask_in, mask_out, mask_direct)")
    if tuple(mi.shape) != (H, D) or tuple(mo.shape) != (D, H) or tuple(md.shape) != (D, D):
        return report(name, False, f"形状应为 ({H}, {D})、({D}, {H})、({D}, {D})，实际 {tuple(mi.shape)}、{tuple(mo.shape)}、{tuple(md.shape)}",
                      "nn.Linear 的权重形状是 (输出, 输入)，掩码要与权重同形状")
    if not all(torch.all((m == 0) | (m == 1)) for m in (mi, mo, md)):
        return report(name, False, "掩码只能含 0 和 1")
    paths = mo @ mi                                        # (D, D)：输出 d ← 输入 d′ 经隐藏层的路径条数
    upper = torch.triu(torch.ones(D, D))                   # d′ ≥ d 的位置（含对角线）
    leak = (paths + md) * upper
    if leak.sum() > 0:
        i, j = map(int, torch.nonzero(leak)[0])
        return report(name, False, f"输出 {i + 1} 能看到输入 {j + 1}（编号从 1 起）：违反了自回归性质"
                      "（输出 d 只能依赖 x_1 … x_{d−1}）",
                      "输入→隐藏用 m(k) ≥ d（大于等于），隐藏→输出用 d > m(k)（严格大于），直连用严格下三角")
    m = mi.sum(1)                                          # 自然顺序下，单元 k 连接的输入个数就是 m(k)
    if m.min() < 1 or m.max() > D - 1:
        return report(name, False, f"每个隐藏单元的 m(k) 应在 1 … D−1 之间，实际 {int(m.min())} … {int(m.max())}",
                      "m(k) = 0 的单元什么也看不到；m(k) = D 的单元看到了全部输入、哪个输出都不能用。"
                      "注意 rng.integers(low, high) 不含 high")
    d = torch.arange(1, D + 1, dtype=torch.float32)
    if not torch.equal(mi, (d[None, :] <= m[:, None]).float()):
        return report(name, False, "输入→隐藏的掩码不是式 (8) 的形状：单元 k 应连接 x_1 … x_{m(k)}（前 m(k) 个输入全为 1）",
                      "M^W[k, d] = 1[m(k) ≥ d]——是“大于等于”，不是“等于”")
    if not torch.equal(mo, (d[:, None] > m[None, :]).float()):
        return report(name, False, "隐藏→输出的掩码不是式 (9) 的形状：输出 d 应连接所有 m(k) < d 的隐藏单元",
                      "M^V[d, k] = 1[d > m(k)]；只要不违反自回归性质，能连的都要连上")
    if not torch.equal(md, torch.tril(torch.ones(D, D), -1)):
        return report(name, False, "直连掩码应是严格下三角（对角线及以上为 0、以下全为 1）：输出 d 直接看到所有 x_{<d}",
                      "式 (11)：直连同样要满足自回归性质，但不经过隐藏层，所以 d′ < d 的位置全连")
    ok2, out2 = safe_call(name, fn, D, H, np.random.default_rng(1))
    if not ok2:
        return False
    if len(torch.unique(m)) < 2 or torch.equal(torch.as_tensor(out2[0]).detach().float().cpu(), mi):
        return report(name, False, "换了 rng，掩码却完全相同（或所有单元的 m(k) 都一样）", "m(k) 要用传进来的 rng 随机抽")
    ok3, out3 = safe_call(name, fn, D, H, np.random.default_rng(0))
    if not ok3:
        return False
    if not torch.equal(torch.as_tensor(out3[0]).detach().float().cpu(), mi):
        return report(name, False, "同一个种子的 rng 两次调用，得到的掩码不同——MADE(seed=0) 每次运行都会不一样，结果不可复现",
                      "m(k) 要用传进来的 rng 抽（rng.integers(...)），不要在函数里新建 np.random.default_rng() 或用全局的 np.random")
    return report(name, True)


def made_autoregressive(model, n_in=784):
    """用雅可比矩阵检查整个网络：输出 d 对输入 x_d, x_{d+1}, … 的导数必须全为 0（模型可以在任何 device 上，
    检查时复制一份到 CPU）。“偷看”了答案的模型似然会好得不真实。"""
    import copy

    import torch

    name = "MADE 的自回归性质（雅可比检查）"
    try:
        cpu_model = copy.deepcopy(model).cpu().eval()
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"无法复制模型 {type(e).__name__}: {e}", "传入 MADE 模型本身（nn.Module）")
    x = torch.rand(1, n_in, generator=torch.Generator().manual_seed(0))    # 不动全局随机数
    try:
        J = torch.autograd.functional.jacobian(lambda z: cpu_model(z)[0], x)[:, 0, :]
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"前向计算出错 {type(e).__name__}: {e}")
    if tuple(J.shape) != (n_in, n_in):
        return report(name, False, f"模型输出应有 {n_in} 个 logit（每个像素一个），雅可比形状实际 {tuple(J.shape)}")
    upper = torch.triu(torch.ones(n_in, n_in))
    bad = J.abs() * upper > 0
    if bad.any():
        i, j = map(int, torch.nonzero(bad)[0])
        return report(name, False, f"输出 {i + 1} 依赖输入 {j + 1}（编号从 1 起）：模型“偷看”了答案，似然会低得不真实",
                      "每个 MaskedLinear 都要用 weight * mask；直连那一层也要掩码")
    if float(J.abs().sum()) == 0:
        return report(name, False, "输出对任何输入都没有导数", "检查掩码是不是全为 0")
    return report(name, True)


def made_test_nll(value):
    """二值化 MNIST 测试集上 MADE（784-500-784 + 直连、自然顺序、Adam 1e-3、batch 100、10 epochs）的
    −log p(x)（nats/张）。区间 [88, 97]：本机 CPU 3 个种子 92.2–92.5。"""
    return _in_range("MADE 测试集负对数似然", value, 88, 97,
                     "本机 CPU 3 个种子 92.2–92.5（30 epochs 最好约 91.3）。约 100–105：多半漏了输入→输出的直连"
                     "（本机无直连 10 epochs 103.7）；约 72：用了阈值二值化；远低于 85：检查自回归性质"
                     "（ck.made_autoregressive）——“偷看”答案的模型似然会好得不真实；"
                     "约 206：模型没训练起来（这是独立像素基线的水平）")
