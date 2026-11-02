"""Generative-modelling toolkit for weeks 16-17 (Flow Matching, DDPM, classifier-free guidance).

课程不教但绕不开的部分都在这里：2D 玩具分布、2D 样本质量指标、Flow Matching 与 DDPM 之间的
时间/参数化换算、小型 U-Net、MNIST 张量，以及判定生成数字用的小分类器。

**课程要教的部分不在这里**（学习者自己写）：Flow Matching 的插值与速度目标、训练循环、欧拉采样、
DDPM 的 ᾱ 调度与祖先采样、classifier-free guidance 的组合公式。

全课程统一的两套约定（第 16–17 周讲义）::

    Flow Matching（MIT 6.S184 讲义约定）:  x_t = t * x_data + (1 - t) * eps    t = 0 是噪声，t = 1 是数据
                                           速度目标 velocity = x_data - eps
    DDPM（Ho, Jain & Abbeel 2020 约定）:     x   = sqrt(abar) * x_data + sqrt(1 - abar) * eps
                                           时间 0 是数据；abar（ᾱ）越小噪声越多

两套约定的时间方向相反。本模块的换算函数一律用 ᾱ（``alpha_bar``）描述 DDPM 一侧的噪声水平，
所以与具体的 β 调度无关；参数名带约定后缀（``t_fm``、``alpha_bar``），从命名上杜绝混用。

数据与指标函数接收、返回 float32 的 torch 张量（指标返回 Python float）。
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "sample_moons", "sample_gaussian_mixture", "gaussian_mixture_means", "sample_checkerboard",
    "sample_spiral", "energy_distance", "knn_precision", "knn_recall",
    "fm_to_ddpm_time", "ddpm_to_fm_time", "fm_to_ddpm", "ddpm_to_fm", "convert_prediction",
    "SmallUNet", "count_parameters", "load_mnist", "downsample_mnist",
    "DigitClassifier", "train_mnist_classifier", "classifier_scores", "within_class_diversity",
]


# ---------------------------------------------------------------------------
# 随机数：rng 可以是 None（用 torch 的全局随机数，受 torch.manual_seed 控制）、
# 一个整数种子，或一个 torch.Generator
# ---------------------------------------------------------------------------
def _generator(rng):
    if rng is None:
        return None
    if isinstance(rng, torch.Generator):
        return rng
    return torch.Generator().manual_seed(int(rng))


# ---------------------------------------------------------------------------
# 2D 玩具分布（全部独立同分布采样，返回 (n, 2) float32 张量）
# ---------------------------------------------------------------------------
def sample_moons(n, noise=0.05, scale=3.5, rng=None):
    """两个交错的半月（"月牙"），**独立同分布**版本。

    与 ``sklearn.datasets.make_moons`` 形状相同（上弧 (cos θ, sin θ)，下弧 (1 − cos θ, 0.5 − sin θ)，
    再加各向同性高斯噪声 ``noise``），但每个点**独立地**随机选一条弧、随机取角度 θ ~ U(0, π)。
    make_moons 沿弧用 ``np.linspace`` 等距放点，两次调用的样本不是独立同分布，会让"真样本 vs 真样本"
    的能量距离稳定地偏到约 −0.004（第六阶段调研报告 4.4）；用这个版本，噪声地板才在 0 附近。

    最后乘 ``scale`` 并平移到原点附近（默认 3.5 倍，数据尺度与 N(0, I) 的噪声相当）。
    """
    g = _generator(rng)
    upper = torch.rand(n, generator=g) < 0.5
    theta = torch.rand(n, generator=g) * math.pi
    x = torch.where(upper, torch.cos(theta), 1.0 - torch.cos(theta))
    y = torch.where(upper, torch.sin(theta), 0.5 - torch.sin(theta))
    pts = torch.stack([x, y], dim=1) + noise * torch.randn(n, 2, generator=g)
    return ((pts - torch.tensor([0.5, 0.25])) * scale).float()


def gaussian_mixture_means(k=2, radius=3.0):
    """k 个分量的中心：均匀排在半径为 radius 的圆上，第 j 个在角度 2πj/k 处。返回 (k, 2) 张量。"""
    angles = torch.arange(k, dtype=torch.float32) * (2 * math.pi / k)
    return radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


def sample_gaussian_mixture(n, k=2, radius=3.0, std=0.7, rng=None, return_labels=False):
    """k 个等权、各向同性的高斯分量（中心见 ``gaussian_mixture_means``），每个点独立选分量。

    默认 k=2：中心 (±3, 0)、标准差 0.7（调研报告的双高斯）。第 17 周条件生成用
    ``k=3, radius=1.5, std=0.8``（三个刻意重叠的高斯）。``return_labels=True`` 时同时返回分量编号 (n,)。
    """
    g = _generator(rng)
    labels = torch.randint(0, k, (n,), generator=g)
    x = gaussian_mixture_means(k, radius)[labels] + std * torch.randn(n, 2, generator=g)
    return (x.float(), labels) if return_labels else x.float()


def sample_checkerboard(n, rng=None):
    """8 个格子的棋盘格，范围 [−4, 4]²。

    改写自 FFJORD ``lib/toy_data.py`` 的 ``checkerboard``（rtqichen/ffjord，MIT License，
    Copyright (c) 2018 Ricky Tian Qi Chen and Will Grathwohl）：numpy → torch，加 rng 参数。
    """
    g = _generator(rng)
    x1 = torch.rand(n, generator=g) * 4 - 2
    x2_ = torch.rand(n, generator=g) - torch.randint(0, 2, (n,), generator=g).float() * 2
    x2 = x2_ + torch.floor(x1) % 2
    return (torch.stack([x1, x2], dim=1) * 2).float()


def sample_spiral(n, noise=0.1, rng=None):
    """两条交织的螺旋臂（约 1.5 圈）。

    改写自 FFJORD ``lib/toy_data.py`` 的 ``2spirals``（MIT License，版权同上）：numpy → torch；
    原版前一半点画一条臂、后一半画另一条，这里改成每个点**独立地**随机选臂（保证独立同分布）。
    """
    g = _generator(rng)
    r = torch.sqrt(torch.rand(n, generator=g)) * 540 * (2 * math.pi) / 360
    dx = -torch.cos(r) * r + torch.rand(n, generator=g) * 0.5
    dy = torch.sin(r) * r + torch.rand(n, generator=g) * 0.5
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    x = torch.stack([dx, dy], dim=1) * sign[:, None] / 3
    return (x + noise * torch.randn(n, 2, generator=g)).float()


# ---------------------------------------------------------------------------
# 2D 样本质量指标
# ---------------------------------------------------------------------------
def _as_2d_float64(a):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().to(torch.float64)
    else:
        a = torch.as_tensor(np.asarray(a, dtype=np.float64))
    return a.reshape(len(a), -1)


def _cdist(a, b):
    return torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")


def energy_distance(x, y):
    """两组样本之间的能量距离（Székely & Rizzo 2013），**平方形式、U 统计量**估计。

        E(X, Y) = 2·E‖X − Y‖ − E‖X − X'‖ − E‖Y − Y'‖

    总体上 E ≥ 0，且当且仅当两个分布相同时 E = 0。这里组内两项都去掉了对角线（自己到自己的距离 0），
    除以 n(n − 1)，是无偏的 U 统计量——所以两组样本来自同一分布时，估计值在 0 上下波动，
    **可以是很小的负数**（n = 2000 时"真 vs 真"的标准差约 0.002–0.005，随数据而变：
    本机 150 对独立样本，螺旋约 0.002、月牙与棋盘格约 0.003、双高斯约 0.004–0.005）。

    注意：``scipy.stats.energy_distance`` 只支持 1D，而且返回的是上式的平方根，与这里不可直接比较。
    能量距离对"整体位置与尺度"敏感，对"细结构"（例如棋盘格塌成一团）不敏感——课程同时报告
    ``knn_precision``。数值随数据尺度变化，不同数据集之间的能量距离不可比较。

    x: (n, d)，y: (m, d)，张量或数组；返回 Python float。
    """
    x, y = _as_2d_float64(x), _as_2d_float64(y)
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        raise ValueError("每组至少需要 2 个样本")
    dxy = _cdist(x, y).mean()
    dxx = _cdist(x, x).sum() / (n * (n - 1))
    dyy = _cdist(y, y).sum() / (m * (m - 1))
    return float(2 * dxy - dxx - dyy)


def _knn_radii(points, k, chunk=2048):
    radii = []
    for i in range(0, len(points), chunk):
        d = _cdist(points[i:i + chunk], points)
        radii.append(d.kthvalue(k + 1, dim=1).values)   # 第 1 小是自己（距离 0），所以取第 k+1 小
    return torch.cat(radii)


def knn_precision(samples, reference, k=5):
    """k-NN precision（Kynkäänniemi et al. 2019, "Improved Precision and Recall Metric for
    Assessing Generative Models" 的口径）：生成样本中，落在**某个真实样本的 k-NN 球**内的比例。

    每个真实点的球半径 = 它到第 k 个最近的其他真实点的距离；球的并集近似"真实数据所在的地方"。
    precision 衡量"生成的点有多少画在了数据该在的地方"（1 = 没有画到空白处）；
    它不管有没有漏掉模式——那由 ``knn_recall`` 衡量。取值 [0, 1]，越大越好。

    与原文的区别：原文默认 k = 3，在 VGG-16 特征空间里算，每组 5 万个样本；本课程直接在 2D 坐标上算、
    每组约 2000 个点，默认取 k = 5——点少时 k = 3 的球很细，连"真 vs 真"也只有约 0.96–0.98，
    k = 5 让它接近 1（本机四份 2D 数据 0.99–1.00），"满分"更好读。k 越大球越大、precision 越高，
    所以只有同一个 k 下的数值可以比较。
    """
    samples, reference = _as_2d_float64(samples), _as_2d_float64(reference)
    radii = _knn_radii(reference, k)
    inside = []
    for i in range(0, len(samples), 2048):
        d = _cdist(samples[i:i + 2048], reference)
        inside.append((d <= radii[None, :]).any(dim=1))
    return float(torch.cat(inside).double().mean())


def knn_recall(samples, reference, k=5):
    """k-NN recall：真实样本中，落在某个**生成样本**的 k-NN 球内的比例（衡量有没有漏掉模式）。

    就是把 ``knn_precision`` 的两组样本对调。取值 [0, 1]，越大越好。
    """
    return knn_precision(reference, samples, k)


# ---------------------------------------------------------------------------
# Flow Matching（t=0 噪声）与 DDPM（时间 0 = 数据）之间的换算
# ---------------------------------------------------------------------------
def _t(v, like=None):
    if isinstance(v, torch.Tensor):
        return v
    dtype = like.dtype if isinstance(like, torch.Tensor) else torch.float32
    device = like.device if isinstance(like, torch.Tensor) else None
    return torch.as_tensor(v, dtype=dtype, device=device)


def fm_to_ddpm_time(t_fm):
    """FM 时间 t（t=0 噪声）→ 噪声水平相同的 DDPM ᾱ。

    "噪声水平相同"指信噪比相同：FM 状态 t·x + (1−t)·ε 的信噪比是 t²/(1−t)²，DDPM 状态的是 ᾱ/(1−ᾱ)。
    令两者相等得 ᾱ = t² / (t² + (1−t)²)。注意这**不是**简单的 1 − t：两条路径的噪声随时间变化的方式不同。
    """
    t = _t(t_fm)
    return t ** 2 / (t ** 2 + (1 - t) ** 2)


def ddpm_to_fm_time(alpha_bar):
    """DDPM 的 ᾱ → 信噪比相同的 FM 时间 t = √ᾱ / (√ᾱ + √(1−ᾱ))（``fm_to_ddpm_time`` 的反函数）。"""
    ab = _t(alpha_bar)
    a, s = ab.sqrt(), (1 - ab).sqrt()
    return a / (a + s)


def ddpm_to_fm(x_ddpm, alpha_bar):
    """DDPM 状态 x = √ᾱ·x_data + √(1−ᾱ)·ε → 同一对 (x_data, ε) 在 FM 路径上的状态。

    返回 (x_fm, t_fm)，其中 x_fm = x_ddpm / (√ᾱ + √(1−ᾱ))，t_fm = √ᾱ / (√ᾱ + √(1−ᾱ))。
    两个状态只差一个缩放因子：数据与噪声的"配方比例"相同，所以 ε 与 x_data 在两种语言里是同一个东西。
    alpha_bar 可以是数或能与 x_ddpm 广播的张量（例如 (B, 1) 或 (B, 1, 1, 1)）。
    """
    ab = _t(alpha_bar, x_ddpm)
    a, s = ab.sqrt(), (1 - ab).sqrt()
    return x_ddpm / (a + s), a / (a + s)


def fm_to_ddpm(x_fm, t_fm):
    """FM 状态 x = t·x_data + (1−t)·ε → DDPM 状态；返回 (x_ddpm, alpha_bar)。``ddpm_to_fm`` 的反函数。"""
    t = _t(t_fm, x_fm)
    c = 1.0 / torch.sqrt(t ** 2 + (1 - t) ** 2)
    return c * x_fm, (c * t) ** 2


_PARAMS = {"fm": ("eps", "data", "velocity", "score"), "ddpm": ("eps", "data", "v", "score")}


def convert_prediction(pred, x_t, from_param, to_param, *, t_fm=None, alpha_bar=None):
    """在同一个状态 x_t 上，把网络的一种预测换算成另一种。

    **恰好给一个**时间参数，它同时指明约定：

    - ``t_fm=``：FM 约定，x_t = t·data + (1−t)·ε。可用参数化 ``"eps"``（噪声 ε）、``"data"``（干净数据）、
      ``"velocity"``（FM 速度 u = data − ε，第 16 周网络的输出）、``"score"``。
    - ``alpha_bar=``：DDPM 约定，x_t = √ᾱ·data + √(1−ᾱ)·ε。可用 ``"eps"``、``"data"``、
      ``"v"``（Salimans & Ho 2022 的 v-prediction：v = √ᾱ·ε − √(1−ᾱ)·data）、``"score"``。

    ``"score"`` 是该约定下 x_t 的分布的 score ∇log p_t(x_t) 的估计。记数据系数 a、噪声系数 σ
    （FM：a = t、σ = 1 − t；DDPM：a = √ᾱ、σ = √(1−ᾱ)），则 score = −ε/σ；反过来 ε = −σ·score，
    data = (x_t + σ²·score)/a（Tweedie 公式）。所以"预测噪声"就是在做（按 σ² 加权的）去噪 score matching。

    思路：任何一种预测加上 x_t，都能解出 (data, ε) 这一对，再由这一对算出想要的参数化。
    特别提醒：(1) 用 "data" 而不是 "x0" 命名干净数据——FM 文献里 x0 往往指噪声；
    (2) v-prediction 的 v **不是** FM 的速度（它是 DDPM 约定下、方向"数据→噪声"的量）；
    (3) 第三套约定（例如 Google DeepMind 博客 *Diffusion Meets Flow Matching*）把速度定义成 ε − x，
    与这里的 data − ε 正好差一个负号。
    (4) 从 ε 求 data 要除以 t（或 √ᾱ），在纯噪声端会把误差放大很多倍。
    (5) score 依约定而不同：同一对 (data, ε) 在两套约定里 ε 相同，score 却差一个倍数——
    FM 状态 = DDPM 状态 ÷ k（k = √ᾱ + √(1−ᾱ)，见 ``ddpm_to_fm``），所以 score_FM = k·score_DDPM。
    在数据端 σ → 0，由 ε 求 score 要除以 σ，会发散。
    """
    if (t_fm is None) == (alpha_bar is None):
        raise ValueError("恰好给一个时间参数：t_fm=（FM 约定，t=0 噪声）或 alpha_bar=（DDPM 约定）")
    conv = "fm" if t_fm is not None else "ddpm"
    for p in (from_param, to_param):
        if p not in _PARAMS[conv]:
            raise ValueError(f"{conv.upper()} 约定下的参数化只有 {_PARAMS[conv]}，收到 {p!r}"
                             + ("（FM 速度请用 t_fm=）" if p == "velocity" else "")
                             + ("（v-prediction 请用 alpha_bar=）" if p == "v" else ""))
    if conv == "fm":
        a = _t(t_fm, x_t)
        s = 1 - a
    else:
        ab = _t(alpha_bar, x_t)
        a, s = ab.sqrt(), (1 - ab).sqrt()
    if from_param == "eps":
        eps = pred
        data = (x_t - s * eps) / a
    elif from_param == "data":
        data = pred
        eps = (x_t - a * data) / s
    elif from_param == "velocity":          # u = data − ε，且 a + s = 1
        data = x_t + s * pred
        eps = x_t - a * pred
    elif from_param == "score":             # score = −ε/s（s 是噪声系数）；data 即 Tweedie (x_t + s²·score)/a
        eps = -s * pred
        data = (x_t + s ** 2 * pred) / a
    else:                                   # "v" = a·ε − s·data，且 a² + s² = 1
        data = a * x_t - s * pred
        eps = s * x_t + a * pred
    if to_param == "eps":
        return eps
    if to_param == "data":
        return data
    if to_param == "velocity":
        return data - eps
    if to_param == "score":
        return -eps / s
    return a * eps - s * data


# ---------------------------------------------------------------------------
# 小型 U-Net（改编自 MIT 6.S184 2025 Lab 3 的 MNISTUNet）
# ---------------------------------------------------------------------------
class _FourierEncoder(nn.Module):
    """随机 Fourier 特征的时间编码（MIT Lab 3 原样；其自身基于 lucidrains/denoising-diffusion-pytorch, MIT）。"""

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.weights = nn.Parameter(torch.randn(1, dim // 2))

    def forward(self, t):
        freqs = t.view(-1, 1) * self.weights * 2 * math.pi
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1) * math.sqrt(2)


def _norm(c):
    return nn.GroupNorm(min(8, c), c)


class _ResidualLayer(nn.Module):
    def __init__(self, channels, t_dim, y_dim):
        super().__init__()
        self.block1 = nn.Sequential(nn.SiLU(), _norm(channels), nn.Conv2d(channels, channels, 3, padding=1))
        self.block2 = nn.Sequential(nn.SiLU(), _norm(channels), nn.Conv2d(channels, channels, 3, padding=1))
        self.time_adapter = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, channels))
        self.y_adapter = (nn.Sequential(nn.Linear(y_dim, y_dim), nn.SiLU(), nn.Linear(y_dim, channels))
                          if y_dim else None)

    def forward(self, x, t_embed, y_embed):
        h = self.block1(x) + self.time_adapter(t_embed)[:, :, None, None]
        if self.y_adapter is not None:
            h = h + self.y_adapter(y_embed)[:, :, None, None]
        return x + self.block2(h)


class SmallUNet(nn.Module):
    """小型 U-Net：``model(x, t)`` 或 ``model(x, t, y)`` → 与 x 同形状的张量。

    课程把它当黑盒用，只需要懂输入输出：

    - ``x``：(B, in_ch, img_size, img_size) 的图像（课程里是缩放到 [−1, 1] 的 MNIST）；
    - ``t``：每张图一个时间，形状 (B,)、(B, 1)、(B, 1, 1, 1) 或一个数都行（内部展平成 (B,)）；
      时间的**含义由训练方式决定**：FM 里是 t_fm（0 噪声），DDPM 里是 DDPM 时间（0 数据）；
    - ``y``：类别标签 (B,)，整数 0…num_classes−1；**null（无条件）类别 = num_classes**，
      即 ``model.null_class``。只有 ``num_classes`` 不是 None 时才需要（也才接受）y。
    - 输出的含义也由训练决定：FM 训练出速度，DDPM 训练出噪声 ε。

    结构：初始卷积 → 每一级 ``n_res`` 个残差层 + stride-2 卷积下采样 → 中间残差层 →
    双线性上采样 + 卷积 + 加法 skip → 残差层 → 输出卷积；时间（与类别）嵌入经小 MLP 加到每个残差层的通道上。
    通道数 ``channels[i] = base_ch · 2^i``，共 ``levels`` 级（默认 14×14 两级、28×28 三级）；
    ``img_size`` 必须能被 2^(levels−1) 整除。参数量：

    =================================================  ===========
    配置                                               参数量
    =================================================  ===========
    ``SmallUNet(img_size=14)``（base 32、2 级、n_res 1）   约 16 万
    同上 + ``num_classes=10``                          约 17 万
    ``SmallUNet(img_size=28, n_res=1)``（3 级）           约 69 万
    ``SmallUNet(img_size=28, n_res=2)``（无条件）          约 119 万
    同上 + ``num_classes=10``（= MIT 原配置）            约 123 万
    =================================================  ===========

    MIT 原版 ``MNISTUNet([32, 64, 128], 2, 40, 40)`` 带类别嵌入，参数张量的形状与
    ``SmallUNet(img_size=28, n_res=2, num_classes=10)`` 逐一相同（1,233,325 个参数）；
    119 万是去掉类别支路之后的数。

    出处与修改：改编自 MIT 6.S184（IAP 2025）Lab 3 的 ``MNISTUNet``
    （https://github.com/eje24/iap-diffusion-labs ，``2025`` 分支；MIT License，
    Copyright (c) 2025 Ezra Erives and Peter Holderrieth；许可全文见本文件末尾）。修改：
    (1) BatchNorm → GroupNorm（避免 eval 模式下统计量随时间 t 漂移）；(2) 直接处理 28×28 或 14×14，
    不再先放大到 32×32；(3) 通道数由 ``base_ch``/``levels`` 生成；(4) 类别嵌入可选，
    ``num_classes=None`` 时整条类别支路不建；(5) 时间输入接受多种形状；(6) 无注意力层（原版也没有）。
    Fourier 时间编码与残差层结构保持原样。
    """

    def __init__(self, in_ch=1, base_ch=32, levels=None, n_res=1, num_classes=None, img_size=14,
                 t_embed_dim=40, y_embed_dim=40, channels=None):
        super().__init__()
        if channels is None:
            if levels is None:
                levels = 2 if img_size <= 16 else 3
            channels = [base_ch * 2 ** i for i in range(levels)]
        channels = list(channels)
        if img_size % (2 ** (len(channels) - 1)) != 0:
            raise ValueError(f"img_size={img_size} 不能被 2^{len(channels) - 1} 整除："
                             f"{len(channels)} 级 U-Net 要下采样 {len(channels) - 1} 次")
        self.img_size, self.in_ch, self.channels = img_size, in_ch, channels
        self.num_classes = num_classes
        self.null_class = num_classes
        y_dim = y_embed_dim if num_classes is not None else 0
        self.init_conv = nn.Sequential(nn.Conv2d(in_ch, channels[0], 3, padding=1), _norm(channels[0]), nn.SiLU())
        self.time_embedder = _FourierEncoder(t_embed_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, y_dim) if num_classes is not None else None
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            self.encoders.append(nn.ModuleList([_ResidualLayer(c_in, t_embed_dim, y_dim) for _ in range(n_res)]))
            self.downs.append(nn.Conv2d(c_in, c_out, 3, stride=2, padding=1))
        self.mid = nn.ModuleList([_ResidualLayer(channels[-1], t_embed_dim, y_dim) for _ in range(n_res)])
        for c_in, c_out in reversed(list(zip(channels[:-1], channels[1:]))):
            self.ups.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                          nn.Conv2d(c_out, c_in, 3, padding=1)))
            self.decoders.append(nn.ModuleList([_ResidualLayer(c_in, t_embed_dim, y_dim) for _ in range(n_res)]))
        self.final_conv = nn.Conv2d(channels[0], in_ch, 3, padding=1)

    def forward(self, x, t, y=None):
        B = x.shape[0]
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(B)                     # 一个数：所有图共用同一个时间（采样时的常见用法）
        elif t.numel() != B:
            raise ValueError(f"时间 t 应是一个数或每张图一个（共 {B} 个），实际有 {t.numel()} 个")
        t_embed = self.time_embedder(t)
        if self.y_embedder is None:
            if y is not None:
                raise ValueError("这是无条件模型（num_classes=None），不接受类别标签 y")
            y_embed = None
        else:
            if y is None:
                raise ValueError(f"条件模型需要类别标签 y；无条件预测请传 null 类别 {self.null_class}")
            y_embed = self.y_embedder(torch.as_tensor(y, device=x.device).long().reshape(B))
        h = self.init_conv(x)
        skips = []
        for blocks, down in zip(self.encoders, self.downs):
            for block in blocks:
                h = block(h, t_embed, y_embed)
            h = down(h)
            skips.append(h)
        for block in self.mid:
            h = block(h, t_embed, y_embed)
        for up, blocks in zip(self.ups, self.decoders):
            h = up(h + skips.pop())
            for block in blocks:
                h = block(h, t_embed, y_embed)
        return self.final_conv(h)


def count_parameters(model):
    """可训练参数的总个数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# MNIST 张量
# ---------------------------------------------------------------------------
def downsample_mnist(x, size=14):
    """把 28×28 的图像用平均池化缩小到 size×size（size 须整除 28，课程用 14）。

    接受 (N, 28, 28) 或 (N, C, 28, 28)，返回 (N, C, size, size)。CPU 上 14×14 是"几分钟内训出
    像样数字"的现实上限（第六阶段调研报告 5.6）。
    """
    x = torch.as_tensor(x).float()
    if x.dim() == 3:
        x = x[:, None]
    if x.shape[-1] % size != 0:
        raise ValueError(f"size={size} 必须整除原图边长 {x.shape[-1]}")
    k = x.shape[-1] // size
    return F.avg_pool2d(x, k) if k > 1 else x


def load_mnist(size=14, train=True, n=None, seed=0):
    """MNIST 图像与标签，缩放到 [−1, 1]（生成模型的惯例：背景 −1、笔画 +1）。

    返回 (x, y)：x 为 (N, 1, size, size) float32，y 为 (N,) int64。``n`` 不为 None 时按 ``seed``
    固定地随机抽 n 张。首次调用下载到 ``data.download_root()``（约 11MB，各周共用）。
    """
    from torchvision import datasets
    from . import data

    for attempt in (1, 2):
        try:
            ds = datasets.MNIST(root=data.download_root(), train=train, download=True)
            break
        except Exception as e:  # noqa: BLE001  网络抖动时重试一次
            if attempt == 2:
                raise
            print("MNIST 下载失败，重试一次：", e)
    x, y = ds.data, ds.targets.clone()
    if n is not None and n < len(x):
        idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[:n]
        x, y = x[idx], y[idx]
    x = downsample_mnist(x.float() / 255.0, size) if size != 28 else x.float()[:, None] / 255.0
    return x * 2 - 1, y


# ---------------------------------------------------------------------------
# 判定生成数字用的小分类器（第 10 周 SmallCNN 的结构）
# ---------------------------------------------------------------------------
class DigitClassifier(nn.Module):
    """两层 3×3 卷积（1→32→64，各接 ReLU 与 2×2 最大池化）+ 一个全连接层，输入 [−1, 1] 的 MNIST。"""

    def __init__(self, img_size=14):
        super().__init__()
        side = img_size // 2 // 2
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * side * side, 10))
        self.img_size = img_size

    def forward(self, x):
        return self.net(x)


def train_mnist_classifier(size=14, epochs=1, seed=0, device="cpu", verbose=True):
    """在完整 MNIST 训练集（6 万张，缩放到 [−1, 1]）上训练 ``DigitClassifier``，返回 eval 模式的模型。

    默认 14×14、1 个 epoch、Adam 1e-3、batch 128，CPU 上十几秒，测试准确率约 0.98；
    测试准确率存在 ``model.test_accuracy``。用于第 17 周判定"生成的数字被认成了几"。

    **警告**：分类器只在真实数字上训练过，对"四不像"的输入也会给出高置信度——
    纯高斯噪声会被它以很高的置信度判成同一个类别。所以判定结果必须配合阴性对照与类别熵报告
    （见 ``classifier_scores``）。
    """
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    x, y = load_mnist(size=size, train=True)
    xt, yt = load_mnist(size=size, train=False)
    model = DigitClassifier(size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), 128):
            b = perm[i:i + 128]
            loss = F.cross_entropy(model(x[b].to(device)), y[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(xt[i:i + 2000].to(device)).argmax(1).cpu() for i in range(0, len(xt), 2000)])
    model.test_accuracy = float((pred == yt).float().mean())
    if verbose:
        print(f"判定用分类器（{size}×{size}）测试准确率 {model.test_accuracy:.4f}")
    return model


@torch.no_grad()
def classifier_scores(clf, x, targets=None):
    """用分类器给一批图像打分，返回 dict：

    - ``frac_target``：判为目标类别的比例（只有给了 ``targets`` 才有）；
    - ``mean_confidence``：最大 softmax 概率的平均；``frac_confident``：最大概率 > 0.9 的比例；
    - ``class_entropy``：**预测类别直方图**的归一化熵 H/ln 10 ∈ [0, 1]——1 表示 10 类被均匀地预测到，
      0 表示全部判成同一类（纯噪声就是这样）；
    - ``pred_counts``：10 个类别各被预测了几次。

    图像先截断到 [−1, 1]（分类器只见过这个范围）。高置信度**不**说明样本好：先看阴性对照。
    """
    clf.eval()
    dev = next(clf.parameters()).device
    x = torch.as_tensor(x).float().clamp(-1, 1)
    probs = torch.cat([F.softmax(clf(x[i:i + 1000].to(dev)), dim=1).cpu() for i in range(0, len(x), 1000)])
    conf, pred = probs.max(dim=1)
    counts = torch.bincount(pred, minlength=10).double()
    p = counts / counts.sum()
    ent = float(-(p[p > 0] * p[p > 0].log()).sum() / math.log(10))
    out = {"mean_confidence": float(conf.mean()), "frac_confident": float((conf > 0.9).float().mean()),
           "class_entropy": ent, "pred_counts": [int(c) for c in counts]}
    if targets is not None:
        out["frac_target"] = float((pred == torch.as_tensor(targets).cpu()).float().mean())
    return out


@torch.no_grad()
def within_class_diversity(x, labels):
    """同类样本两两之间像素 L2 距离的平均（先在每个类内平均，再对类别平均）。越大越多样。"""
    x = torch.as_tensor(x).float().reshape(len(x), -1).cpu()
    labels = torch.as_tensor(labels).cpu()
    per_class = [torch.pdist(x[labels == c]).mean() for c in labels.unique() if int((labels == c).sum()) > 1]
    return float(torch.stack(per_class).mean())


# ---------------------------------------------------------------------------
# Third-party notices
#
# SmallUNet (and its helper classes above) is adapted from MNISTUNet in
# https://github.com/eje24/iap-diffusion-labs (branch 2025, labs/lab_three.ipynb):
#     Copyright (c) 2025 Ezra Erives and Peter Holderrieth
# sample_checkerboard and sample_spiral are adapted from lib/toy_data.py in
# https://github.com/rtqichen/ffjord:
#     Copyright (c) 2018 Ricky Tian Qi Chen and Will Grathwohl
# Both are distributed under the MIT License:
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.
# ---------------------------------------------------------------------------
