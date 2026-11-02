"""第14周自查函数。用法：from zero2researcher.checks import week14 as ck; ck.count_params(count_trainable)

函数/类练习把函数或类本身传进来（不加括号），例如 ck.lora_init_identity(LoRALinear)；
其余检查直接传入你算出的数。涉及训练随机性的结果按合理区间检查（区间依据本机实测 ± 余量）。

LoRA 的约定照 Hu et al. 2021（arXiv:2106.09685）式 (3) h = W₀x + BAx 及其下文 §4.1 的缩放：
ΔWx 乘以 α/r（α 是与 r 无关的常数），即 h = W₀x + (α/r)·BAx；A ∈ R^{r×d_in} 高斯初始化、B ∈ R^{d_out×r} 零初始化；LoRA 层的构造写法为 LoRALinear(base, r, alpha)，
其中 base 是要包住的（冻结的）nn.Linear，两个低秩矩阵存为 self.A 与 self.B。
数学工作坊 3b/3c 用第 4 周的 California Housing（同一 75/25 划分、random_state=42、按训练集标准化）：
高斯噪声下的最大似然 = 最小二乘（σ² 的最大似然估计 = 训练 MSE）；高斯先验 w ~ N(0, τ²I)、截距不加先验的
MAP = Ridge(alpha=σ²/τ²)。检查函数在内部重算参考解。
torch 只在函数内部 import，保证不装 torch 也能 import 本模块。
"""
import functools
import math

import numpy as np

from . import report, safe_call


def _in_range(name, value, lo, hi, hint=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(value).__name__}")
    if not np.isfinite(v):
        return report(name, False, f"数值不是有限数（{v}）", hint)
    return report(name, lo <= v <= hi, f"期望落在 [{lo}, {hi}]，实际 {v:.4f}", hint)


def _as_int(name, value, what="一个整数"):
    try:
        f = float(value)
    except (TypeError, ValueError):
        report(name, False, f"应传入{what}，实际是 {type(value).__name__}")
        return None
    if not np.isfinite(f) or f != int(f):
        report(name, False, f"应传入{what}，实际是 {value!r}")
        return None
    return int(f)


def _ref_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------- 数可训练参数 ----------
def count_params(fn):
    """fn(model) -> 可训练参数（requires_grad=True）的元素总数。用几个部分冻结的小模型测试。"""
    import torch
    from torch import nn

    name = "count_trainable：数可训练参数"
    torch.manual_seed(0)
    full = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))           # 55 + 12 = 67
    frozen_first = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    for p in frozen_first[0].parameters():
        p.requires_grad = False                                                    # 只剩 12
    emb = nn.Sequential(nn.Embedding(100, 8), nn.Linear(8, 3))
    emb[0].weight.requires_grad = False                                            # 只剩 27
    all_frozen = nn.Linear(4, 4).requires_grad_(False)                             # 0
    cases = [("全部可训练的 10-5-2 MLP", full), ("第一层冻结的 10-5-2 MLP", frozen_first),
             ("冻结 Embedding(100, 8) + Linear(8, 3)", emb), ("全部冻结的 Linear(4, 4)", all_frozen)]
    for desc, model in cases:
        ok, got = safe_call(name, fn, model)
        if not ok:
            return False
        ref = _ref_trainable(model)
        n_tensors = sum(1 for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        try:
            got_i = int(got)
        except (TypeError, ValueError):
            return report(name, False, f"{desc}：应返回一个整数，实际是 {got!r}")
        if got_i != ref:
            hint = "对 model.parameters() 里 requires_grad 为 True 的张量求 p.numel() 之和"
            if got_i == total and total != ref:
                hint = "你把冻结的参数也数进去了：只数 p.requires_grad 为 True 的"
            elif got_i == n_tensors:
                hint = "你数的是张量的个数，不是元素个数：每个张量要用 p.numel()"
            return report(name, False, f"{desc}：期望 {ref:,}，实际 {got_i:,}", hint)
    return report(name, True)


def trainable_params(n, model):
    """n：你数出的可训练参数个数；model：你冻结过部分层的模型。

    检查 n 与模型的实际可训练参数一致，且确实冻结了一部分、分类头仍可训练。"""
    name = "冻结后的可训练参数个数"
    got = _as_int(name, n)
    if got is None:
        return False
    try:
        ref = _ref_trainable(model)
        total = sum(p.numel() for p in model.parameters())
    except AttributeError:
        return report(name, False, "第二个参数应传入模型本身（nn.Module）")
    if got != ref:
        return report(name, False, f"模型实际可训练参数 {ref:,}，你数出 {got:,}",
                      "只数 requires_grad 为 True 的参数，并且数的是元素个数（numel），不是张量个数")
    if ref == total:
        return report(name, False, "模型的参数全部可训练——冻结没有生效",
                      "把要冻结的那些模块里每一个参数的 requires_grad 都设为 False")
    head = getattr(model, "classifier", None)
    if head is not None and not any(p.requires_grad for p in head.parameters()):
        return report(name, False, "分类头 classifier 也被冻结了——新加的头是随机初始化的，必须训练",
                      "冻结的范围是 embeddings 和部分 transformer 层，不包括 pre_classifier / classifier")
    return report(name, True)


# ---------- LoRA ----------
def lora_init_identity(cls):
    """cls(base, r, alpha)：包住冻结的 nn.Linear 的 LoRA 层。

    检查：①初始输出与 base 完全相同（B 零初始化）；②base 被冻结；③可训练的只有
    A (r, d_in) 与 B (d_out, r)，A 不全为 0；④B 非零时输出 = base(x) + (alpha/r)·x Aᵀ Bᵀ；
    ⑤梯度只流向 A、B；⑥支持 (batch, seq, d_in) 形状的输入（DistilBERT 里就是这样）。"""
    import torch
    from torch import nn

    name = "LoRALinear"
    torch.manual_seed(0)
    d_in, d_out, r, alpha = 12, 7, 4, 8
    base = nn.Linear(d_in, d_out)
    ok, layer = safe_call(name, cls, base, r, alpha)
    if not ok:
        return False
    if not isinstance(layer, nn.Module):
        return report(name, False, "LoRALinear 应继承 nn.Module")
    x = torch.randn(5, d_in)
    with torch.no_grad():
        y0 = base(x)
    ok, y = safe_call(name, layer, x)
    if not ok:
        return False
    if tuple(y.shape) != (5, d_out):
        return report(name, False, f"输入 (5, {d_in}) 时输出应为 (5, {d_out})，实际 {tuple(y.shape)}")
    if not torch.allclose(y.detach(), y0, atol=1e-6):
        return report(name, False, "初始化时 LoRA 层的输出应与原来的 Linear 完全相同",
                      "B 要零初始化（torch.zeros）：BA = 0，于是 h = W₀x，模型一开始就是预训练模型本身")
    if any(p.requires_grad for p in base.parameters()):
        return report(name, False, "底层 Linear 的 weight / bias 仍可训练",
                      "在 __init__ 里对 base.parameters() 逐个设 requires_grad = False")
    A, B = getattr(layer, "A", None), getattr(layer, "B", None)
    if not isinstance(A, nn.Parameter) or not isinstance(B, nn.Parameter):
        return report(name, False, "两个低秩矩阵应存为 nn.Parameter：self.A 与 self.B",
                      "self.A = nn.Parameter(...)；只有包成 nn.Parameter，优化器才找得到它们")
    if tuple(A.shape) != (r, d_in) or tuple(B.shape) != (d_out, r):
        return report(name, False,
                      f"形状应为 A: ({r}, {d_in})、B: ({d_out}, {r})；实际 A {tuple(A.shape)}、B {tuple(B.shape)}",
                      "照论文写法 A ∈ R^{r×d_in}、B ∈ R^{d_out×r}，ΔW = BA 的形状才和 W₀ (d_out, d_in) 一样")
    n_train = _ref_trainable(layer)
    if n_train != r * (d_in + d_out):
        return report(name, False, f"可训练参数应只有 A、B 共 {r * (d_in + d_out)} 个，实际 {n_train}")
    if torch.count_nonzero(A.detach()) == 0:
        return report(name, False, "A 全是 0：B 的梯度 ∝ A x 也会是 0，LoRA 永远学不动",
                      "A 用高斯随机初始化（torch.randn），只有 B 零初始化")
    with torch.no_grad():
        B.copy_(torch.randn_like(B))
        y1 = layer(x)
        expected = y0 + (alpha / r) * (x @ A.T @ B.T)
    if not torch.allclose(y1, expected, atol=1e-5):
        return report(name, False, "B 非零时输出与 h = W₀x + (alpha/r)·BAx 对不上",
                      "前向 = 原层输出 + 低秩支路；检查三件事：缩放是不是 alpha/r（不是 alpha、也不是 r/alpha），"
                      "低秩支路里先乘 A 再乘 B，以及按 x 是“行向量”的约定该用哪个转置")
    layer.zero_grad()
    layer(x).sum().backward()
    if A.grad is None or B.grad is None:
        return report(name, False, "反向传播之后 A 或 B 没有梯度")
    if base.weight.grad is not None:
        return report(name, False, "底层 Linear 的 weight 拿到了梯度——它应该是冻结的")
    x3 = torch.randn(2, 3, d_in)
    ok, y3 = safe_call(name, layer, x3)
    if not ok:
        return False
    if tuple(y3.shape) != (2, 3, d_out):
        return report(name, False, f"输入 (2, 3, {d_in}) 时输出应为 (2, 3, {d_out})，实际 {tuple(y3.shape)}",
                      "让低秩支路作用在输入的最后一维（用 @ 对最后一维做矩阵乘法即可），不要假设输入是二维的")
    return report(name, True)


def lora_param_count(n, r, d_in, d_out):
    """一层 LoRA（d_in → d_out，秩 r）的参数个数应为 r·(d_in + d_out)。"""
    name = f"LoRA 参数个数（r={r}，{d_in}→{d_out}）"
    got = _as_int(name, n)
    if got is None:
        return False
    ref = r * (d_in + d_out)
    hint = "A 是 r×d_in、B 是 d_out×r，没有偏置：r·d_in + d_out·r"
    if got == d_in * d_out:
        hint = "这是完整 ΔW（d_out×d_in）的参数量；LoRA 只训练两个低秩因子 A 和 B"
    return report(name, got == ref, f"期望 {ref:,}，实际 {got:,}", hint)


# ---------- 数学工作坊 3：高斯 KL ----------
def _ref_gauss_kl(m1, s1, m2, s2):
    return math.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5


def gaussian_kl(fn):
    """fn(mu1, sigma1, mu2, sigma2) -> KL(N(mu1, sigma1²) ‖ N(mu2, sigma2²))。注意传入的是标准差。"""
    name = "两个一维高斯的解析 KL"
    cases = [(0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0), (0.0, 2.0, 0.0, 1.0),
             (0.0, 1.0, 0.0, 2.0), (0.5, 0.3, -1.0, 1.5), (2.0, 0.5, 0.0, 1.0)]
    for m1, s1, m2, s2 in cases:
        ok, got = safe_call(name, fn, m1, s1, m2, s2)
        if not ok:
            return False
        try:
            g = float(got)
        except (TypeError, ValueError):
            return report(name, False, f"应返回一个数，实际 {got!r}")
        ref = _ref_gauss_kl(m1, s1, m2, s2)
        if not math.isclose(g, ref, rel_tol=1e-6, abs_tol=1e-6):
            hint = "log(σ₂/σ₁) + (σ₁² + (μ₁−μ₂)²) / (2σ₂²) − 1/2"
            swapped = _ref_gauss_kl(m2, s2, m1, s1)
            as_var = _ref_gauss_kl(m1, math.sqrt(s1), m2, math.sqrt(s2))
            if math.isclose(g, swapped, rel_tol=1e-6, abs_tol=1e-6):
                hint = "你算的是 KL(q‖p)：KL 不对称，第一个分布是 p = N(μ₁, σ₁²)"
            elif math.isclose(g, as_var, rel_tol=1e-6, abs_tol=1e-6):
                hint = "传入的 sigma 是标准差，不是方差：公式里要用 sigma**2"
            return report(name, False,
                          f"KL(N({m1}, {s1}²) ‖ N({m2}, {s2}²)) 应为 {ref:.6f}，实际 {g:.6f}", hint)
    return report(name, True)


def mc_kl(estimate, exact):
    """蒙特卡洛估计与解析值之差应在容差内：|估计 − 解析| ≤ 0.02 + 3%·解析。"""
    name = "KL 的蒙特卡洛估计"
    try:
        est, ex = float(estimate), float(exact)
    except (TypeError, ValueError):
        return report(name, False, "应传入两个数：蒙特卡洛估计值、解析值")
    if not (np.isfinite(est) and np.isfinite(ex)):
        return report(name, False, f"数值不是有限数（估计 {est}，解析 {ex}）")
    tol = 0.02 + 0.03 * abs(ex)
    hint = ("从 p 采样 x，对 log p(x) − log q(x) 求平均；样本数至少几万。"
            "若估计值是负数或偏差很大，检查是不是从 q 采样了，或者把 log 密度的 −log σ 项漏了")
    return report(name, abs(est - ex) <= tol,
                  f"估计 {est:.4f} 与解析值 {ex:.4f} 相差 {abs(est - ex):.4f}，超过容差 {tol:.4f}", hint)


# ---------- 数学工作坊 3b/3c：MSE = 高斯负对数似然；Ridge = 高斯先验的 MAP ----------
@functools.lru_cache(maxsize=1)
def _calhousing_std():
    """与讲义 3b 相同（也与第 4 周 S4 相同）：75/25 划分（random_state=42），按训练集的均值与标准差（ddof=0）标准化。
    返回 (X_train_std, y_train)，两个数组只读。"""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    housing = fetch_california_housing(as_frame=True)
    X_train, _, y_train, _ = train_test_split(housing.data, housing.target, test_size=0.25, random_state=42)
    X_raw, y = X_train.to_numpy(), y_train.to_numpy()
    X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
    X.flags.writeable = False
    y.flags.writeable = False
    return X, y


def _as_wb_array(w, b, name, d=8):
    """把 (w, b) 转成 (形状 (d,) 的数组, 浮点数)；失败时打印提示并返回 None。"""
    try:
        w = np.ravel(np.asarray(w, dtype=float))
        b = float(np.ravel(np.asarray(b, dtype=float))[0])
    except (TypeError, ValueError, IndexError):
        report(name, False, "w 应是 8 个系数组成的数组，b 应是一个数")
        return None
    if w.shape != (d,):
        report(name, False, f"w 应有 {d} 个系数（每个特征一个），实际 {w.size} 个",
               "theta 的前 8 个是 w、最后一个是 b（3b 里 theta 还多一个 log σ）")
        return None
    if not (np.all(np.isfinite(w)) and np.isfinite(b)):
        report(name, False, "w 或 b 里有 NaN / inf", "检查 minimize 有没有收敛（打印 res.success 与 res.message）")
        return None
    return w, b


def gaussian_nll(fn):
    """fn(y, y_hat, sigma) -> y_i ~ N(y_hat_i, σ²) 时每个样本的负对数似然，**对样本取平均**。

    与 −scipy.stats.norm.logpdf(y, y_hat, σ).mean() 比较（σ ∈ {1, 0.5, 2}，相对误差 1e-9）。
    讲义 3b 已经写出推导和公式，这是照公式写代码的引导式练习（仿射关系的验证在答案块之后的下一格）；
    失败时只给方向，不打印期望值。识别：求和而不是平均、漏了 ½·log(2πσ²)、把 σ 当成方差、符号反了。"""
    from scipy.stats import norm

    name = "gaussian_nll：高斯负对数似然"
    rng = np.random.default_rng(9)
    y, y_hat = rng.normal(size=50), rng.normal(size=50)
    for s in (1.0, 0.5, 2.0):
        ok, got = safe_call(name, fn, y.copy(), y_hat.copy(), s)
        if not ok:
            return False
        try:
            g = float(got)
        except (TypeError, ValueError):
            return report(name, False, f"应返回一个数（对样本取平均），实际 {type(got).__name__}，形状 {np.shape(got)}",
                          "最后对所有样本取 np.mean")
        want = -norm.logpdf(y, loc=y_hat, scale=s).mean()
        if math.isclose(g, want, rel_tol=1e-9, abs_tol=1e-12):
            continue
        sq = np.mean((y - y_hat) ** 2)
        hint = "每个样本：½·log(2πσ²) + (y − ŷ)²/(2σ²)，再对样本取平均"
        if math.isclose(g, want * len(y), rel_tol=1e-9):
            hint = "你返回的是求和；本节约定对样本取平均（与 MSE 的口径一致）"
        elif math.isclose(g, sq / (2 * s ** 2), rel_tol=1e-9):
            hint = "漏了 ½·log(2πσ²) 这一项：σ 固定时它是常数，不影响 w、b 的最优解，但要一起估计 σ 时少不了它"
        elif math.isclose(g, 0.5 * math.log(2 * math.pi * s) + sq / (2 * s), rel_tol=1e-9):
            hint = "传入的 sigma 是标准差，不是方差：公式里要用 sigma**2"
        elif math.isclose(g, -want, rel_tol=1e-9):
            hint = "符号反了：这是负对数似然，−log p，数值越小越好"
        return report(name, False, f"σ = {s} 时与 −log N(y; ŷ, σ²) 的样本平均不一致", hint)
    return report(name, True)


def mle_matches_ols(w, b, sigma2):
    """讲义 3b：数值最大化高斯似然得到的 (w, b, σ²)，应与标准化训练集上的 LinearRegression 一致
    （w、b 的 atol 1e-5；本机差 1e-7），σ² 应等于训练集 MSE（相对误差 1e-5；本机 0.520552 vs 0.520552）。"""
    from sklearn.linear_model import LinearRegression
    name = "最大似然 = 最小二乘（CalHousing）"
    wb = _as_wb_array(w, b, name)
    if wb is None:
        return False
    w, b = wb
    try:
        s2 = float(sigma2)
    except (TypeError, ValueError):
        return report(name, False, "sigma2 应是一个数（σ² 的最大似然估计）")
    X, y = _calhousing_std()
    ols = LinearRegression().fit(X, y)
    gap = max(np.abs(w - ols.coef_).max(), abs(b - ols.intercept_))
    if gap > 1e-5:
        hint = ("在标准化后的训练集 X_h、y_h 上最小化 gaussian_nll（参数是 w、b、log σ）；"
                "若 minimize 提前停了，把 gtol 调小、maxiter 调大")
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        housing = fetch_california_housing(as_frame=True)
        X_train, _, y_train, _ = train_test_split(housing.data, housing.target, test_size=0.25, random_state=42)
        if np.allclose(w, LinearRegression().fit(X_train, y_train).coef_, atol=1e-4):
            hint = "这是在原始（未标准化）特征上的系数；本节用按训练集均值与标准差标准化后的 X_h"
        return report(name, False, f"与 LinearRegression 的系数（含截距）最大差 {gap:.1e}，应 < 1e-5", hint)
    mse = float(np.mean((y - ols.predict(X)) ** 2))
    if not math.isclose(s2, mse, rel_tol=1e-5):
        hint = "σ² 的最大似然估计就是训练集上的 MSE：对 σ² 求导令其为 0 即得"
        if math.isclose(s2, math.sqrt(mse), rel_tol=1e-4):
            hint = "你传入的是 σ（标准差）；这里要 σ² = exp(2·log σ)"
        elif math.isclose(s2, math.log(math.sqrt(mse)), rel_tol=1e-4) or math.isclose(s2, math.log(mse), rel_tol=1e-4):
            hint = "你传入的是 log σ（优化用的参数）；要换回 σ² = exp(2·log σ)"
        return report(name, False, f"σ² = {s2:.6f}，与训练集 MSE 不一致（相对误差 {abs(s2 / mse - 1):.1e}）", hint)
    return report(name, True)


def map_equals_ridge(w, b, alpha):
    """讲义 3c：高斯先验 w ~ N(0, τ²I)（截距不加先验）、噪声方差 σ² 下的 MAP (w, b)，应与 sklearn
    Ridge(alpha=σ²/τ²) 在标准化训练集上的解一致（atol 1e-5；本机差 ≤ 2e-7）。alpha 是你算出的 σ²/τ²。"""
    from sklearn.linear_model import LinearRegression, Ridge
    name = "MAP（高斯先验）= Ridge"
    wb = _as_wb_array(w, b, name)
    if wb is None:
        return False
    w, b = wb
    try:
        a = float(alpha)
    except (TypeError, ValueError):
        return report(name, False, "alpha 应是一个数：σ² / τ²")
    if not (np.isfinite(a) and a > 0):
        return report(name, False, f"alpha 应是正数，实际 {a}")
    X, y = _calhousing_std()
    n = len(y)

    def gap_to(alpha_):
        ref = Ridge(alpha=alpha_).fit(X, y)
        return max(np.abs(w - ref.coef_).max(), abs(b - ref.intercept_))

    gap = gap_to(a)
    if gap <= 1e-5:
        return report(name, True)
    hint = "负对数后验 = SSE/(2σ²) + ‖w‖²/(2τ²)；两边乘 2σ² 就是 Ridge，α = σ²/τ²，截距不加先验"
    ols = LinearRegression().fit(X, y)
    s2 = float(np.mean((y - ols.predict(X)) ** 2))       # 讲义里的 σ²（最大似然估计 = 训练 MSE）
    if gap_to(1 / a) <= 1e-5:
        hint = "α 写反了：α = σ²/τ²（先验越窄、惩罚越强），不是 τ²/σ²"
    elif gap_to(a / s2) <= 1e-5:                     # 等效 α = 1/τ²：SSE/2 + ‖w‖²/(2τ²)，或 SSE/(2σ²) + ‖w‖²/(2τ²σ²)
        hint = "似然项少除了 σ²（或先验项多除了 σ²）：负对数后验 = SSE/(2σ²) + ‖w‖²/(2τ²)"
    elif gap_to(a / (2 * s2)) <= 1e-5:               # 等效 α = 1/(2τ²)：SSE + ‖w‖²/(2τ²)
        hint = "似然项要除以 2σ²：负对数后验 = SSE/(2σ²) + ‖w‖²/(2τ²)"
    elif gap_to(a * n) <= 1e-5:
        hint = "只给似然项除了 n、先验项没除：要么都除以 n，要么都不除——最优解只取决于两项的比例"
    else:
        A = np.column_stack([X, np.ones(n)])
        theta = np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]), A.T @ y)
        if max(np.abs(w - theta[:-1]).max(), abs(b - theta[-1])) <= 1e-5:
            hint = "截距也加了先验：只有 w 有高斯先验，截距 b 不进 ‖w‖²"
    return report(name, False, f"与 Ridge(alpha={a:.4g}) 的解（含截距）最大差 {gap:.1e}，应 ≤ 1e-5", hint)


# ---------- 微调与 PEFT ----------
FINETUNE_BANDS = {
    # 缩小配置（train ≤ 500、max_len ≤ 128、≥ 3 epochs；不论跑在 CPU 还是 GPU 上）：本机 CPU 实测
    # 500 / 128 / 3 epochs 0.766–0.842（4 个种子）、300 / 96 / 3 epochs 0.768–0.786
    "cpu": (0.68, 0.90),
    # Colab T4 正式配置（train 5000 / max_len 256 / 2 epochs / fp16）：尚未实测，按调研预期 0.88–0.91
    # 与公开模型卡（全量 25k：0.907–0.931）留余量；发布前在 T4 上实测后收紧
    "t4": (0.85, 0.95),
}


def finetune_accuracy(acc, config="cpu"):
    """DistilBERT 全量微调的测试准确率。config="cpu"：缩小配置（讲义与作业的 CPU 配置）；
    config="t4"：作业的 Colab T4 正式配置（5000 / 256 / 2 epochs）。两种配置分开设区间。"""
    if config not in FINETUNE_BANDS:
        return report("DistilBERT 全量微调测试准确率", False,
                      f"config 应是 {sorted(FINETUNE_BANDS)} 之一，实际 {config!r}")
    lo, hi = FINETUNE_BANDS[config]
    label = "CPU 缩小配置" if config == "cpu" else "T4 正式配置"
    return _in_range(f"DistilBERT 全量微调测试准确率（{label}）", acc, lo, hi,
                     "本机 CPU 实测：train 500 / max_len 128 / 3 epochs / AdamW 2e-5 约 0.77–0.84，"
                     "train 300 / max_len 96 约 0.77–0.79（测试集前 500 条）；T4 正式配置预期 0.88 以上（尚未实测）。"
                     "≈0.50 说明没学动：检查训练步数够不够（缩小配置下 1 个 epoch 只有几十步）、学习率是否过小、"
                     "是否每步都清零梯度、评估时是否处于 eval 模式")


def scratch_lora_accuracy(acc):
    """自己写的 LoRALinear 装进 DistilBERT（r=4，CPU 小配置 train 300 / max_len 96 / 3 epochs）后的测试准确率。"""
    return _in_range("从零实现的 LoRA 微调测试准确率", acc, 0.62, 0.92,
                     "本机 CPU 实测 r=4、alpha=8、AdamW 1e-3、3 epochs 约 0.74（PEFT 同配置约 0.78）。"
                     "≈0.50 说明没学动：检查 add_lora 是否重新打开了分类头的梯度、优化器拿到的是不是 requires_grad 的参数、"
                     "q_lin / v_lin 是否真的被替换成了你的 LoRALinear")


METHODS = ("probe", "full", "lora4", "lora16")


def adaptation_experiment(results, n_seeds=None):
    """研究实验 3 的结果结构：results[method] = {"accs": [...], "seconds": [...],
    "trainable": int, "head": int, "lora": int}，method ∈ {"probe", "full", "lora4", "lora16"}。

    检查结构与参数拆账的一致性：LoRA 两行 head + lora == trainable，且 lora 部分 r=16 是 r=4 的 4 倍；
    探针只训头、全量微调的可训练参数最多。n_seeds 给定时检查每种方法的种子数。"""
    name = "研究实验 3 结果结构"
    if not isinstance(results, dict):
        return report(name, False, f"results 应是 dict，实际是 {type(results).__name__}")
    if set(results) != set(METHODS):
        return report(name, False, f"键应恰好是 {METHODS}，实际 {sorted(results)}")
    for m in METHODS:
        v = results[m]
        need = {"accs", "seconds", "trainable", "head", "lora"}
        if not isinstance(v, dict) or not need <= set(v):
            return report(name, False, f"results['{m}'] 应是含 {sorted(need)} 的 dict")
        accs, secs = list(v["accs"]), list(v["seconds"])
        if not accs or len(accs) != len(secs):
            return report(name, False, f"results['{m}'] 的 accs 与 seconds 应一一对应（每个种子一个），实际 {len(accs)} / {len(secs)}")
        if n_seeds is not None and len(accs) != n_seeds:
            return report(name, False, f"results['{m}'] 应有 {n_seeds} 个种子的结果，实际 {len(accs)} 个")
        if not all(np.isfinite(float(a)) and 0.0 <= float(a) <= 1.0 for a in accs):
            return report(name, False, f"results['{m}']['accs'] 里有不在 [0, 1] 的值：{accs}")
        if not all(np.isfinite(float(s)) and float(s) > 0 for s in secs):
            return report(name, False, f"results['{m}']['seconds'] 应是正的训练用时（秒）：{secs}")
    t = {m: (int(results[m]["trainable"]), int(results[m]["head"]), int(results[m]["lora"])) for m in METHODS}
    for m in ("lora4", "lora16"):
        if t[m][1] + t[m][2] != t[m][0]:
            return report(name, False, f"{m}：head + lora = {t[m][1] + t[m][2]:,}，与 trainable = {t[m][0]:,} 对不上",
                          "PEFT 的 SEQ_CLS 把分类头也设为可训练；按参数名里是否含 'lora_' 拆账")
    if t["lora16"][2] != 4 * t["lora4"][2]:
        return report(name, False, f"LoRA A/B 参数应与 r 成正比：r=16 的 {t['lora16'][2]:,} 应是 r=4 的 {t['lora4'][2]:,} 的 4 倍")
    if t["probe"][2] != 0 or t["probe"][0] != t["probe"][1]:
        return report(name, False, f"探针只训练分类头：trainable 应等于 head、lora 应为 0，实际 {t['probe']}")
    if not t["full"][0] > max(t[m][0] for m in ("probe", "lora4", "lora16")):
        return report(name, False, "全量微调的可训练参数应最多（约 6700 万）")
    return report(name, True)


HEAD_MODULE_NAMES = ("classifier", "score")   # PEFT 对 SEQ_CLS 默认放进 modules_to_save 的分类头名字（按后缀匹配）


def _is_head_param(name):
    parts = name.split(".")
    return any(p.endswith(h) for p in parts for h in HEAD_MODULE_NAMES)


def peft_accounting(total, head, ab, model=None):
    """PEFT 可训练参数拆账：total（print_trainable_parameters 报的 trainable 数）= head（分类头）+ ab（LoRA A/B）。

    必须同时传入 PEFT 模型本身：自查会独立地从 model.named_parameters() 重新数一遍——
    名字含 "lora_" 且可训练的算 A/B，名字里有分类头模块名（classifier / pre_classifier / score）且可训练的算头，
    total 与 PEFT 的 get_nb_trainable_parameters() 比对——再和你的三个数逐项核对。"""
    name = "LoRA 可训练参数拆账"
    vals = []
    for label, v in (("total", total), ("head", head), ("ab", ab)):
        iv = _as_int(name, v, f"整数（{label}）")
        if iv is None:
            return False
        vals.append(iv)
    t, h, a = vals
    if model is None:
        return report(name, False, "还要把 PEFT 模型本身传进来，自查才能独立重数一遍",
                      "ck.peft_accounting(total, head, ab, peft_model)")
    try:
        named = list(model.named_parameters())
    except AttributeError:
        return report(name, False, "第四个参数应是 PEFT 包装后的模型（nn.Module）")
    ref_ab = sum(p.numel() for n, p in named if p.requires_grad and "lora_" in n)
    ref_head = sum(p.numel() for n, p in named if p.requires_grad and "lora_" not in n and _is_head_param(n))
    ref_total = sum(p.numel() for n, p in named if p.requires_grad)
    if hasattr(model, "get_nb_trainable_parameters"):          # print_trainable_parameters 打印的就是它
        ref_total = int(model.get_nb_trainable_parameters()[0])
    all_params = sum(p.numel() for _, p in named)
    if ref_head + ref_ab != ref_total:
        return report(name, False,
                      f"模型里除了 LoRA A/B（{ref_ab:,}）和分类头（{ref_head:,}）还有别的可训练参数，合计 {ref_total:,}",
                      "PEFT 包装后，身体应全部冻结；检查是否在 get_peft_model 之后又手动打开了别的参数")
    if t != ref_total:
        hint = "total 应是 print_trainable_parameters 报的 trainable 数：只数 requires_grad 为 True 的参数"
        if t == all_params or t >= 10 * ref_total:
            hint = "你把冻结的参数也数进去了：拆账只看 requires_grad 为 True 的参数"
        return report(name, False, f"total = {t:,}，但这个模型的可训练参数是 {ref_total:,}", hint)
    if a != ref_ab:
        return report(name, False, f"ab = {a:,}，但名字含 'lora_' 的可训练参数是 {ref_ab:,}",
                      "按参数名分组：带 'lora_' 的（lora_A、lora_B）算 A/B")
    if h != ref_head:
        hint = ("task_type=SEQ_CLS 时 PEFT 会把分类头（pre_classifier、classifier）放进 modules_to_save 一起训练；"
                "head 只算这些可训练的分类头参数")
        if h > ref_head:
            hint = "head 数多了：别把冻结的原始分类头副本（original_module）或身体参数算进去，只数 requires_grad 为 True 的"
        return report(name, False, f"head = {h:,}，但可训练的分类头参数是 {ref_head:,}", hint)
    return report(name, True)
