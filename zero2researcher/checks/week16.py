"""第16周自查函数。用法：from zero2researcher.checks import week16 as ck; ck.euler_integrate(euler_integrate)

函数练习把函数本身传进来（不加括号），例如 ck.fm_loss(fm_loss)；其余检查直接传入你算出的结果。

本周统一约定（MIT 6.S184 讲义 §3.3 Algorithm 3，Holderrieth & Erives, arXiv:2506.02070）：
x0 是噪声、x1 是数据，线性路径 x_t = (1 − t)·x0 + t·x1，**t = 0 是噪声、t = 1 是数据**；
条件速度目标 x1 − x0；损失是网络输出与目标之间的均方误差。
欧拉法（讲义 §2.1 Algorithm 1）：x_{t+h} = x_t + h·u(x_t, t)，每一步用**这一步起点**的时间。
训练结果按合理区间检查，区间依据本机实测 ± 余量。torch 只在函数内部 import。
"""
import math

import numpy as np

from . import report, safe_call

STEPS = (2, 5, 20, 100)
N_SEEDS = 3


def _in_range(name, value, lo, hi, hint=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(value).__name__}")
    if not np.isfinite(v):
        return report(name, False, f"数值不是有限数（{v}），训练或采样大概率发散了", hint)
    return report(name, lo <= v <= hi, f"期望落在 [{lo}, {hi}]，实际 {v:.4f}", hint)


# ---------- 欧拉法 ----------
def euler_integrate(fn):
    """fn(u, x0, n_steps, t_start=0.0, t_end=1.0) -> 轨迹张量，形状 (n_steps + 1, *x0.shape)。

    u(x, t) 是速度场（t 是一个数）；轨迹第 0 行是 x0，最后一行是终点。用三个有解析解的速度场检查。"""
    import torch

    name = "欧拉法 euler_integrate"
    torch.manual_seed(0)
    x0 = torch.randn(50, 2)
    c = torch.tensor([1.0, -2.0])
    ok, path = safe_call(name, fn, lambda x, t: c.expand_as(x), x0, 10)
    if not ok:
        return False
    if not isinstance(path, torch.Tensor) or tuple(path.shape) != (11, 50, 2):
        return report(name, False, f"x0 形状 (50, 2)、n_steps=10 时应返回 (11, 50, 2) 的轨迹，实际 "
                      f"{getattr(path, 'shape', type(path).__name__)}",
                      "把起点和每一步之后的 x 都存进列表，最后 torch.stack(path)：共 n_steps + 1 个状态")
    if not torch.allclose(path[0], x0):
        return report(name, False, "轨迹的第 0 行应该就是起点 x0")
    if not torch.allclose(path[-1], x0 + c, atol=1e-5):
        return report(name, False, "常速度场 u = (1, −2) 从 t=0 走到 t=1，终点应是 x0 + (1, −2)",
                      "步长 h = (t_end − t_start) / n_steps；每步 x ← x + h·u(x, t)")
    # 线性衰减 u = −x：欧拉法的精确结果是 x0·(1 − h)^n，n 很大时趋近解析解 x0·e^{−1}
    ok, path = safe_call(name, fn, lambda x, t: -x, x0, 10)
    if not ok:
        return False
    if not torch.allclose(path[-1], x0 * 0.9 ** 10, atol=1e-5):
        return report(name, False, "u = −x、10 步时，欧拉法的终点应恰好是 x0·0.9^10",
                      "每一步都要用**当前的** x 计算速度，而不是一直用 x0")
    ok, path = safe_call(name, fn, lambda x, t: -x, x0, 1000)
    if not ok:
        return False
    if not torch.allclose(path[-1], x0 * math.exp(-1), atol=1e-3):
        return report(name, False, "u = −x、1000 步时终点应非常接近解析解 x0·e^{−1}")
    # 依赖时间的场 u = (t, 0)：用每一步起点的时间，4 步的结果是 x0 + (0.375, 0)
    ok, path = safe_call(name, fn, lambda x, t: torch.stack([torch.full_like(x[:, 0], float(t)),
                                                           torch.zeros_like(x[:, 1])], dim=1), x0, 4)
    if not ok:
        return False
    got = float((path[-1] - x0)[:, 0].mean())
    if not math.isclose(got, 0.375, abs_tol=1e-5):
        hint = "第 i 步（i = 0, 1, …）的时间是 t_start + i·h"
        if math.isclose(got, 0.625, abs_tol=1e-5):
            hint = "你用了每一步**终点**的时间 t + h；欧拉法用起点的时间 t_start + i·h（第一步 t = 0）"
        elif math.isclose(got, 0.5, abs_tol=1e-5):
            hint = "结果等于解析解 0.5，说明你用了更精确的格式（如中点法）；本周要求的是朴素欧拉法"
        return report(name, False, f"u = (t, 0)、4 步时 x 坐标应增加 0.375（欧拉法），实际 {got:.4f}", hint)
    ok, path = safe_call(name, fn, lambda x, t: c.expand_as(x), x0, 5, 0.2, 0.7)
    if not ok:
        return False
    if not torch.allclose(path[-1], x0 + 0.5 * c, atol=1e-5):
        return report(name, False, "从 t_start=0.2 走到 t_end=0.7，常速度场的位移应是 0.5·u",
                      "h = (t_end − t_start) / n_steps，时间从 t_start 开始")
    return report(name, True)


# ---------- 线性插值与条件速度目标 ----------
def interpolant(fn):
    """fn(x0, x1, t) -> x_t = (1 − t)·x0 + t·x1（x0 噪声，x1 数据；t 形状 (B, 1)，可广播）。"""
    import torch

    name = "线性插值 interpolant"
    torch.manual_seed(0)
    x0, x1 = torch.randn(64, 2), torch.randn(64, 2) * 3 + 5
    for tv in (0.0, 1.0):
        t = torch.full((64, 1), tv)
        ok, xt = safe_call(name, fn, x0, x1, t)
        if not ok:
            return False
        if not isinstance(xt, torch.Tensor) or tuple(xt.shape) != (64, 2):
            return report(name, False, f"应返回与 x0 同形状 (64, 2) 的张量，实际 {getattr(xt, 'shape', xt)!r}")
        want = x0 if tv == 0.0 else x1
        if not torch.allclose(xt, want, atol=1e-6):
            hint = "x_t = (1 − t)·x0 + t·x1"
            if torch.allclose(xt, x1 if tv == 0.0 else x0, atol=1e-6):
                hint = "时间方向反了：本课程（MIT 讲义）约定 t = 0 是噪声 x0、t = 1 是数据 x1"
            return report(name, False, f"t = {tv:g} 时 x_t 应等于 {'噪声 x0' if tv == 0 else '数据 x1'}", hint)
    t = torch.rand(64, 1)
    ok, xt = safe_call(name, fn, x0, x1, t)
    if not ok:
        return False
    if not torch.allclose(xt, (1 - t) * x0 + t * x1, atol=1e-5):
        return report(name, False, "中间时刻的插值不对", "每一行用自己的 t：t 的形状 (B, 1) 会自动广播到 (B, 2)")
    return report(name, True)


def fm_target(fn):
    """fn(x0, x1) -> 条件速度目标 x1 − x0（沿直线从噪声 x0 走到数据 x1 的速度，与 t 无关）。"""
    import torch

    name = "条件速度目标 fm_target"
    torch.manual_seed(0)
    x0, x1 = torch.randn(64, 2), torch.randn(64, 2) * 3 + 5
    ok, u = safe_call(name, fn, x0, x1)
    if not ok:
        return False
    if not isinstance(u, torch.Tensor) or tuple(u.shape) != (64, 2):
        return report(name, False, f"应返回 (64, 2) 的张量，实际 {getattr(u, 'shape', u)!r}")
    if torch.allclose(u, x0 - x1, atol=1e-6):
        return report(name, False, "符号反了：粒子从噪声 x0 走向数据 x1，速度是 x1 − x0")
    return report(name, torch.allclose(u, x1 - x0, atol=1e-6), "目标应是 x1 − x0",
                  "x_t = (1 − t)·x0 + t·x1 对 t 求导，就是 x1 − x0")


# ---------- Flow Matching 损失 ----------
class _Spy:
    """记录被调用时收到的 (x, t)，输出全 0（带梯度）。"""

    def __init__(self):
        import torch
        self.w = torch.zeros((), requires_grad=True)
        self.calls = []

    def __call__(self, x, t):
        self.calls.append((x.detach().clone(), t.detach().clone() if hasattr(t, "detach") else t))
        return x * 0 + self.w


def fm_loss(fn):
    """fn(model, x1) -> 标量损失：内部采样噪声 x0 与 t ~ U(0, 1)（形状 (B, 1)），插值得 x_t，
    返回 mean((model(x_t, t) − (x1 − x0))²)。

    检查 ①（解析解）：数据只有一个点 c 时，最优速度场是 u*(x, t) = (c − x)/(1 − t)，损失应 ≈ 0；
    ② 全零模型的损失 ≈ mean(x1²) + 1；③ t 的形状与分布；④ 每次调用重新采样噪声；⑤ 可求导。"""
    import torch

    name = "Flow Matching 损失 fm_loss"
    torch.manual_seed(0)
    B = 20000
    c = torch.tensor([2.0, -1.0])
    x1 = c.repeat(B, 1)

    def oracle(x, t):
        return (c - x) / (1 - t)

    spy = _Spy()
    ok, loss = safe_call(name, fn, spy, x1)
    if not ok:
        return False
    if not spy.calls:
        return report(name, False, "fm_loss 没有调用 model(x_t, t)")
    _, t_seen = spy.calls[-1]
    if not isinstance(t_seen, torch.Tensor) or t_seen.numel() != B:
        return report(name, False, f"传给模型的 t 应是每个样本一个的张量（{B} 个数），实际 "
                      f"{getattr(t_seen, 'shape', t_seen)!r}", "t = torch.rand(B, 1)")
    if tuple(t_seen.shape) != (B, 1):
        return report(name, False, f"2D 数据上 t 的形状应是 (B, 1)，实际 {tuple(t_seen.shape)}",
                      "(B, 1) 才能和 (B, 2) 的 x0、x1 广播相乘；写成 torch.rand(B, 1)")
    tm, tv = float(t_seen.mean()), float(t_seen.var())
    if not (t_seen.min() >= 0 and t_seen.max() <= 1 and abs(tm - 0.5) < 0.02 and abs(tv - 1 / 12) < 0.01):
        return report(name, False, f"t 应服从 [0, 1] 上的均匀分布（均值 0.5、方差 1/12），实际均值 {tm:.3f}、方差 {tv:.4f}",
                      "用 torch.rand（均匀分布），不是 torch.randn（正态分布）")
    try:
        lv = float(loss.detach() if isinstance(loss, torch.Tensor) else loss)
    except (TypeError, ValueError, RuntimeError):
        return report(name, False, f"应返回一个标量，实际形状 {getattr(loss, 'shape', None)}", "最后 .mean()")
    expected_zero = float((c ** 2).mean()) + 1.0
    if not math.isclose(lv, expected_zero, rel_tol=0.05):
        hint = "损失 = ((model(x_t, t) − target) ** 2).mean()——对所有元素取平均"
        if lv > 100 * expected_zero:
            hint = "数值太大：你可能用了 .sum()；课程统一用 .mean()"
        elif math.isclose(lv, expected_zero / 3, rel_tol=0.05):
            # 目标写成 x1 − x_t = (1 − t)(x1 − x0)（或 x_t − x0 = t(x1 − x0)）：全零模型的损失 ≈ E[(1 − t)²] = 1/3 倍
            hint = ("目标写错了：应是 x1 − x0（与 t 无关），不是 x1 − x_t。x1 − x_t = (1 − t)·(x1 − x0)，"
                    "所以全零模型的损失只有正确值的约 1/3")
        return report(name, False, f"全零模型的损失应 ≈ mean(x1²) + 1 = {expected_zero:.2f}，实际 {lv:.3f}", hint)
    ok, loss2 = safe_call(name, fn, spy, x1)
    if not ok:
        return False
    if float(loss2.detach() if isinstance(loss2, torch.Tensor) else loss2) == lv:
        return report(name, False, "两次调用的损失完全相同：每次都应重新采样噪声 x0 与 t",
                      "x0 = torch.randn_like(x1)、t = torch.rand(B, 1) 写在函数里面")
    if not (isinstance(loss, torch.Tensor) and loss.requires_grad):
        return report(name, False, "损失不能对模型参数求导", "返回 torch 张量；不要 .item() 或转成 numpy")
    loss.backward()
    if spy.w.grad is None:
        return report(name, False, "损失不能对模型参数求导", "不要 .item() 或转成 numpy；保持 torch 运算")
    ok, loss_o = safe_call(name, fn, oracle, x1)
    if not ok:
        return False
    if not float(loss_o) < 1e-3:
        return report(name, False, f"数据只有一个点 c 时，最优速度场 (c − x)/(1 − t) 的损失应 ≈ 0，实际 {float(loss_o):.4f}",
                      "插值与目标要配套：x_t = (1 − t)·x0 + t·x1，目标 x1 − x0；t = 0 是噪声。"
                      "检查是否把 x0、x1 的位置写反了，或者模型收到的不是 x_t")
    return report(name, True)


def train_step(fn):
    """fn(model, optimizer, x1) -> 这一步的损失（float）：清零梯度 → fm_loss → 反向传播 → 更新。

    用一个全新的小 MLP 在双高斯数据上跑 300 步，检查损失明显下降、参数确实被更新。"""
    import torch
    from torch import nn

    name = "训练一步 train_step"

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

        def forward(self, x, t):
            t = torch.as_tensor(t, dtype=x.dtype).reshape(-1, 1).expand(len(x), 1)
            return self.net(torch.cat([x, t], dim=1))

    torch.manual_seed(0)
    model = _MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    means = torch.tensor([[-3.0, 0.0], [3.0, 0.0]])
    losses = []
    for _ in range(300):
        x1 = means[torch.randint(0, 2, (256,))] + 0.7 * torch.randn(256, 2)
        ok, loss = safe_call(name, fn, model, opt, x1)
        if not ok:
            return False
        try:
            losses.append(float(loss))
        except (TypeError, ValueError):
            return report(name, False, f"应返回这一步的损失（一个数），实际 {loss!r}", "return loss.item()")
    if all(torch.equal(a, p.detach()) for a, p in zip(before, model.parameters())):
        return report(name, False, "300 步之后参数一点没变", "五步核心：optimizer.zero_grad() → loss → loss.backward() → optimizer.step()")
    first, last = float(np.mean(losses[:30])), float(np.mean(losses[-30:]))
    if not np.isfinite(last):
        return report(name, False, "损失变成了 NaN/inf", "检查学习率与损失的写法")
    return report(name, last < 0.75 * first,
                  f"损失应明显下降：前 30 步平均 {first:.3f}，后 30 步平均 {last:.3f}",
                  "确认每一步都调用了 optimizer.zero_grad()（否则梯度累加），且 fm_loss 用的是传进来的 model")


# ---------- 作业：步数 × 种子的结果结构与趋势 ----------
def steps_trend(results):
    """期望结构：results[数据集名][欧拉步数] = {"ed": [3 个种子], "precision": [3 个种子]}，
    步数恰为 {2, 5, 20, 100}，至少一个数据集。检查结构，并检查每个数据集上 100 步优于 2 步
    （能量距离更小、precision 更大）。"""
    name = "欧拉步数实验的结果与趋势"
    if not isinstance(results, dict) or not results:
        return report(name, False, f"results 应是非空 dict（键 = 数据集名），实际 {type(results).__name__}")
    for ds, by_steps in results.items():
        if not isinstance(by_steps, dict):
            return report(name, False, f"results[{ds!r}] 应是 {{步数: {{'ed': [...], 'precision': [...]}}}}")
        try:
            got = sorted(int(k) for k in by_steps)
        except (TypeError, ValueError):
            return report(name, False, f"results[{ds!r}] 的键应是欧拉步数（整数），实际 {list(by_steps)!r}")
        if got != sorted(STEPS):
            return report(name, False, f"results[{ds!r}] 应恰好包含步数 {STEPS}，实际 {got}")
        stats = {}
        for k, v in by_steps.items():
            if not isinstance(v, dict) or not {"ed", "precision"} <= set(v):
                return report(name, False, f"results[{ds!r}][{k}] 应是含 'ed' 与 'precision' 两个列表的 dict")
            for key in ("ed", "precision"):
                vals = [float(x) for x in v[key]]
                if len(vals) != N_SEEDS:
                    return report(name, False, f"results[{ds!r}][{k}]['{key}'] 应有 {N_SEEDS} 个种子的结果，实际 {len(vals)} 个",
                                  "同一配置跑 seed 0/1/2 三次——2 步时种子间方差很大，少于 3 个种子无法报告均值 ± 标准差")
                if not all(np.isfinite(vals)):
                    return report(name, False, f"results[{ds!r}][{k}]['{key}'] 里有非有限数：{vals}")
            if not all(0.0 <= float(p) <= 1.0 for p in v["precision"]):
                return report(name, False, f"results[{ds!r}][{k}]['precision'] 应在 [0, 1] 内：{list(v['precision'])}")
            stats[int(k)] = (float(np.mean(v["ed"])), float(np.mean(v["precision"])))
        if not stats[100][0] < stats[2][0]:
            return report(name, False, f"{ds}：100 步的能量距离（{stats[100][0]:.4f}）应小于 2 步（{stats[2][0]:.4f}）",
                          "检查欧拉采样：步长 1/n_steps、时间从 0（噪声）走到 1（数据）；不同步数要用同一个训练好的模型")
        if not stats[100][1] > stats[2][1]:
            return report(name, False, f"{ds}：100 步的 precision（{stats[100][1]:.3f}）应大于 2 步（{stats[2][1]:.3f}）",
                          "knn_precision(样本, 真实参考样本)：第一个参数是生成样本")
    return report(name, True)


def energy_distance_value(d):
    """月牙数据上，MLP 3→128→128→2、3000 步 Flow Matching、欧拉 100 步、2000 个样本的能量距离。"""
    return _in_range("月牙：欧拉 100 步样本的能量距离", d, -0.01, 0.04,
                     "本机实测 0.004–0.02（真 vs 真的噪声地板约 0 ± 0.003）。明显偏大：检查采样是否从 t=0 的"
                     "N(0, I) 噪声出发、是否走满 0→1、训练时插值与目标是否配套；"
                     "远小于 −0.01：检查是否把同一组样本和自己比较")


def mnist_samples_shape(x):
    """MNIST Flow Matching 的生成样本：(N, 1, 14, 14) 或 (N, 1, 28, 28) 的有限张量，像素大体在 [−1, 1]，
    整体均值接近真实 MNIST（背景 −1 占多数，均值约 −0.7）。"""
    import torch

    name = "MNIST 生成样本"
    if not isinstance(x, torch.Tensor):
        return report(name, False, f"应传入 torch 张量，实际 {type(x).__name__}")
    if x.dim() != 4 or x.shape[1] != 1 or tuple(x.shape[2:]) not in ((14, 14), (28, 28)):
        return report(name, False, f"形状应是 (N, 1, 14, 14) 或 (N, 1, 28, 28)，实际 {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        return report(name, False, "样本里有 NaN/inf", "采样时是否在 torch.no_grad() 下、模型是否 eval()、步长是否 1/n_steps")
    m = float(x.float().mean())
    if m > -0.35:
        return report(name, False, f"样本均值 {m:.3f} 太高（真实数据约 −0.7，纯噪声约 0）——看起来还接近噪声",
                      "检查欧拉采样是否走满 t 从 0 到 1、训练步数是否足够、数据是否缩放到 [−1, 1]")
    frac_out = float((x.abs() > 1.5).float().mean())
    return report(name, frac_out < 0.05, f"有 {frac_out:.1%} 的像素绝对值 > 1.5，远超数据范围 [−1, 1]",
                  "训练数据要先缩放到 [−1, 1]（generative.load_mnist 已经做了）；采样步长过大也会冲出范围")
