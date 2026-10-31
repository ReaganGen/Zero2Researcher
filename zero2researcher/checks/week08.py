"""第8周自查函数。用法：from zero2researcher.checks import week08 as ck; ck.mnist_mlp_accuracy(acc)

函数类练习把函数本身传进来（不加括号），例如 ck.train_loop(train_one_epoch, evaluate)；
其余检查直接传入你算出的数值。涉及随机性的结果按合理区间检查。
torch 只在函数内部 import，保证不装 torch 也能 import 本模块。
"""
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


# ---------- 课堂练习 ----------
def tensor_shapes(a, b, c):
    """E1：reshape、转置、广播三件套。"""
    import torch
    name = "tensor 形状操练"
    for label, t in [("a", a), ("b", b), ("c", c)]:
        if not isinstance(t, torch.Tensor):
            return report(name, False, f"{label} 应是 torch.Tensor，实际是 {type(t).__name__}")
    exp_a = torch.arange(12).reshape(3, 4)
    if a.shape != (3, 4) or not torch.equal(a.reshape(3, 4), exp_a):
        return report(name, False, f"a 应是 0..11 组成的 3×4 tensor，实际 shape {tuple(a.shape)}",
                      "torch.arange(12).reshape(3, 4)")
    if b.shape != (4, 3) or not torch.equal(b, exp_a.T):
        return report(name, False, f"b 应是 a 的转置（shape (4, 3)），实际 shape {tuple(b.shape)}",
                      "a.T 或 a.permute(1, 0)")
    exp_c = torch.arange(3).reshape(3, 1) * 10 + torch.arange(4)
    if c.shape != (3, 4) or not torch.equal(c, exp_c):
        return report(name, False, f"c 应满足 c[i, j] = 10*i + j（shape (3, 4)），实际 shape {tuple(c.shape)}",
                      "列向量 (3,1) 和行向量 (4,) 相加/相乘时会自动广播成 (3,4)")
    return report(name, True)


def numpy_roundtrip(t, back):
    """E2：NumPy 数组 → tensor → NumPy 数组。"""
    import torch
    name = "NumPy ↔ tensor 往返"
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    if not isinstance(t, torch.Tensor):
        return report(name, False, f"t 应是 torch.Tensor，实际是 {type(t).__name__}",
                      "torch.from_numpy(arr) 或 torch.tensor(arr)")
    if tuple(t.shape) != (2, 3) or not np.allclose(t.detach().cpu().numpy(), arr):
        return report(name, False, "t 的 shape 或数值与原数组不一致")
    if not isinstance(back, np.ndarray):
        return report(name, False, f"back 应是 np.ndarray，实际是 {type(back).__name__}", "t.numpy()")
    if back.shape != (2, 3) or not np.allclose(back, arr):
        return report(name, False, "back 的 shape 或数值与原数组不一致")
    return report(name, True)


def five_step_order(order):
    """E3：训练循环五步排序（A backward、B step、C 前向、D zero_grad、E 损失）。

    判定规则：C 在 E 前、E 在 A 前、A 在 B 前，并且 D 不夹在 A 与 B 之间。
    因此接受全部四种正确顺序：DCEAB（清零放最前）、CDEAB、CEDAB（本课教的）、
    CEABD（官方 Quickstart / Optimization 教程：step 之后清零，下一轮 backward 之前梯度已是 0）。"""
    name = "训练循环五步排序"
    try:
        got = [str(x).strip().upper() for x in order]
    except TypeError:
        return report(name, False, f"应传入 5 个字母组成的列表，实际是 {type(order).__name__}")
    if sorted(got) != ["A", "B", "C", "D", "E"]:
        return report(name, False, f"应恰好包含 A–E 各一次，实际 {got}")
    pos = {s: i for i, s in enumerate(got)}
    if not pos["C"] < pos["E"] < pos["A"] < pos["B"]:
        return report(name, False, f"顺序 {got} 不对",
                      "前向 → 损失 → 反向 → 更新 这四步的先后是固定的："
                      "没有前向就算不出损失，没有损失就没法反向，没有梯度就没法更新")
    return report(name, not (pos["A"] < pos["D"] < pos["B"]),
                  f"顺序 {got} 不对",
                  "zero_grad 夹在 backward 和 step 之间，会把刚算好的梯度清零，step 就什么也没更新。"
                  "想一想：清零只要发生在下一次 backward 之前就行，它可以放在哪几个位置？")


def mse_grad_agreement(rel_err):
    return _in_range("autograd 与手推 MSE 梯度的相对误差", rel_err, 0.0, 1e-5,
                     "rel_err = |w.grad - 手推梯度| / max(|手推梯度|, 1e-12)；"
                     "注意用 float64（torch.tensor(x_data) 对 float64 的 NumPy 数组默认就是 float64）")


# ---------- 作业：训练循环 ----------
def train_loop(train_fn, eval_fn):
    """在小合成数据上验证 train_one_epoch / evaluate：
    损失下降、evaluate 的准确率正确、evaluate 不改参数、用了 no_grad 与 eval 模式。"""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    name = "训练循环 train_one_epoch / evaluate"

    class _Probe(nn.Module):
        """记录每次前向时的 (是否开着梯度, 是否训练模式)。"""

        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.lin = nn.Linear(2, 2)
            self.records = []

        def forward(self, x):
            self.records.append((torch.is_grad_enabled(), self.training))
            return self.lin(x)

    g = torch.Generator().manual_seed(0)
    X = torch.randn(256, 2, generator=g)
    y = (X[:, 0] + X[:, 1] > 0).long()
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=False)
    device = torch.device("cpu")
    model = _Probe()
    criterion = nn.CrossEntropyLoss()

    # 0) 用一个"不更新参数"的探针优化器跑一遍，在每次 step 时抓拍梯度，
    #    据此判断 backward 之前梯度是否被清零（接受 optimizer.zero_grad() 或 model.zero_grad()）。
    probe_opt = torch.optim.SGD(model.parameters(), lr=0.0)
    snaps = []
    _orig_step = probe_opt.step

    def _recording_step(closure=None):
        grad = model.lin.weight.grad
        snaps.append(None if grad is None else grad.detach().clone())
        return _orig_step(closure)

    probe_opt.step = _recording_step
    ok, first = safe_call(name, train_fn, model, loader, criterion, probe_opt, device)
    if not ok:
        return False
    if len(snaps) >= 2 and snaps[0] is not None and snaps[1] is not None:
        # 参考梯度：参数没动过（lr=0），对前两个 batch 分别算"干净"的梯度
        batches = list(loader)
        refs = []
        for xb, yb in batches[:2]:
            model.zero_grad()
            criterion(model.lin(xb), yb).backward()
            refs.append(model.lin.weight.grad.detach().clone())
        model.zero_grad()
        if not torch.allclose(snaps[1], refs[1], atol=1e-5):
            if torch.allclose(snaps[1], refs[0] + refs[1], atol=1e-5):
                return report(name, False, "第二个 batch 的梯度是前两个 batch 之和——梯度在跨 batch 累加",
                              "loss.backward() 之前要 optimizer.zero_grad()（.grad 是累加的，这正是 micrograd 教的）")
            return report(name, False, "step 时的梯度与该 batch 应有的梯度不一致",
                          "每个 batch 都要：前向 → 损失 → zero_grad → backward → step")

    # 1) train_one_epoch 返回 float，且多个 epoch 后损失下降
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    if not isinstance(first, float):
        return report(name, False, f"train_one_epoch 应返回 float 平均损失，实际 {type(first).__name__}",
                      "criterion 返回的是这个 batch 的平均损失：按 batch 大小加权累加，最后除以样本总数，返回 Python float")
    losses = [first]
    for _ in range(4):
        losses.append(train_fn(model, loader, criterion, optimizer, device))
    if not (np.isfinite(losses[-1]) and losses[-1] < losses[0]):
        return report(name, False,
                      f"训练 5 个 epoch 后损失应明显下降，实际 {losses[0]:.4f} → {losses[-1]:.4f}",
                      "检查五步的顺序，尤其是 optimizer.zero_grad() 是否在 loss.backward() 之前")
    # 参考准确率（此时模型已训练好，线性可分数据应 >= 0.9）
    with torch.no_grad():
        ref_acc = (model.lin(X).argmax(dim=1) == y).float().mean().item()
    # 2) evaluate：返回正确的准确率
    model.records.clear()
    before = [p.detach().clone() for p in model.parameters()]
    ok, acc = safe_call(name, eval_fn, model, loader, device)
    if not ok:
        return False
    if not isinstance(acc, float) or not (0.0 <= acc <= 1.0):
        return report(name, False, f"evaluate 应返回 0–1 之间的 float 准确率，实际 {acc!r}")
    if abs(acc - ref_acc) > 1e-6:
        return report(name, False, f"evaluate 的准确率不对：期望 {ref_acc:.4f}，实际 {acc:.4f}",
                      "预测类别 = logits 在类别维（dim=1）上最大的那一个；数出与标签相等的个数，最后除以样本总数")
    # 3) evaluate 不得改参数
    after = list(model.parameters())
    if not all(torch.equal(b, a.detach()) for b, a in zip(before, after)):
        return report(name, False, "evaluate 改动了模型参数——评估阶段绝不能更新模型",
                      "evaluate 里不应出现 optimizer 或 backward")
    # 4) evaluate 期间必须关闭梯度并处于 eval 模式
    if any(grad_on for grad_on, _ in model.records):
        return report(name, False, "evaluate 的前向没有关闭梯度",
                      "把整个评估循环放进 with torch.no_grad(): 里")
    if any(training for _, training in model.records):
        return report(name, False, "evaluate 期间模型仍处于训练模式",
                      "evaluate 开头调用 model.eval()")
    # 5) evaluate 之后再训练时，train_one_epoch 必须把模型切回训练模式
    model.records.clear()
    train_fn(model, loader, criterion, optimizer, device)
    if not any(training for _, training in model.records):
        return report(name, False, "evaluate 之后再训练时模型没有回到训练模式",
                      "train_one_epoch 开头调用 model.train()")
    return report(name, True)


# ---------- 作业：MNIST 三方对比（10k 训练子集；区间按本机实测 ± 余量） ----------
def mnist_mlp_accuracy(acc):
    return _in_range("MNIST MLP 测试准确率", acc, 0.93, 0.97,
                     "MLP 784-256-10、Adam lr=1e-3、3 epochs、batch 64、"
                     "Normalize((0.1307,), (0.3081,))，训练用固定的前 1 万张")


def mnist_lr_accuracy(acc):
    return _in_range("MNIST 逻辑回归测试准确率", acc, 0.88, 0.93,
                     "LogisticRegression(max_iter=1000)，输入为展平像素 / 255，"
                     "训练子集与 MLP 完全相同")


def mnist_rf_accuracy(acc):
    return _in_range("MNIST 随机森林测试准确率", acc, 0.93, 0.97,
                     "RandomForestClassifier(n_estimators=100)，输入为展平像素，"
                     "训练子集与 MLP 完全相同")
