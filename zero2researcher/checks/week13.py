"""第13周自查函数。用法：from zero2researcher.checks import week13 as ck; ck.normalized_adjacency(normalized_adjacency)

函数/类练习把函数或类本身传进来（不加括号），例如 ck.gcn_layer(GCNLayer)、ck.mean_pool(mean_pool)；
其余检查直接传入你算出的数。涉及训练随机性的结果按合理区间检查（区间依据本机实测 ± 余量）。

参考实现：GCN 的传播规则照 Kipf & Welling, ICLR 2017（arXiv:1609.02907）Eq. 8 的
renormalization trick：Â = D̃^{-1/2} (A + I) D̃^{-1/2}，一层 H' = Â H W（激活放在层外）。
torch 只在函数内部 import，保证不装 torch 也能 import 本模块。
"""
import numpy as np

from . import report, safe_call

GCN_DEPTHS = (1, 2, 4, 8)
N_SEEDS = 3
CORA = {"n_nodes": 2708, "n_features": 1433, "n_classes": 7}


def _in_range(name, value, lo, hi, hint=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(value).__name__}")
    if not np.isfinite(v):
        return report(name, False, f"数值不是有限数（{v}）", hint)
    return report(name, lo <= v <= hi, f"期望落在 [{lo}, {hi}]，实际 {v:.4f}", hint)


def _np(x):
    """torch 张量 / numpy 数组 / 嵌套列表 → float64 numpy 数组。"""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


# ---------- 参考实现与测试图 ----------
def _ref_normalized_adjacency(A):
    A = _np(A)
    A_tilde = A + np.eye(len(A))
    d_inv_sqrt = 1.0 / np.sqrt(A_tilde.sum(axis=1))
    return d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]


def _test_graphs():
    """几个小的无向图（稠密 0/1 邻接矩阵）：路径、星形、带一个孤立节点的三角形。"""
    path = np.zeros((4, 4))
    for i in range(3):
        path[i, i + 1] = path[i + 1, i] = 1
    star = np.zeros((5, 5))
    star[0, 1:] = star[1:, 0] = 1
    tri_iso = np.zeros((4, 4))
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        tri_iso[i, j] = tri_iso[j, i] = 1          # 节点 3 是孤立节点（度为 0）
    return [path, star, tri_iso]


def _random_graph(n, p, seed):
    rng = np.random.default_rng(seed)
    upper = np.triu((rng.random((n, n)) < p).astype(float), k=1)
    return upper + upper.T


# ---------- S1 · 消息传递 ----------
def message_passing(fn):
    """fn(A, X, v)：A 是 (n, n) 稠密 0/1 邻接矩阵，X 是 (n, d) 节点特征，返回节点 v 的**邻居特征的平均** (d,)。"""
    import torch
    name = "单个节点的消息传递（邻居平均）"
    rng = np.random.default_rng(0)
    cases = [(_test_graphs()[0], 1), (_test_graphs()[1], 0), (_test_graphs()[1], 3),
             (_random_graph(8, 0.4, 1), 5)]
    for A_np, v in cases:
        X_np = rng.normal(size=(len(A_np), 3)).round(2)
        A, X = torch.tensor(A_np, dtype=torch.float32), torch.tensor(X_np, dtype=torch.float32)
        A_before = A.clone()
        ok, got = safe_call(name, fn, A, X, v)
        if not ok:
            return False
        try:
            got = _np(got).reshape(-1)
        except (TypeError, ValueError):
            return report(name, False, f"应返回一个长度为 d 的向量，实际是 {type(got).__name__}")
        if not torch.equal(A, A_before):
            return report(name, False, "函数改动了传进来的邻接矩阵 A",
                          "不要在原地修改 A（例如 A[v, v] = 1）；需要时先 A.clone()")
        nbrs = np.flatnonzero(A_np[v])
        want = X_np[nbrs].mean(axis=0)
        if got.shape != want.shape:
            return report(name, False, f"返回的向量长度应为 d = {want.size}，实际 {got.size}",
                          "对邻居那几行取平均时，要沿“节点”那一维（第 0 维）求平均，保留特征维")
        if np.allclose(got, want, atol=1e-5):
            continue
        if np.allclose(got, X_np[nbrs].sum(axis=0), atol=1e-5):
            hint = "你返回的是邻居特征的**和**；题目要的是平均（再除以邻居个数）"
        elif np.allclose(got, X_np[np.append(nbrs, v)].mean(axis=0), atol=1e-5):
            hint = "你把节点 v 自己也算进去了；这里只平均**邻居**（自环留到 Â 再加）"
        elif np.allclose(got, X_np.mean(axis=0), atol=1e-5):
            hint = "你对所有节点取了平均；邻居只是 A 的第 v 行里等于 1 的那些位置"
        else:
            hint = "先从 A 的第 v 行找出邻居的下标，再取出这些邻居的特征行求平均（不含 v 自己）"
        return report(name, False, f"节点 {v} 的结果与参考不一致：期望 {np.round(want, 4)}，实际 {np.round(got, 4)}",
                      hint)
    return report(name, True)


def _adjacency_hint(A, got):
    """常见错误的诊断：A 与学生结果都是 numpy。"""
    n = len(A)
    deg = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    A_tilde = A + np.eye(n)
    if not np.all(np.isfinite(got)):
        return ("结果里有 inf / nan：通常是对整个矩阵 D̃ 取了 -1/2 次方（非对角的 0 变成 inf），"
                "或者忘了加自环、孤立节点的度为 0。应该先算度**向量**、对向量取 -1/2 次方，"
                "再把它放到对角线上")
    if np.allclose(got, d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :], atol=1e-6):
        return "看起来忘了加自环：先给 A 加上单位矩阵得到 Ã，度也要按 Ã 算"
    if np.allclose(got, A_tilde / A_tilde.sum(axis=1, keepdims=True), atol=1e-6):
        return "这是行归一化 D̃^{-1} Ã（均值聚合）；GCN 用对称归一化 D̃^{-1/2} Ã D̃^{-1/2}"
    if np.allclose(got, A_tilde, atol=1e-6):
        return "只加了自环，还没有做归一化：两边各乘一次 D̃^{-1/2}"
    d_tilde = A_tilde.sum(axis=1)
    if np.allclose(got, A_tilde * d_tilde[:, None] ** -0.5 * d_tilde[None, :] ** 0.5, atol=1e-6) or \
            np.allclose(got, A_tilde / np.sqrt(d_tilde.sum()), atol=1e-6):
        return "两边的 D̃ 指数要都是 -1/2：Â = D̃^{-1/2} Ã D̃^{-1/2}（不是 D̃^{-1/2} Ã D̃^{1/2}，也不是除以一个总数）"
    return "按 Kipf & Welling Eq. 8：Â = D̃^{-1/2} (A + I) D̃^{-1/2}，D̃ 是 A + I 的度矩阵；用 @ 做矩阵乘法"


def normalized_adjacency(answer, A=None):
    """answer 可以是函数 normalized_adjacency(A) → Â，也可以是你算好的矩阵（此时要同时传 A=邻接矩阵）。"""
    name = "对称归一化邻接矩阵 Â"
    if callable(answer):
        import torch
        cases = _test_graphs() + [_random_graph(10, 0.3, 2)]
        if A is not None:
            cases.append(_np(A))
        for A_np in cases:
            At = torch.tensor(A_np, dtype=torch.float32)
            A_before = At.clone()
            ok, got = safe_call(name, answer, At)
            if not ok:
                return False
            if not torch.equal(At, A_before):
                return report(name, False, "函数改动了传进来的 A（原地加了自环？）",
                              "加自环时要得到一个新矩阵，不要用 += 这类原地运算修改传进来的 A")
            try:
                got = _np(got)
            except (TypeError, ValueError):
                return report(name, False, f"应返回一个 (n, n) 矩阵，实际是 {type(got).__name__}")
            if got.shape != A_np.shape:
                return report(name, False, f"形状应为 {A_np.shape}，实际 {got.shape}",
                              "度矩阵要放到对角线上（torch.diag），再用 @ 做矩阵乘法；不要用 * 逐元素乘一个向量")
            if not np.allclose(got, _ref_normalized_adjacency(A_np), atol=1e-6):
                return report(name, False, f"在一个 {len(A_np)} 个节点的测试图上与参考不一致",
                              _adjacency_hint(A_np, got))
        return report(name, True)
    if A is None:
        return report(name, False, "传入矩阵时还要传入它对应的邻接矩阵：ck.normalized_adjacency(A_hat, A=A)",
                      "或者直接传函数本身：ck.normalized_adjacency(normalized_adjacency)")
    try:
        A_np, got = _np(A), _np(answer)
    except (TypeError, ValueError):
        return report(name, False, "应传入 (n, n) 矩阵（torch 张量或 numpy 数组）")
    if got.shape != A_np.shape:
        return report(name, False, f"形状应为 {A_np.shape}，实际 {got.shape}")
    if np.allclose(got, _ref_normalized_adjacency(A_np), atol=1e-6):
        return report(name, True)
    return report(name, False, "与参考不一致（逐元素误差超过 1e-6）", _adjacency_hint(A_np, got))


def _find_weights(layer, in_dim, out_dim):
    """在学生的层里找出唯一的 2 维权重（及可选的 bias），返回 (W 形如 (in, out), b 或 None)。"""
    mats = [p for p in layer.parameters() if p.dim() == 2]
    vecs = [p for p in layer.parameters() if p.dim() == 1]
    if len(mats) != 1:
        return None
    W = mats[0].detach().double()
    if tuple(W.shape) == (out_dim, in_dim):        # nn.Linear 的布局：y = x @ W.T
        W = W.T
    elif tuple(W.shape) != (in_dim, out_dim):
        return None
    b = None
    if len(vecs) == 1 and vecs[0].numel() == out_dim:
        b = vecs[0].detach().double()
    elif vecs:
        return None
    return W, b


def gcn_layer(cls):
    """学生的 GCNLayer(in_dim, out_dim)：forward(X, A_hat) 返回 Â X W（可带 bias，不带激活）。"""
    import torch
    name = "GCNLayer"
    torch.manual_seed(0)
    try:
        layer = cls(5, 3)
    except NotImplementedError:
        return report(name, False, "__init__ 还没有实现（仍是 NotImplementedError）")
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"构造 GCNLayer(5, 3) 时出错 {type(e).__name__}: {e}",
                      "构造参数是 (in_dim, out_dim)；别忘了 super().__init__()")
    if not any(True for _ in layer.parameters()):
        return report(name, False, "层里没有可训练参数",
                      "权重要注册成参数：用一个 nn.Linear 子模块，或者包成 nn.Parameter，优化器才找得到它")
    A_np = _random_graph(6, 0.5, 3)
    A_hat = torch.tensor(_ref_normalized_adjacency(A_np), dtype=torch.float32)
    X = torch.randn(6, 5)
    try:
        with torch.no_grad():
            out = layer(X, A_hat)
    except NotImplementedError:
        return report(name, False, "forward 还没有实现（仍是 NotImplementedError）")
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"前向时出错 {type(e).__name__}: {e}",
                      "forward 收到 (X, A_hat) 两个参数；先对每个节点做线性变换，再用 Â 在节点之间混合。"
                      "逐行 print(t.shape) 看形状：(n, n) @ (n, out) → (n, out)")
    if tuple(out.shape) != (6, 3):
        return report(name, False, f"输入 X (6, 5)、Â (6, 6) 时输出形状应为 (6, 3)，实际 {tuple(out.shape)}",
                      "每一行是一个节点：(n, in) @ (in, out) 得 (n, out)，再 Â (n, n) @ 它")
    found = _find_weights(layer, 5, 3)
    if found is not None:
        W, b = found
        Xd, Ad = X.double(), A_hat.double()
        ref = Ad @ Xd @ W + (b if b is not None else 0.0)
        got = out.detach().double()
        # 带 bias 的 nn.Linear 再左乘 Â（即 Â (XW + b)）也是合理写法，同样接受
        ref_alt = Ad @ (Xd @ W + b) if b is not None else ref
        if not (torch.allclose(got, ref, atol=1e-5) or torch.allclose(got, ref_alt, atol=1e-5)):
            if torch.allclose(got, Xd @ W + (b if b is not None else 0.0), atol=1e-5):
                hint = "输出等于 X W，没有乘 Â——在邻居之间汇总（消息传递）那一步漏了"
            elif torch.allclose(got, torch.relu(ref), atol=1e-5) or torch.allclose(got, torch.tanh(ref), atol=1e-5):
                hint = "层内多加了激活函数；本课程的 GCNLayer 只做 Â H W，激活放在模型的 forward 里"
            else:
                hint = "按 Kipf & Welling Eq. 8：H' = Â H W；注意是 Â 在左边乘（聚合的是节点那一维）"
            return report(name, False, "形状对，但数值与用同一组权重按 Â X W 重算的结果不一致", hint)
    # 无论能否识别权重，都验证置换等变性
    perm = torch.randperm(6, generator=torch.Generator().manual_seed(1))
    P = torch.eye(6)[perm]
    with torch.no_grad():
        out_p = layer(P @ X, P @ A_hat @ P.T)
    if not torch.allclose(out_p, P @ out, atol=1e-5):
        return report(name, False, "节点重新编号后，输出没有被同样地重新编号（不满足置换等变性）",
                      "检查是否在层里用到了节点的绝对位置（例如按下标切片、或把 Â 存成了固定的属性）")
    return report(name, True)


def permutation_equivariance(diff_node, diff_graph):
    """两个数：max|f(PX, PÂPᵀ) − P f(X, Â)|（节点级，应≈0）与
    |readout(f(PX, PÂPᵀ)) − readout(f(X, Â))|（图级求和/平均读出，应≈0）。"""
    name = "置换等变性 / 不变性验证"
    try:
        a, b = float(diff_node), float(diff_graph)
    except (TypeError, ValueError):
        return report(name, False, "应传入两个数：节点级最大差异、图级读出的最大差异")
    if not (np.isfinite(a) and np.isfinite(b)):
        return report(name, False, "差异不是有限数")
    if a > 1e-4:
        return report(name, False, f"节点级差异应≈0（浮点误差 < 1e-4），实际 {a:.2e}",
                      "比较的是 f(PX, P Â Pᵀ) 与 P f(X, Â)——输出也要用同一个 P 重排；"
                      "节点重新编号时邻接矩阵的行和列都要换：P Â Pᵀ（只换 X 不换 Â 等于把图改了）")
    if b > 1e-4:
        return report(name, False, f"图级读出的差异应≈0，实际 {b:.2e}",
                      "读出要对节点那一维求和或平均：H.sum(dim=0)；它与节点顺序无关")
    return report(name, True)


# ---------- S2 · Cora ----------
def cora_stats(n_nodes, n_features, n_classes, n_edges=None, n_train=None):
    name = "Cora 数据统计"
    got = {"n_nodes": n_nodes, "n_features": n_features, "n_classes": n_classes}
    for key, want in CORA.items():
        try:
            v = int(got[key])
        except (TypeError, ValueError):
            return report(name, False, f"{key} 应是整数，实际 {got[key]!r}")
        if v != want:
            hints = {"n_nodes": "节点数可以直接从 Data 对象上读到（先 print(cora) 看看它有哪些属性）",
                     "n_features": "特征维度就是节点特征矩阵 x 的列数（dataset 上也有对应的属性）",
                     "n_classes": "类别数可以从 dataset 上读，也可以由标签的最大值推出（类别从 0 开始编号）"}
            return report(name, False, f"{key} 应为 {want}，实际 {v}", hints[key])
    if n_edges is not None:
        try:
            e = int(n_edges)
        except (TypeError, ValueError):
            return report(name, False, f"n_edges 应是整数，实际 {n_edges!r}")
        if e == 5429:
            return report(name, False, "5429 是原始 Cora 数据集文献里的引用数，不是 PyG 里的边数",
                          "PyG 的 data.num_edges 是 10556 条**有向**边（每条无向边存两次），即 5278 条无向边")
        if e not in (10556, 5278):
            return report(name, False, f"n_edges 应为 10556（有向，data.num_edges）或 5278（无向），实际 {e}")
    if n_train is not None:
        try:
            t = int(n_train)
        except (TypeError, ValueError):
            return report(name, False, f"n_train 应是整数，实际 {n_train!r}")
        if t != 140:
            return report(name, False, f"公开划分的训练节点应为 140（每类 20 个），实际 {t}",
                          "训练节点由 train_mask 标出：数的是其中 True 的个数，不是 mask 的长度")
    return report(name, True)


def mlp_cora_accuracy(acc):
    return _in_range("Cora 上 MLP 基线测试准确率", acc, 0.45, 0.65,
                     "只用节点特征、不看图：Linear(1433, 16) → ReLU → Dropout(0.5) → Linear(16, 7)，"
                     "Adam lr 0.01、weight decay 5e-4、200 epochs，只在 train_mask 上算损失、在 test_mask 上评估；"
                     "本机实测 3 个种子 0.538–0.560（Kipf & Welling 表 3 的 MLP：55.1%）")


def _eval_mode_accuracy(model, data):
    """用 eval 模式（dropout 关闭）重算 data.test_mask 上的准确率；不改变模型原来的 train/eval 状态。"""
    import torch
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
    finally:
        model.train(was_training)
    mask = data.test_mask
    return float((pred[mask] == data.y[mask]).float().mean())


def gcn_cora_accuracy(acc, model=None, data=None):
    """2 层 GCN 在 Cora 公开划分上的测试准确率。

    同时传入 model（训练好的模型）与 data（PyG 的 Cora Data）时，还会用 eval 模式重算一次准确率，
    与 acc 对账：评估时忘了 model.eval()（dropout 还开着），报告的数会与重算的不一致。"""
    name = "Cora 上 2 层 GCN 测试准确率"
    # 下界 0.785：本机正确实现 10 个种子 0.798–0.818；评估时忘了 eval()（dropout 开着）10 个种子 0.755–0.777
    lo, hi = 0.785, 0.84
    hint = ("GCNConv(1433, 16) → ReLU → Dropout(0.5) → GCNConv(16, 7)，其余同 MLP；"
            "本机实测 10 个种子 0.798–0.818（论文表 2：81.5%）。明显偏低时检查：forward 有没有把 "
            "edge_index 传给每一层；评估前有没有切到 eval 模式（dropout 在评估时必须关掉）")
    try:
        in_band = bool(np.isfinite(float(acc)) and lo <= float(acc) <= hi)
    except (TypeError, ValueError):
        in_band = False
    if model is None and data is None or not in_band:
        return _in_range(name, acc, lo, hi, hint)     # 不对账时只查区间；区间外直接报失败
    if model is None or data is None:
        return report(name, False, "要对账就同时传入 model 与 data：ck.gcn_cora_accuracy(acc, model=..., data=cora)")
    try:
        ref = _eval_mode_accuracy(model, data)
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"用传入的 model 在 data 上重算准确率时出错 {type(e).__name__}: {e}",
                      "model 应是训练好的模型本身，data 是 Planetoid 的 Cora（含 x、edge_index、y、test_mask）")
    if abs(float(acc) - ref) <= 0.0015:          # 允许 1 个测试节点的浮点抖动
        return report(name, True)
    if model.training:
        hint = ("模型现在仍处于 train 模式：评估时 dropout 还开着，每次前向都随机丢掉一半隐单元，"
                "准确率偏低而且每次不同。评估前先切换到 eval 模式")
    else:
        hint = ("检查评估用的是不是这个模型、这份数据与 test_mask；"
                "以及计算准确率时模型是否处于 eval 模式（之后再切回 eval 不算数）")
    return report(name, False, f"你报告的 {float(acc):.4f} 与用这个模型在 eval 模式下重算的 {ref:.4f} 不一致", hint)


def oversmoothing(acc2, acc8):
    name = "过平滑：8 层 vs 2 层 GCN"
    try:
        a2, a8 = float(acc2), float(acc8)
    except (TypeError, ValueError):
        return report(name, False, "应传入两个数：2 层的测试准确率、8 层的测试准确率")
    if not (np.isfinite(a2) and np.isfinite(a8) and 0 <= a2 <= 1 and 0 <= a8 <= 1):
        return report(name, False, f"准确率应在 [0, 1] 内，实际 {a2}, {a8}")
    if a2 < 0.70:
        return report(name, False, f"2 层 GCN 应在 0.80 左右，实际 {a2:.4f}", "先让 2 层模型达到正常水平再比较")
    return report(name, a8 < a2 - 0.05,
                  f"8 层应比 2 层低 5 个百分点以上，实际 2 层 {a2:.4f} / 8 层 {a8:.4f}",
                  "8 层要真的叠 8 个 GCNConv（中间层 16 → 16），其余设置与 2 层完全相同；"
                  "本机实测（原始 0/1 特征）8 层约 0.62")


# ---------- S3 · 句子嵌入 ----------
def mean_pool(fn):
    """fn(last_hidden_state, attention_mask)：(B, T, H) 与 (B, T) → (B, H)，只平均 attention_mask 为 1 的位置。"""
    import torch
    name = "按 attention_mask 的平均池化"
    torch.manual_seed(0)
    h = torch.randn(3, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    h[mask == 0] = 100.0                    # padding 位置填一个很大的数：没按 mask 平均会一眼看出来
    h_before, mask_before = h.clone(), mask.clone()
    ok, got = safe_call(name, fn, h, mask)
    if not ok:
        return False
    if not (torch.equal(h, h_before) and torch.equal(mask, mask_before)):
        return report(name, False, "函数改动了传进来的张量", "不要原地修改输入（例如 h *= mask）；写成 h * m")
    try:
        got = _np(got)
    except (TypeError, ValueError):
        return report(name, False, f"应返回 (B, H) 张量，实际是 {type(got).__name__}")
    hd, md = h.double().numpy(), mask.double().numpy()
    want = (hd * md[..., None]).sum(1) / md.sum(1, keepdims=True)
    if got.shape != want.shape:
        return report(name, False, f"输入 (3, 5, 4) 与 mask (3, 5) 时输出形状应为 (3, 4)，实际 {got.shape}",
                      "沿 token 那一维（dim=1）求和；mask 先 unsqueeze(-1) 变成 (B, T, 1) 才能和 (B, T, H) 相乘")
    if np.allclose(got, want, atol=1e-5):
        return report(name, True)
    if np.allclose(got, hd.mean(1), atol=1e-4):
        hint = "你对所有位置（含 padding）取了平均；只平均 attention_mask 为 1 的位置"
    elif np.allclose(got, (hd * md[..., None]).sum(1), atol=1e-4):
        hint = "只做了加权求和，还要除以每个句子的真实 token 数：mask.sum(dim=1)"
    elif np.allclose(got, (hd * md[..., None]).sum(1) / hd.shape[1], atol=1e-4):
        hint = "除数应是每个句子的真实 token 数 mask.sum(dim=1, keepdim=True)，而不是总长度 T"
    elif np.allclose(got, hd[:, 0], atol=1e-4):
        hint = "你取的是第一个位置（[CLS]）的向量；这一题要的是按 mask 平均"
    else:
        hint = ("思路：先让 mask 能和 (B, T, H) 的隐藏状态逐元素相乘（形状要对齐、类型要是浮点），"
                "把 padding 位置清零后沿 token 维求和，再除以每句的真实 token 数")
    return report(name, False, "与参考结果不一致", hint)


def probe_accuracy(acc):
    return _in_range("DistilBERT 嵌入 + 逻辑回归（线性探针）测试准确率", acc, 0.80, 0.89,
                     "冻结的 distilbert-base-uncased、max_length 256、torch.no_grad() 下取 last_hidden_state；"
                     "mean-pool 要按 attention_mask 加权。本机实测：只用前 500/500 条时 mean-pool 0.816、"
                     "[CLS] 0.830；全量 5000/2000 时两者都在 0.85–0.87 之间。明显偏低时检查：模型是否处于 eval 模式、"
                     "嵌入与标签的顺序是否对得上")


def tfidf_baseline(acc):
    return _in_range("TF-IDF + 逻辑回归基线", acc, 0.85, 0.88,
                     "与第 6 周完全相同：TfidfVectorizer(min_df=2) 只在训练集上 fit，"
                     "LogisticRegression(max_iter=1000)；本机实测 0.8665")


# ---------- 作业：GCN 层数消融（只查结构，不卡数值） ----------
def gcn_depth_results(results):
    """期望结构：results["MLP"] 与 results[L]（L ∈ {1, 2, 4, 8}）都是
    {"accs": [3 个种子的测试准确率], "mean": float, "std": float}。"""
    name = "GCN 层数消融结果结构"
    if not isinstance(results, dict):
        return report(name, False, f"results 应是 dict，实际是 {type(results).__name__}")
    keys = set()
    for k in results:
        if isinstance(k, str) and k.upper() == "MLP":
            keys.add("MLP")
            continue
        try:
            keys.add(int(k))
        except (TypeError, ValueError):
            return report(name, False, f"键应是 \"MLP\" 或层数（整数），实际出现 {k!r}")
    expected = {"MLP", *GCN_DEPTHS}
    if keys != expected:
        missing = sorted(map(str, expected - keys))
        extra = sorted(map(str, keys - expected))
        return report(name, False, f"应恰好包含 MLP 基线与层数 {list(GCN_DEPTHS)}；缺 {missing}，多 {extra}",
                      "没有基线的消融无法解读：results[\"MLP\"] 也要跑 3 个种子")
    accs_by_key = {}
    for k, v in results.items():
        if not isinstance(v, dict) or not {"accs", "mean", "std"} <= set(v):
            return report(name, False, f"results[{k!r}] 应是含 accs / mean / std 的 dict，实际 {v!r}")
        try:
            accs = [float(a) for a in v["accs"]]
        except (TypeError, ValueError):
            return report(name, False, f"results[{k!r}]['accs'] 应是数的列表")
        if len(accs) != N_SEEDS:
            return report(name, False, f"results[{k!r}]['accs'] 应是 {N_SEEDS} 个种子的结果，实际 {len(accs)} 个",
                          "同一配置跑 seed 0/1/2 三次——少于 3 个种子无法报告均值±标准差")
        if not all(np.isfinite(a) and 0.0 <= a <= 1.0 for a in accs):
            return report(name, False, f"results[{k!r}]['accs'] 里有不在 [0, 1] 的值：{accs}")
        m, s = float(np.mean(accs)), float(np.std(accs, ddof=1))
        try:
            mean_ok = abs(float(v["mean"]) - m) <= 1e-6
            std_ok = min(abs(float(v["std"]) - s), abs(float(v["std"]) - float(np.std(accs)))) <= 1e-6
        except (TypeError, ValueError):
            return report(name, False, f"results[{k!r}] 的 mean / std 应是数")
        if not mean_ok:
            return report(name, False, f"results[{k!r}]['mean'] 与 accs 的均值不一致",
                          "mean, std = experiments.mean_std(accs)")
        if not std_ok:
            return report(name, False, f"results[{k!r}]['std'] 与 accs 的标准差不一致",
                          "mean, std = experiments.mean_std(accs)")
        accs_by_key[k] = tuple(round(a, 6) for a in accs)
    gcn_rows = [a for k, a in accs_by_key.items() if not (isinstance(k, str) and k.upper() == "MLP")]
    if len(set(gcn_rows)) == 1:
        return report(name, False, "四种层数的 accs 完全相同——层数真的改了吗？",
                      "检查建模型时是否把 num_layers 传进去了")
    return report(name, True)
