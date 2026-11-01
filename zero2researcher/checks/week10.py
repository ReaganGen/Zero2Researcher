"""第10周自查函数。用法：from zero2researcher.checks import week10 as ck; ck.conv_output_size(conv_output_size)

函数类练习把函数本身传进来（不加括号），例如 ck.conv2d_numpy(conv2d_numpy)；
模型类检查可以传类本身或一个实例，例如 ck.textcnn_shapes(TextCNN)；
其余检查直接传入你算出的数值。涉及训练随机性的结果按合理区间检查（区间依据本机实测 ± 余量）。
torch 只在函数内部 import，保证不装 torch 也能 import 本模块。
"""
import numpy as np

from . import report, safe_call

KERNEL_WIDTHS = (2, 3, 5, 7)
N_SEEDS = 3
BOARD_COLUMNS = ("model", "test_acc", "week")


def _in_range(name, value, lo, hi, hint=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(value).__name__}")
    if not np.isfinite(v):
        return report(name, False, f"数值不是有限数（{v}），训练大概率发散了", hint)
    return report(name, lo <= v <= hi,
                  f"期望落在 [{lo}, {hi}]，实际 {v:.4f}", hint)


# ---------- 参考实现 ----------
def _conv_out(W, K, P=0, S=1):
    """卷积 / 池化输出尺寸：floor((W − K + 2P) / S) + 1。"""
    return (W - K + 2 * P) // S + 1


def _ref_conv2d(img, kernel):
    """valid、stride 1 的二维互相关（与 torch.nn.functional.conv2d 同口径：核不翻转）。"""
    img, kernel = np.asarray(img, dtype=float), np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    oh, ow = img.shape[0] - kh + 1, img.shape[1] - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = (img[i:i + kh, j:j + kw] * kernel).sum()
    return out


def _exercise_cnn():
    """练习 E3 里那个小 CNN（MNIST 尺寸输入 1×28×28）。"""
    from torch import nn
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 5 * 5, 10),
    )


def _ref_param_count():
    return sum(p.numel() for p in _exercise_cnn().parameters())


# ---------- 练习：卷积的算术 ----------
def conv_output_size(fn):
    """学习者写的 conv_output_size(W, K, P, S)：在多组 (W, K, P, S) 上与公式比对。"""
    name = "卷积输出尺寸 conv_output_size"
    cases = [
        (28, 3, 0, 1),    # MNIST，3×3，valid → 26
        (28, 3, 1, 1),    # padding 1 保持尺寸 → 28
        (32, 5, 0, 1),    # blitz 第一层 → 28
        (32, 5, 2, 1),    # padding 2 保持尺寸 → 32
        (28, 2, 0, 2),    # 2×2 池化、步长 2 → 14
        (7, 3, 0, 2),     # 步长 2 → 3
        (8, 3, 0, 2),     # 除不尽：向下取整 → 3
        (32, 3, 1, 2),    # 步长 2 + padding 1 → 16
        (256, 5, 0, 1),   # 文本 Conv1d，L=256、k=5 → 252
    ]
    for W, K, P, S in cases:
        ok, got = safe_call(name, fn, W, K, P, S)
        if not ok:
            return False
        exp = _conv_out(W, K, P, S)
        if isinstance(got, float) and not float(got).is_integer():
            return report(name, False, f"W={W}, K={K}, P={P}, S={S} 时返回了小数 {got}",
                          "输出尺寸必须是整数：除不尽时向下取整，用整除 //")
        try:
            same = int(got) == exp
        except (TypeError, ValueError):
            return report(name, False, f"应返回一个整数，实际是 {type(got).__name__}")
        if not same:
            return report(name, False, f"W={W}, K={K}, P={P}, S={S} 时期望 {exp}，实际 {got}",
                          "公式：(W − K + 2P) // S + 1——padding 两边都加，所以是 2P")
    return report(name, True)


def conv2d_numpy(fn):
    """学习者写的 conv2d_numpy(img, kernel)：与 torch.nn.functional.conv2d 逐元素比对（容差 1e-6）。"""
    import torch
    import torch.nn.functional as F

    name = "conv2d_numpy 与 F.conv2d 对照"
    rng = np.random.default_rng(10)
    cases = [
        (rng.normal(size=(6, 6)), rng.normal(size=(3, 3))),
        (rng.normal(size=(7, 5)), rng.normal(size=(2, 3))),     # 非方形图与非方形核
        (rng.normal(size=(5, 5)), rng.normal(size=(5, 5))),     # 核与图一样大 → 1×1
        (np.tile(np.arange(6.0), (6, 1)),                       # 水平渐变图 + Sobel Gx
         np.array([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]])),
    ]
    for img, k in cases:
        ok, got = safe_call(name, fn, img.copy(), k.copy())
        if not ok:
            return False
        exp = F.conv2d(torch.tensor(img)[None, None], torch.tensor(k)[None, None])[0, 0].numpy()
        try:
            got = np.asarray(got, dtype=float)
        except (TypeError, ValueError):
            return report(name, False, f"应返回 NumPy 数组，实际是 {type(got).__name__}")
        if got.shape != exp.shape:
            return report(name, False,
                          f"图 {img.shape}、核 {k.shape} 时输出 shape 应为 {exp.shape}，实际 {got.shape}",
                          "valid 卷积：输出高 = H − kh + 1，宽 = W − kw + 1；"
                          "循环范围写成 range(H) 会越界或多出一圈")
        if not np.allclose(got, exp, atol=1e-6):
            flipped = _ref_conv2d(img, k[::-1, ::-1])
            if np.allclose(got, flipped, atol=1e-6):
                return report(name, False, "你的结果等于把核翻转 180° 之后的结果",
                              "你实现的是数学教科书里的'真卷积'（先翻转核）。PyTorch 的 conv2d 实际做的是"
                              "互相关：核不翻转，直接逐元素相乘再求和")
            return report(name, False,
                          f"图 {img.shape}、核 {k.shape} 时数值不一致（最大差 {np.abs(got - exp).max():.3g}）",
                          "每个输出位置 = (img[i:i+kh, j:j+kw] * kernel).sum()")
    return report(name, True)


def cnn_param_count(n):
    """练习 E3：那个小 CNN 的可训练参数总数（参考值现场搭同结构网络求和）。"""
    name = "小 CNN 的参数量"
    try:
        v = int(n)
        if v != float(n):
            raise ValueError
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个整数，实际是 {n!r}")
    exp = _ref_param_count()
    if v == exp - 8 - 16 - 10:
        return report(name, False, f"期望 {exp}，实际 {v}——只差三层的偏置",
                      "每个输出通道 / 输出神经元各有 1 个偏置：Conv2d(8→16) 有 16 个偏置")
    return report(name, v == exp, f"期望 {exp}，实际 {v}",
                  "Conv2d 参数 = 输出通道 × 输入通道 × k × k + 输出通道（偏置）；"
                  "Linear 的输入维度要先用尺寸公式算出：28 → 26 → 13 → 11 → 5，所以是 16×5×5；"
                  "ReLU / MaxPool / Flatten 没有参数")


def conv_layout(fn):
    """练习 E4：把 Embedding 的输出 (N, L, C) 换成 Conv1d 要的 (N, C, L)。"""
    import torch

    name = "Embedding → Conv1d 的维度转换"
    emb = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)    # N=2, L=5, C=3
    ok, got = safe_call(name, fn, emb.clone())
    if not ok:
        return False
    if not isinstance(got, torch.Tensor):
        return report(name, False, f"应返回 torch.Tensor，实际是 {type(got).__name__}")
    if tuple(got.shape) != (2, 3, 5):
        return report(name, False, f"输入 (N, L, C) = (2, 5, 3) 时应返回 (2, 3, 5)，实际 {tuple(got.shape)}",
                      "交换第 1、2 维：x.transpose(1, 2)")
    if not torch.equal(got, emb.transpose(1, 2)):
        return report(name, False, "shape 对了，但数值被打乱了",
                      "reshape / view 只是按内存顺序重新切块，不会交换维度——"
                      "每个位置的词向量被拆散了。用 x.transpose(1, 2) 或 x.permute(0, 2, 1)")
    return report(name, True)


def textcnn_shapes(model):
    """TextCNN 的形状契约：输入 (8, 256) 的 int64 词 id → 输出 (8, 2) 的 logits。

    可以传类（会用 vocab_size=20000 实例化）或已经建好的实例（不改变它的设备与模式）。
    额外用长度 100 的输入再测一次：全局 max pooling 与序列长度无关。
    """
    import torch

    name = "TextCNN 的输入输出形状"
    if isinstance(model, type):
        ok, model = safe_call(name, model, 20000)
        if not ok:
            return False
    if not isinstance(model, torch.nn.Module):
        return report(name, False, f"应传入 nn.Module 的类或实例，实际是 {type(model).__name__}")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return report(name, False, "模型没有任何参数——检查 __init__ 里是否定义了 Embedding / Conv1d / Linear")
    for L in (256, 100):
        x = torch.randint(2, 100, (8, L), dtype=torch.int64, device=device)
        try:
            with torch.no_grad():
                out = model(x)
        except NotImplementedError:
            return report(name, False, "forward 还没有实现（仍是 NotImplementedError）")
        except RuntimeError as e:
            msg = str(e)
            hint = ("Embedding 输出 (N, L, C)，Conv1d 要的是 (N, C, L)：卷积之前先 .transpose(1, 2)"
                    if "channels" in msg else
                    "先用 print(x.shape) 逐层打印形状，找到第一个对不上的地方")
            if L == 100 and "channels" not in msg:
                hint = ("长度 256 能跑、长度 100 报错：说明你按固定长度展平了。"
                        "TextCNN 应在时间维上做全局 max pooling：.max(dim=2).values")
            return report(name, False, f"输入 (8, {L}) 时运行出错 RuntimeError: {msg[:160]}", hint)
        except Exception as e:  # noqa: BLE001
            return report(name, False, f"输入 (8, {L}) 时运行出错 {type(e).__name__}: {e}")
        if not isinstance(out, torch.Tensor) or tuple(out.shape) != (8, 2):
            shape = tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__
            return report(name, False, f"输入 (8, {L}) 时输出应为 (8, 2)，实际 {shape}",
                          "最后一层 nn.Linear(n_filters × 核的种类数, 2)；输出 logits，不要 softmax")
    return report(name, True)


# ---------- 讲义：训练结果（区间依据本机实测） ----------
def mnist_cnn_accuracy(acc):
    return _in_range("MNIST 小 CNN 测试准确率", acc, 0.96, 0.99,
                     "Conv(1→32,3×3,pad 1) → ReLU → MaxPool → Conv(32→64,3×3,pad 1) → ReLU → MaxPool → "
                     "Linear(64×7×7, 10)；Adam 1e-3、batch 64、3 epochs，训练用前 1 万张；本机实测约 0.975")


def initial_loss(loss):
    return _in_range("训练前的初始损失（10 类）", loss, 2.15, 2.45,
                     "第 9 周诊断配方第一条：随机初始化的 10 类分类器，初始交叉熵应 ≈ ln(10) = 2.303。"
                     "明显偏离说明训练前就有东西不对，例如：输出层的类别数或标签编码不对（输出 100 类时约 ln100 = 4.6）、"
                     "末层偏置或初始化尺度过大、用错了损失函数、数据/标签有问题。"
                     "注意：先 softmax 再进 CrossEntropyLoss 的错误在初始损失上看不出来（初始时各类概率都≈0.1，"
                     "当成 logits 再算一次交叉熵仍≈ln10）；它要看训练后的损失是否卡在一块下不去的地板上"
                     "（理论下限约 1.46，第 9 周雷 c）")


def cifar_blitz_accuracy(acc, n_train=50000):
    """官方 blitz 网络 + SGD(1e-3, momentum 0.9) + batch 4 + 2 epochs。

    n_train=50000（完整配置，官方页报 55%）区间 [0.52, 0.58]；
    n_train=10000（慢机器的子集配置）区间 [0.32, 0.45]（本机实测 0.37–0.39）。
    """
    hint = ("照搬官方教程：Normalize((0.5,)*3, (0.5,)*3)、batch_size=4、"
            "SGD(lr=0.001, momentum=0.9)、CrossEntropyLoss、2 个 epoch")
    if n_train == 50000:
        return _in_range("CIFAR-10 blitz CNN 测试准确率（完整 50k）", acc, 0.52, 0.58,
                         hint + "；官方页报告 55%（随 PyTorch 版本浮动约 ±3 个百分点）")
    if n_train == 10000:
        return _in_range("CIFAR-10 blitz CNN 测试准确率（1 万张子集）", acc, 0.32, 0.45, hint)
    return report("CIFAR-10 blitz CNN 测试准确率", False,
                  f"只有 n_train=50000（完整配置）与 n_train=10000（子集）两档自查区间，实际传入 {n_train}")


def tfidf_baseline(acc):
    return _in_range("TF-IDF + 逻辑回归基线", acc, 0.85, 0.88,
                     "与第 6 周完全相同：TfidfVectorizer(min_df=2) 只在训练集上 fit，"
                     "LogisticRegression(max_iter=1000)；本机实测 0.8665")


def textcnn_accuracy(acc):
    return _in_range("TextCNN 测试准确率", acc, 0.70, 0.88,
                     "从零 Embedding(·, 64)、核宽 3/4/5 各 64 个、全局 max pooling、dropout 0.5、"
                     "Adam 1e-3、batch 64、5 epochs；本机实测约 0.77（15 epochs 约 0.81–0.82）。"
                     "明显低于 0.70：先查 transpose(1, 2) 与 max(dim=2) 的维度")


def imdb_board(df):
    """跨周 IMDB 对比表：列为 model / test_acc / week，至少含 TF-IDF 基线与 TextCNN 两行。"""
    name = "IMDB 跨周对比表"
    try:
        cols = list(df.columns)
        rows = df.to_dict("records")
    except AttributeError:
        return report(name, False, f"应传入 pandas DataFrame，实际是 {type(df).__name__}")
    if tuple(cols[:3]) != BOARD_COLUMNS:
        return report(name, False, f"前三列应依次为 {list(BOARD_COLUMNS)}，实际 {cols}",
                      "第 11、13、14 周会往同一张表追加行，列名必须固定")
    for r in rows:
        try:
            a = float(r["test_acc"])
        except (TypeError, ValueError):
            return report(name, False, f"test_acc 应是数，实际出现 {r['test_acc']!r}")
        if not 0.0 <= a <= 1.0:
            return report(name, False, f"test_acc 应在 [0, 1] 内，实际出现 {a}")
    models = " ".join(str(r["model"]).lower() for r in rows)
    if "tf-idf" not in models or "textcnn" not in models:
        return report(name, False, "表里至少要有 TF-IDF + LR 基线与 TextCNN 两行",
                      'model 列的写法沿用 "TF-IDF + LR"、"TextCNN (...)"')
    return report(name, True)


# ---------- 作业：研究实验 2（只查结构，不卡数值） ----------
def _accs_of(v):
    if isinstance(v, dict):
        v = v.get("accs")
    return list(v)


def kernel_width_results(results):
    """期望结构：results[k] = [3 个种子的测试准确率]，k ∈ {2, 3, 5, 7}。
    也接受 results[k] = {"accs": [...], ...}。只检查结构与"种子 / 核宽确实变了"，不卡数值。"""
    name = "核宽实验结果结构"
    if not isinstance(results, dict):
        return report(name, False, f"results 应是 dict，实际是 {type(results).__name__}")
    try:
        keys = {int(k) for k in results}
    except (TypeError, ValueError):
        return report(name, False, f"键应是核宽（整数），实际 {list(results)}")
    if keys != set(KERNEL_WIDTHS):
        return report(name, False, f"应恰好覆盖核宽 {list(KERNEL_WIDTHS)}，实际 {sorted(keys)}")
    table = {}
    for k, v in results.items():
        try:
            accs = [float(a) for a in _accs_of(v)]
        except (TypeError, ValueError, AttributeError):
            return report(name, False, f"results[{k}] 应是准确率列表（或含 'accs' 的 dict），实际 {v!r}")
        if len(accs) != N_SEEDS:
            return report(name, False, f"results[{k}] 应有 {N_SEEDS} 个种子的结果，实际 {len(accs)} 个",
                          "同一核宽跑 seed 0/1/2 三次——少于 3 个种子无法报告均值 ± 标准差")
        if not all(np.isfinite(a) and 0.0 <= a <= 1.0 for a in accs):
            return report(name, False, f"results[{k}] 里有不在 [0, 1] 的值：{accs}")
        if len(set(accs)) == 1:
            return report(name, False, f"results[{k}] 的 3 个种子结果完全相同：{accs}",
                          "每次 run 开头调用 experiments.set_seed(seed)，且 seed 要真的不同")
        table[int(k)] = tuple(accs)
    if len(set(table.values())) == 1:
        return report(name, False, "4 种核宽的结果一模一样——核宽大概没有真正传进模型",
                      "检查构造模型时是否用了循环变量 k：SingleWidthCNN(V, k=k)")
    return report(name, True)
