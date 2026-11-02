"""第10周自查函数。用法：from zero2researcher.checks import week10 as ck; ck.conv_output_size(conv_output_size)

函数类练习把函数本身传进来（不加括号），例如 ck.conv2d_numpy(conv2d_numpy)；
模型类检查可以传类本身或一个实例，例如 ck.textcnn_shapes(TextCNN)；
其余检查直接传入你算出的数值。涉及训练随机性的结果按合理区间检查（区间依据本机实测 ± 余量）。
torch 只在函数内部 import，保证不装 torch 也能 import 本模块。
"""
import math

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


# ---------- 讲义 S2 研究方法插曲：误差分析与数据错配 ----------
# 区间依据 8a 调研实测：官方 blitz Net（50k、2 epochs、batch 4、SGD 1e-3/0.9），种子 42 / 0 / 1，CPU 2 线程；
# MPS 训练出的 net 略有不同（种子 42 测试准确率 0.5559 vs CPU 0.5567），区间的余量足以覆盖。
# 只有 N_TRAIN = 50000（完整配置）时才卡数值区间；N_TRAIN = 10000 的子集配置只查结构与算术。
# 称呼（8a 审查 W10-1）：官方测试集前 5000 张 = 训练-开发集（干净、同分布，看错例用）；后 5000 张 + 对比度严重度 2
# = 目标开发集（"B 实验室"，不看错例、不调参数，只算总误差）。参数名与字典键（y_dev、"dev" 等）沿用旧名，
# 给学习者看的文字一律用新称呼。
N_DEV = 5000
HUMAN_ACCURACY = 0.9391          # Ho-Phuoc 2018（arXiv:1811.07270）：人类在 CIFAR-10 上的平均准确率


def _subset_note(name, n_train):
    if n_train == 50000:
        return False
    print(f"   （{name}：N_TRAIN = {n_train} 的子集配置下模型弱得多，只检查结构与算术，不卡数值区间）")
    return True


def error_sample(idx, y_dev, pred_dev, n=100):
    """100 个训练-开发集错例的下标：互不重复、都在训练-开发集（官方测试集前 5000 张）里、而且全是预测错的。"""
    name = "100 个训练-开发集错例"
    try:
        idx = np.asarray(idx)
        yt, yp = np.asarray(y_dev), np.asarray(pred_dev)
    except (TypeError, ValueError):
        return report(name, False, "应传入三个数组：sample（下标）、train_dev_y、train_dev_pred")
    if idx.shape != (n,) or not np.issubdtype(idx.dtype, np.integer):
        return report(name, False, f"sample 应是 {n} 个整数下标，实际形状 {idx.shape}、类型 {idx.dtype}")
    if len(set(idx.tolist())) != n:
        return report(name, False, "下标有重复", "无放回地抽：rng.choice(wrong, size=100, replace=False)")
    if idx.min() < 0 or idx.max() >= min(len(yt), N_DEV):
        return report(name, False, f"下标应在 [0, {N_DEV}) 内：只从训练-开发集（官方测试集的前 5000 张）里抽",
                      "目标开发集（后 5000 张）不看错例；如果你拿的是全部 1 万张测试图的下标，先切出前 5000 张")
    return report(name, bool(np.all(yt[idx] != yp[idx])), "抽到的样本里混进了预测正确的图",
                  "先找出预测错的下标 wrong = np.flatnonzero(train_dev_pred != train_dev_y)，再从 wrong 里抽 100 个")


def _tag_lists(tags):
    """把 tags 统一成"每张图一个标签列表"；格式不对返回 None。"""
    if not isinstance(tags, (list, tuple)):
        return None
    out = []
    for t in tags:
        if not isinstance(t, (list, tuple, set)) or not all(isinstance(s, str) for s in t):
            return None
        out.append([s for s in t if s.strip()])          # 空字符串不算标签
    return out


def error_tags(tags, n=100):
    """人工归类的标签：tags[k] 是第 k 张错例的标签列表（字符串，可以有多个）。

    人工标签没法自动判对错，这里只查结构；某一类占了六成以上时提醒细分（Ng：'其他'类太大要拆）。"""
    name = "错例的人工标签"
    lists = _tag_lists(tags)
    if lists is None:
        return report(name, False, "tags 应是列表，每个元素是一张图的标签列表（字符串），例如 [['背景主导'], ['暗', '太小'], ...]",
                      "单个标签也写成列表：['背景主导']")
    if len(lists) != n:
        return report(name, False, f"tags 应有 {n} 个元素（每张错例一个），实际 {len(lists)} 个")
    empty = [k for k, t in enumerate(lists) if not t]
    if empty:
        return report(name, False, f"还有 {len(empty)} 张图没有标签（编号 {empty[:8]}{' …' if len(empty) > 8 else ''}）",
                      "每张图至少一个标签；实在归不进任何一类，就单独记一个'其他'")
    counts = {}
    for t in lists:
        for s in set(t):
            counts[s] = counts.get(s, 0) + 1
    if len(counts) < 3:
        return report(name, False, f"只分出了 {len(counts)} 类——类别定义得太粗，至少分出 3 类",
                      "常见的类别：非典型外观、太小或被截断、背景主导、太暗或对比度低、清晰典型（模型就是错了）")
    top, c = max(counts.items(), key=lambda kv: kv[1])
    if c > 0.6 * n:
        print(f"   注意：『{top}』占了 {c}/{n}，考虑把它再拆细（Ng：'其他'类太大时要细分）")
    return report(name, True)


def _ref_ceiling(tags, error):
    counts = {}
    for t in tags:
        for s in set(t):
            counts[s] = counts.get(s, 0) + 1
    return {s: (c, 100 * error * c / len(tags)) for s, c in counts.items()}


def ceiling_table(fn):
    """学习者写的 ceiling_table(tags, error)（error 是看错例那份数据——训练-开发集——的误差） -> {类别: (张数, 收益上限的百分点)}。"""
    name = "收益上限表 ceiling_table"
    cases = [
        ([["dark"]] * 50 + [["small"]] * 30 + [["small", "dark"]] * 20, 0.44),    # 一张图两个标签
        ([["a"]] * 30 + [["b"]] * 10, 0.5),                                       # 不是 100 张
    ]
    for tags, err in cases:
        want = _ref_ceiling(tags, err)
        ok, out = safe_call(name, fn, [list(t) for t in tags], err)
        if not ok:
            return False
        if not isinstance(out, dict) or set(out) != set(want):
            return report(name, False, f"应返回以类别为键的 dict，键为 {sorted(want)}，实际 {out!r}")
        for k, (c, ceil) in want.items():
            try:
                gc, gceil = int(out[k][0]), float(out[k][1])
            except (TypeError, ValueError, IndexError, KeyError):
                return report(name, False, f"out[{k!r}] 应是 (张数, 收益上限) 两个数，实际 {out[k]!r}")
            if gc != c:
                return report(name, False, f"测试例里 {k!r} 应有 {c} 张，实际 {gc}",
                              "一张图有两个标签时，两类各计一次")
            if not math.isclose(gceil, ceil, abs_tol=0.05):
                hint = "收益上限 = 看错例那份数据（训练-开发集）的误差 × 该类张数 / 错例张数 × 100（以百分点表示）"
                if math.isclose(gceil, ceil / 100, abs_tol=5e-4):
                    hint = "以百分点表示：再乘 100（0.308 → 30.8 个百分点）"
                elif math.isclose(gceil, 100 * err * c / 100, abs_tol=0.05):
                    hint = "分母是错例的张数 len(tags)，不要写死成 100"
                elif math.isclose(gceil, 100 * err * c / sum(len(t) for t in tags), abs_tol=0.05):
                    hint = "分母是错例的张数 len(tags)，不是标签的总个数——一张图两个标签时两类各计一次，比例加起来可以超过 1"
                return report(name, False, f"测试例（误差 {err}）里 {k!r} 的收益上限应为 {ceil:.1f}，实际 {gceil:.3g}", hint)
    return report(name, True)


def mismatch_table(t, n_train=50000):
    """t = {"human", "train", "train_dev", "dev"}（误差，0–1 的小数）与 {"avoidable_bias", "variance", "mismatch"}。
    "dev" 是目标开发集（B 实验室）的误差，键名沿用旧名。

    本机实测（3 个种子，CPU）：训练（train[:10000]，eval 模式）0.421–0.435；训练-开发（干净的前 5000 张）
    0.431–0.453；目标开发（后 5000 张 + 对比度严重度 2）0.606–0.657；人类 0.061（Ho-Phuoc 2018）。"""
    name = "偏差 / 方差 / 数据错配表"
    keys = ("human", "train", "train_dev", "dev", "avoidable_bias", "variance", "mismatch")
    if not isinstance(t, dict) or not all(k in t for k in keys):
        return report(name, False, f"应是含 {list(keys)} 的 dict（误差用 0–1 的小数）")
    try:
        v = {k: float(t[k]) for k in keys}
    except (TypeError, ValueError):
        return report(name, False, "表里的每一项都应是数")
    if any(not 0.0 <= v[k] <= 1.0 for k in keys[:4]):
        return report(name, False, f"误差应是 0–1 的小数，实际 { {k: round(v[k], 3) for k in keys[:4]} }",
                      "误差 = 1 − 准确率；不要写成百分数")
    gaps = {"avoidable_bias": v["train"] - v["human"], "variance": v["train_dev"] - v["train"],
            "mismatch": v["dev"] - v["train_dev"]}
    bad = [k for k in gaps if not math.isclose(gaps[k], v[k], abs_tol=1e-6)]
    if bad:
        return report(name, False, f"{bad} 与表里的误差对不上",
                      "可避免偏差 = 训练 − 人类；方差 = 训练-开发 − 训练；数据错配 = 目标开发 − 训练-开发")
    if not math.isclose(v["human"], 1 - HUMAN_ACCURACY, abs_tol=0.02):
        return report(name, False, f"人类水平误差约 6%（Ho-Phuoc 2018：人类平均准确率 93.91%），实际 {v['human']:.3f}")
    if _subset_note(name, n_train):
        return report(name, True)
    for k, label, lo, hi, what in (("train", "训练", 0.33, 0.52, "训练集前 1 万张，eval 模式"),
                                   ("train_dev", "训练-开发", 0.36, 0.54, "训练-开发集：干净的前 5000 张"),
                                   ("dev", "目标开发", 0.50, 0.75, "目标开发集：后 5000 张加对比度严重度 2")):
        if not lo <= v[k] <= hi:
            return report(name, False, f"{label}误差（键 \"{k}\"）应在 [{lo}, {hi}]（{what}），实际 {v[k]:.3f}",
                          "误差 = 1 − 准确率；predict 之前模型要切到 eval 模式（predict 里已写好）")
    return report(name, True)


def contrast(fn):
    """学习者写的 contrast(u8, c)：与 zero2researcher.shift 里改编自 CIFAR-10-C 的对比度腐蚀逐像素比对（差 ≤ 1）。"""
    from ..shift import SEVERITY, corrupt_cifar

    name = "对比度腐蚀 contrast"
    rng = np.random.default_rng(10)
    x = rng.integers(0, 256, size=(3, 32, 32, 3)).astype(np.uint8)
    x[1] = np.clip(x[1] // 2 + 100, 0, 255)                       # 一张偏亮、对比度本来就低的图
    x[2, ..., 0] = 250                                            # 一张红色通道几乎饱和的图
    for s, c in enumerate(SEVERITY["contrast"], start=1):
        ok, got = safe_call(name, fn, x.copy(), c)
        if not ok:
            return False
        want = corrupt_cifar(x, "contrast", s)
        if not isinstance(got, np.ndarray) or got.shape != x.shape:
            shape = got.shape if isinstance(got, np.ndarray) else type(got).__name__
            return report(name, False, f"应返回与输入同形状 {x.shape} 的数组，实际 {shape}")
        if got.dtype != np.uint8:
            return report(name, False, f"应返回 uint8（0–255），实际 dtype {got.dtype}",
                          "最后乘 255 再转回 uint8：np.uint8(...)，与上游一样直接截断")
        diff = np.abs(got.astype(int) - want.astype(int)).max()
        if diff > 1:
            f = x / 255.0
            hint = ("上游的写法：x = u8 / 255；m = 每张图、每个通道的均值（axis=(1, 2)，keepdims=True）；"
                    "np.uint8(np.clip((x − m) × c + m, 0, 1) × 255)")
            for axis, msg in (((0, 1, 2), "均值要按每张图分别算，不要对整批图求"),
                              ((1, 2, 3), "均值要按每个通道分别算：axis=(1, 2)，不要把 3 个通道混在一起"),
                              (None, "均值要按每张图、每个通道分别算：axis=(1, 2)")):
                m = f.mean(axis=axis, keepdims=True)
                if np.abs((np.clip((f - m) * c + m, 0, 1) * 255).astype(np.uint8).astype(int) - got.astype(int)).max() <= 1:
                    hint = msg
                    break
            return report(name, False, f"c = {c} 时与参考结果最多差 {diff}（允许 ≤ 1）", hint)
    return report(name, True)


def corruption_sweep(results, clean_acc, n_train=50000):
    """results[腐蚀名] = [严重度 1–5 的准确率]，至少含 "contrast"；clean_acc 是训练-开发集（干净）的准确率。

    本机实测（干净 0.547–0.569）：对比度严重度 3 比干净低 24–28 个百分点，严重度 5 比严重度 1 低 33–39 个百分点。"""
    name = "腐蚀扫描"
    if not isinstance(results, dict) or "contrast" not in results:
        return report(name, False, "results 应是 {腐蚀名: [严重度 1–5 的准确率]}，至少包含 'contrast'")
    try:
        rows = {k: np.asarray(v, dtype=float) for k, v in results.items()}
        clean = float(clean_acc)
    except (TypeError, ValueError):
        return report(name, False, "准确率应是数")
    if any(r.shape != (5,) for r in rows.values()):
        return report(name, False, "每种腐蚀都要有 5 个严重度的准确率")
    if any(np.any((r < 0) | (r > 1)) for r in rows.values()) or not 0 <= clean <= 1:
        return report(name, False, "准确率应在 0–1 之间")
    if _subset_note(name, n_train):
        return report(name, True)
    c = rows["contrast"]
    return report(name, c[2] < clean - 0.15 and c[4] < c[0] - 0.2,
                  f"对比度扰动应让准确率明显下降（本机严重度 3 掉约 25 个百分点），实际 {np.round(c, 3).tolist()}，干净 {clean:.3f}",
                  "腐蚀要加在 uint8 原图（0–255）上，再做与训练时相同的 ToTensor + Normalize（predict 里已写好）")


def batch_correction(before, after, n_train=50000):
    """目标开发集（B 实验室：后 5000 张 + 对比度严重度 2）校正前后的准确率。本机实测 0.383 / 0.394 → 0.485 / 0.487（种子 42 / 0）。"""
    name = "批次校正（通道均值与标准差对齐到训练集）"
    try:
        b, a = float(before), float(after)
    except (TypeError, ValueError):
        return report(name, False, "应传入两个数：校正前、校正后的准确率")
    if not (0 <= b <= 1 and 0 <= a <= 1):
        return report(name, False, f"准确率应在 0–1 之间，实际 {b}, {a}")
    if _subset_note(name, n_train):
        return report(name, a > b, f"校正后应有回升，实际 {b:.3f} → {a:.3f}",
                      "每个通道：(x − 这一批的均值) / 这一批的标准差 × 训练集标准差 + 训练集均值")
    return report(name, a - b > 0.05, f"校正后应明显回升（本机约 +10 个百分点），实际 {b:.3f} → {a:.3f}",
                  "对整批图求每个通道的均值和标准差（axis=(0, 1, 2)，在 0–1 尺度上），"
                  "再算 (x − 批次均值) / 批次标准差 × 训练集标准差 + 训练集均值")
