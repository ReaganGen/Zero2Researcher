"""第12周自查函数。用法：from zero2researcher.checks import week12 as ck; ck.bigram_nll(nll)

函数类练习把函数本身传进来（不加括号），例如 ck.get_batch_shapes(get_batch)；
其余检查直接传入你算出的数或对象。训练类结果按本机实测 ± 余量的区间检查。

参考：bigram 部分照 Karpathy makemore 第 1 讲（https://github.com/karpathy/makemore ，MIT）；
字符级 GPT 部分照 karpathy/ng-video-lecture 的 gpt.py（https://github.com/karpathy/ng-video-lecture ，MIT）。
torch 只在函数内部 import，保证不装 torch 也能 import 本模块。
"""
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


def _charsets():
    """两份语料的字符表：Tiny Shakespeare（65 个字符）与 names.txt（26 个字母 + '.' = 27）。"""
    from .. import data
    shakespeare = sorted(set(data.load_shakespeare()))
    names = ["."] + sorted(set("".join(data.load_baby_names())))
    return {"Tiny Shakespeare": shakespeare, "names.txt（含 '.'）": names}


# ---------- 字符级分词 ----------
def stoi_itos(stoi, itos):
    """stoi: 字符 → 整数；itos: 整数 → 字符。两者互逆，编号从 0 连续，按 sorted 排序。"""
    name = "stoi / itos"
    if not isinstance(stoi, dict) or not isinstance(itos, dict):
        return report(name, False, "stoi 和 itos 都应是 dict",
                      "stoi = {ch: i for i, ch in enumerate(chars)}；itos = {i: ch for i, ch in enumerate(chars)}")
    if sorted(stoi.values()) != list(range(len(stoi))):
        return report(name, False, f"stoi 的编号应是 0 到 {len(stoi) - 1} 的连续整数")
    if any(itos.get(i) != ch for ch, i in stoi.items()) or len(itos) != len(stoi):
        return report(name, False, "itos 不是 stoi 的逆映射", "对每个 ch：itos[stoi[ch]] == ch")
    for corpus, chars in _charsets().items():
        if set(stoi) == set(chars):
            if [itos[i] for i in range(len(itos))] != chars:
                return report(name, False, f"字符集是 {corpus} 的，但编号顺序不是 sorted 顺序",
                              "chars = sorted(set(text))——排好序，编号才可复现")
            return report(name, True)
    return report(name, False, f"字符表有 {len(stoi)} 个字符，与课程语料都对不上",
                  "Tiny Shakespeare 应有 65 个字符：chars = sorted(set(text))")


def encode_decode(encode, decode):
    """encode: str → list[int]；decode: list[int] → str；decode(encode(s)) == s。"""
    name = "encode / decode"
    chars = _charsets()["Tiny Shakespeare"]
    ref = {ch: i for i, ch in enumerate(chars)}
    for s in ["hii there", "First Citizen:\nBefore we proceed any further, hear me speak.", "ROMEO!"]:
        ok, ids = safe_call(name, encode, s)
        if not ok:
            return False
        try:
            ids = [int(i) for i in ids]
        except (TypeError, ValueError):
            return report(name, False, "encode 应返回整数列表")
        if ids != [ref[c] for c in s]:
            return report(name, False, f"encode({s[:12]!r}…) 的编号与 sorted 字符表不一致",
                          "encode = lambda s: [stoi[c] for c in s]；例：encode('hii there') == [46, 47, 47, 1, 58, 46, 43, 56, 43]")
        ok, back = safe_call(name, decode, ids)
        if not ok:
            return False
        if back != s:
            return report(name, False, f"decode(encode(s)) 没有还原出原字符串：得到 {str(back)[:20]!r}",
                          "decode = lambda l: ''.join([itos[i] for i in l])")
    return report(name, True)


# ---------- bigram ----------
def uniform_nll(nll, vocab_size=27):
    name = f"均匀分布基线（{vocab_size} 个字符）"
    exp = math.log(vocab_size)
    try:
        v = float(nll)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(nll).__name__}")
    return report(name, abs(v - exp) < 1e-3, f"期望 ln({vocab_size}) = {exp:.4f}，实际 {v:.4f}",
                  f"每个字符的概率都是 1/{vocab_size}，NLL = −ln(1/{vocab_size}) = ln({vocab_size})")


def bigram_nll(nll, P=None):
    """计数 bigram 的平均 NLL；传入 P（27 × 27 的概率矩阵）时还检查它确实按**行**归一化。

    为什么要单独查 P：忘了 keepdim 的广播 bug 让每一列除以了某一行的和，可是在 names 数据上
    NLL 与正确写法在数学上完全相等（每个名字以 '.' 开头、以 '.' 结尾，每个字符作为"前一个"
    出现的次数恰好等于它作为"后一个"出现的次数），只看 NLL 永远抓不到它。"""
    ok = _in_range("计数 bigram 的平均 NLL（names，+1 平滑）", nll, 2.425, 2.485,
                   "本机实测 2.4546 nats（228,146 个 bigram）。检查：每个名字前后都要加 '.'；"
                   "平滑是 N + 1 再按行归一化；NLL = −mean(log P[a, b])，用自然对数")
    if P is None or not ok:
        return ok
    return row_normalized(P)


def row_normalized(P):
    """bigram 概率矩阵 P[a, b] = P(下一个 = b | 当前 = a)：形状 27 × 27、非负、每一**行**之和为 1。"""
    name = "bigram 概率矩阵按行归一化"
    try:
        M = np.asarray(P.detach().cpu() if hasattr(P, "detach") else P, dtype=float)
    except (TypeError, ValueError):
        return report(name, False, f"P 应是 27 × 27 的 tensor 或数组，实际是 {type(P).__name__}")
    if M.shape != (27, 27):
        return report(name, False, f"P 的形状应为 (27, 27)，实际 {M.shape}")
    if not np.all(np.isfinite(M)) or (M < 0).any():
        return report(name, False, "P 里有负数或非有限数（nan / inf）",
                      "概率必须在 [0, 1] 之间；检查平滑量与除法")
    rows, cols = M.sum(axis=1), M.sum(axis=0)
    if np.allclose(rows, 1.0, atol=1e-4):
        return report(name, True)
    if np.allclose(cols, 1.0, atol=1e-4):
        return report(name, False, f"每一列之和是 1，每一行却不是（前 3 行之和 {np.round(rows[:3], 4).tolist()}）",
                      "你归一化的方向反了：P[a] 这一行才是\"当前字符为 a 时下一个字符的分布\"。"
                      "回顾讲义里的广播 bug：除数的形状是 (27,) 还是 (27, 1)？")
    return report(name, False, f"每一行之和应为 1，实际前 3 行之和 {np.round(rows[:3], 4).tolist()}",
                  "P = 计数（加平滑后）除以它所在行的行和")


def neural_bigram_nll(nll):
    return _in_range("神经 bigram 的平均 NLL（梯度下降后）", nll, 2.45, 2.55,
                     "应逼近计数解 2.45：one-hot(xs) @ W → exp → 按行归一化 → 取正确字符的 −log 均值；"
                     "学习率 50、至少 100 步；记得每步 W.grad = None")


# ---------- get_batch 与因果掩码 ----------
def get_batch_shapes(fn):
    """学生的 get_batch(data, block_size, batch_size) → (x, y)，y 是 x 右移一位。"""
    import torch
    with torch.random.fork_rng(devices=[]):          # 不打乱学生 notebook 里的随机数序列
        torch.manual_seed(0)
        return _get_batch_shapes(fn)


def _get_batch_shapes(fn):
    import torch
    name = "get_batch"
    data = torch.arange(1000)
    starts = []
    for block, batch in [(8, 4), (16, 3)]:
        ok, out = safe_call(name, fn, data, block, batch)
        if not ok:
            return False
        if not (isinstance(out, (tuple, list)) and len(out) == 2):
            return report(name, False, "应返回两个 tensor：x, y")
        x, y = out
        if tuple(x.shape) != (batch, block) or tuple(y.shape) != (batch, block):
            return report(name, False,
                          f"block_size={block}, batch_size={batch} 时 x、y 的 shape 都应是 ({batch}, {block})，"
                          f"实际 {tuple(x.shape)} 与 {tuple(y.shape)}",
                          "torch.stack 把 batch_size 个长度为 block_size 的片段叠成 (B, T)")
        x, y = x.cpu(), y.cpu()
        if not torch.equal(x[:, 1:] - x[:, :-1], torch.ones(batch, block - 1, dtype=x.dtype)):
            return report(name, False, "x 的每一行应是数据里一段连续的片段 data[i : i + block_size]")
        if not torch.equal(y, x + 1):
            return report(name, False, "y 应是 x 右移一位：y = data[i + 1 : i + block_size + 1]",
                          "语言模型的目标是每个位置的'下一个字符'")
        starts += x[:, 0].tolist()
    if len(set(starts)) == 1:
        return report(name, False, "每次取到的片段起点都一样", "起点要随机：ix = torch.randint(len(data) - block_size, (batch_size,))")
    # 长度 10 的数据、block_size 9：唯一合法的起点是 0（y 要用到最后一个元素）
    tiny = torch.arange(10)
    bound_hint = "起点的范围应是 torch.randint(len(data) - block_size, (batch_size,))，保证 y 不越界"
    for _ in range(20):
        try:
            x, y = fn(tiny, 9, 4)
        except Exception as e:  # noqa: BLE001
            return report(name, False, f"数据很短时出错 {type(e).__name__}: {e}", bound_hint)
        if not (torch.equal(x.cpu(), torch.arange(9).expand(4, 9))
                and torch.equal(y.cpu(), torch.arange(1, 10).expand(4, 9))):
            return report(name, False, "起点越界：片段读到了数据末尾之外", bound_hint)
    return report(name, True)


def causal_mask(mask):
    """T×T 的下三角矩阵：mask[i, j] = 1 当且仅当 j ≤ i（只能看过去和自己）。"""
    name = "因果掩码"
    try:
        m = np.asarray(mask.detach().cpu() if hasattr(mask, "detach") else mask, dtype=float)
    except (TypeError, ValueError):
        return report(name, False, "应传入一个 T×T 的矩阵（tensor 或数组）")
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2:
        return report(name, False, f"应是方阵 (T, T)，实际 shape {m.shape}")
    exp = np.tril(np.ones_like(m))
    if np.array_equal(m, exp):
        return report(name, True)
    if np.array_equal(m, np.triu(np.ones_like(m))):
        return report(name, False, "方向反了：这是上三角", "torch.tril(torch.ones(T, T))——第 i 行只保留前 i+1 个位置")
    if np.array_equal(m, np.triu(np.ones_like(m), 1)):
        return report(name, False, "这是'要遮住的位置'（严格上三角）", "按课程约定传入保留位置：torch.tril(torch.ones(T, T))")
    return report(name, False, "不是下三角的 0/1 矩阵", "torch.tril(torch.ones(T, T))")


# ---------- 字符级 GPT ----------
def tiny_gpt_val_loss(loss):
    return _in_range("tiny GPT 验证集 loss（Tiny Shakespeare，CPU 配置 2000 步）", loss, 1.73, 1.89,
                     "本机实测 1.80–1.82（4 个种子，gpt.py 的 N(0, 0.02) 初始化）。"
                     "基线阶梯：uniform 4.174 → bigram 2.482 → tiny GPT ≈ 1.81。"
                     "明显偏高时检查：位置 embedding 是否加上；因果 mask 是否用 tril；学习率是否为 1e-3；"
                     "是否调用了 self.apply(self._init_weights)")


def ablation_results(results):
    """结构检查：{条件名: [种子1, 种子2, 种子3] 的验证 loss}，至少两个条件、每个至少 3 个种子。"""
    name = "消融结果的结构"
    if not isinstance(results, dict) or len(results) < 2:
        return report(name, False, "应是一个 dict，至少两个条件（如 {'with PE': [...], 'no PE': [...]}）")
    for cond, vals in results.items():
        try:
            arr = np.asarray(list(vals), dtype=float)
        except (TypeError, ValueError):
            return report(name, False, f"条件 {cond!r} 的值应是一组数（每个种子一个验证 loss）")
        if arr.size < 3:
            return report(name, False, f"条件 {cond!r} 只有 {arr.size} 个结果，至少需要 3 个种子")
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0) or np.any(arr > 6):
            return report(name, False, f"条件 {cond!r} 的值不像验证 loss（应是 0–6 之间的有限数）：{arr.round(4).tolist()}")
        if np.ptp(arr) == 0:
            return report(name, False, f"条件 {cond!r} 的 {arr.size} 个种子结果完全相同",
                          "每次训练前要用不同的种子 experiments.set_seed(seed)，否则等于只跑了一次")
    return report(name, True)


def novelty(frac):
    """生成的名字中不在训练集里的比例（0–1）。"""
    name = "生成名字的新颖度"
    try:
        v = float(frac)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个比例（0–1 之间的数），实际是 {type(frac).__name__}")
    if not np.isfinite(v) or v < 0 or v > 1:
        return report(name, False, f"比例应在 0 到 1 之间，实际 {v}",
                      "传入的是比例（如 0.6），不是百分数或个数：np.mean([n not in train_set for n in generated])")
    if v == 0:
        return report(name, False, "新颖度为 0：生成的名字全部出现在训练集里",
                      "检查是否把训练集样本当成了生成结果；或比较时用错了集合")
    return report(name, True)
