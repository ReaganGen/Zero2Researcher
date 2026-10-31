"""第11周自查函数。用法：from zero2researcher.checks import week11 as ck; ck.scaled_dot_product(scaled_dot_product)

函数/类练习把函数或类本身传进来（不加括号），例如 ck.multihead_shapes(MultiheadAttention)；
其余检查直接传入你算出的数。

参考实现照 UvA DL Tutorial 6 "Transformers and Multi-Head Attention"
（Phillip Lippe, MIT License, https://github.com/phlippe/uvadlc_notebooks ）：
scaled_dot_product 返回 (values, attention) 两个结果；mask 中 1/True = 保留、0/False = 遮住
（与 torch.nn.functional.scaled_dot_product_attention 的布尔 attn_mask 约定一致）。
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


# ---------- 参考实现（UvA Tutorial 6 的 scaled_dot_product，遮挡值改用 -inf） ----------
def _ref_sdp(q, k, v, mask=None):
    import torch
    d_k = q.size(-1)
    logits = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        logits = logits.masked_fill(mask == 0, float("-inf"))
    attention = torch.softmax(logits, dim=-1)
    return attention @ v, attention


def _sdp_cases():
    """(说明, q, k, v, mask)：无 mask / 因果 mask / padding mask，覆盖 3 维与 4 维输入。"""
    import torch
    g = torch.Generator().manual_seed(0)
    q3, k3, v3 = (torch.randn(2, 4, 8, generator=g) for _ in range(3))
    q4, k4, v4 = (torch.randn(2, 3, 5, 4, generator=g) for _ in range(3))
    causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))                  # (T, T)
    keep = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    padding = keep[:, None, None, :]                                          # (B, 1, 1, T)
    return [("无 mask，输入 (B, T, d)", q3, k3, v3, None),
            ("因果 mask，输入 (B, H, T, d)", q4, k4, v4, causal),
            ("padding mask (B, 1, 1, T)", q4, k4, v4, padding)]


def _sdp_hint(fn_values, q, k, v, mask):
    """猜一猜学生最可能犯的错，给出有针对性的提示。"""
    import torch
    d_k = q.size(-1)
    raw = q @ k.transpose(-2, -1)
    m = (lambda t: t) if mask is None else (lambda t: t.masked_fill(mask == 0, float("-inf")))
    guesses = [
        (torch.softmax(m(raw), -1) @ v, "看起来忘了缩放：点积要先除以 √d_k（d_k 是 q、k 最后一维的长度）再进 softmax"),
        (torch.softmax(m(raw / d_k), -1) @ v, "除的是 d_k 而不是 √d_k"),
        (torch.softmax(m(raw / math.sqrt(d_k)), -2) @ v,
         "softmax 的维度错了：应在最后一维（dim=-1，对每个 query 在所有 key 上归一化）"),
    ]
    if mask is not None:
        inv = (raw / math.sqrt(d_k)).masked_fill(mask != 0, float("-inf"))
        guesses.append((torch.softmax(inv, -1) @ v,
                        "mask 的约定反了：1/True 表示保留，应 masked_fill(mask == 0, -inf)"))
        guesses.append((torch.softmax(raw / math.sqrt(d_k), -1) @ v,
                        "mask 没有生效：在 softmax 之前用 masked_fill 把 mask == 0 的位置填成 -inf"))
    for cand, hint in guesses:
        if cand.shape == fn_values.shape and torch.allclose(cand, fn_values, atol=1e-5):
            return hint
    return "对照公式 softmax(QKᵀ/√d_k)·V 逐行检查；用小例子 print 中间的 logits 和 attention"


def scaled_dot_product(fn):
    """学生的 scaled_dot_product(q, k, v, mask=None) → (values, attention)。"""
    import torch
    import torch.nn.functional as F
    name = "scaled_dot_product"
    for desc, q, k, v, mask in _sdp_cases():
        ok, out = safe_call(name, fn, q, k, v, mask)
        if not ok:
            return False
        if not (isinstance(out, (tuple, list)) and len(out) == 2):
            return report(name, False, f"[{desc}] 应返回两个结果 (values, attention)，实际是 {type(out).__name__}",
                          "照 UvA 的写法最后一行 return values, attention")
        values, attention = out
        ref_v, ref_a = _ref_sdp(q, k, v, mask)
        if not (isinstance(values, torch.Tensor) and isinstance(attention, torch.Tensor)):
            return report(name, False, f"[{desc}] values 与 attention 都应是 torch.Tensor")
        if tuple(attention.shape) != tuple(ref_a.shape):
            return report(name, False, f"[{desc}] attention 的 shape 应为 {tuple(ref_a.shape)}，实际 {tuple(attention.shape)}",
                          "q @ k.transpose(-2, -1) 得到 (…, T_q, T_k)；转置的是最后两维")
        if tuple(values.shape) != tuple(ref_v.shape):
            return report(name, False, f"[{desc}] values 的 shape 应为 {tuple(ref_v.shape)}，实际 {tuple(values.shape)}")
        if not torch.allclose(values.float(), ref_v, atol=1e-5):
            return report(name, False, f"[{desc}] 输出与参考实现不一致", _sdp_hint(values.float(), q, k, v, mask))
        if not torch.allclose(attention.float(), ref_a, atol=1e-5):
            return report(name, False, f"[{desc}] values 对了但返回的 attention 不对",
                          "attention 是 softmax 之后、乘 v 之前的那个矩阵")
        # 第二道对照：PyTorch 内置实现（同样的 mask 约定）
        torch_v = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask.bool())
        if not torch.allclose(values.float(), torch_v, atol=1e-5):
            return report(name, False, f"[{desc}] 与 F.scaled_dot_product_attention 不一致")
    return report(name, True)


def masked_softmax(fn):
    """学生的 masked_softmax(scores, mask)：mask 中 1 = 保留、0 = 遮住，遮住的位置概率必须恰好为 0。"""
    import torch
    name = "masked_softmax"
    scores = torch.tensor([[2.0, 1.0, 0.5, -1.0], [0.3, 0.3, 4.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]])
    ref = torch.softmax(scores.masked_fill(mask == 0, float("-inf")), dim=-1)
    for m in (mask, mask.bool()):
        ok, out = safe_call(name, fn, scores.clone(), m)
        if not ok:
            return False
        if not isinstance(out, torch.Tensor) or tuple(out.shape) != (3, 4):
            return report(name, False, "应返回与 scores 同 shape (3, 4) 的 tensor")
        out = out.float()
        if torch.isnan(out).any():
            return report(name, False, "结果里有 NaN", "只有整行都被遮住时才会出现 NaN；检查 mask 的方向是否反了")
        if (out[m == 0].abs() > 1e-7).any():
            return report(name, False, "被遮住的位置概率不是 0",
                          "被遮住的位置要在 softmax **之前**变成 −∞（e^(−∞) = 0）；"
                          "填 0 不行——e⁰ = 1，被遮住的位置仍会分到概率")
        if not torch.allclose(out.sum(-1), torch.ones(3), atol=1e-5):
            return report(name, False, "每一行的概率之和不是 1",
                          "先 softmax 再乘 mask 会让每行不再归一——顺序应该是先遮、再 softmax")
        if not torch.allclose(out, ref, atol=1e-6):
            return report(name, False, "数值与参考实现不一致", "softmax 要沿最后一维（dim=-1）")
    return report(name, True)


def hand_attention(weights, out):
    """课堂练习 E1：第 2 个 token 的注意力权重（3 个数）与输出向量（2 个数）。"""
    name = "手算注意力（token 2）"
    Q = np.array([[0, 1, 0, 1], [2, 0, 2, 0], [1, 0, 0, 1]], dtype=float)
    K = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    V = np.array([[2, 0], [0, 2], [1, 1]], dtype=float)
    s = Q[1] @ K.T / 2.0
    w = np.exp(s) / np.exp(s).sum()
    o = w @ V
    try:
        wv = np.asarray(weights, dtype=float).ravel()
        ov = np.asarray(out, dtype=float).ravel()
    except (TypeError, ValueError):
        return report(name, False, "weights 应是 3 个数、out 应是 2 个数")
    if wv.shape != (3,) or ov.shape != (2,):
        return report(name, False, f"weights 应有 3 个数、out 应有 2 个数，实际 {wv.size} 和 {ov.size}")
    if not np.allclose(wv, w, atol=2e-3):
        unscaled = np.exp(Q[1] @ K.T) / np.exp(Q[1] @ K.T).sum()
        hint = ("看起来没除以 √d = 2" if np.allclose(wv, unscaled, atol=2e-3)
                else "三步：q₂·kⱼ 得分数 [4, 2, 2] → 除以 √4 = 2 → softmax（e 的幂再归一化）")
        return report(name, False, f"权重应约为 {np.round(w, 3).tolist()}，实际 {np.round(wv, 3).tolist()}", hint)
    if not np.allclose(ov, o, atol=2e-3):
        return report(name, False, f"输出应约为 {np.round(o, 3).tolist()}，实际 {np.round(ov, 3).tolist()}",
                      "输出 = Σⱼ wⱼ·vⱼ：用三个权重对 V 的三行做加权平均")
    return report(name, True)


def unscaled_std(std):
    return _in_range("未缩放点积的标准差（d = 256）", std, 15.0, 17.0,
                     "q、k 各分量独立、均值 0、方差 1 时，q·k 是 256 项之和，方差 = 256、标准差 = √256 = 16"
                     "（Vaswani et al. 2017 §3.2.1 脚注）。确认用的是 torch.randn 且没有除以 √d")


# ---------- 多头注意力 ----------
def _ref_mha_from(student, layout, scale="head", permute_in=True, permute_out=True):
    """用学生模块的 qkv_proj / o_proj 权重重算输出。layout='uva'（每个头一段 [q|k|v]）或 'chunk'（先切 qkv）；
    其余参数用来模拟常见错误：scale='embed'（按 √embed_dim 缩放，gpt.py 视频版的写法）、
    permute_in=False（直接 reshape 成 (B, H, T, 3d)）、permute_out=False（合并前不 permute 回去）。"""
    import torch

    def forward(x, mask=None):
        B, T, C = x.shape
        H = student.num_heads
        d = C // H
        qkv = student.qkv_proj(x)
        if layout == "uva":
            qkv = (qkv.reshape(B, T, H, 3 * d).permute(0, 2, 1, 3) if permute_in
                   else qkv.reshape(B, H, T, 3 * d))
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q, k, v = ((t.reshape(B, T, H, d).permute(0, 2, 1, 3) if permute_in else t.reshape(B, H, T, d))
                       for t in qkv.chunk(3, dim=-1))
        logits = q @ k.transpose(-2, -1) / math.sqrt(d if scale == "head" else C)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        vals = attn @ v
        vals = vals.permute(0, 2, 1, 3).reshape(B, T, C) if permute_out else vals.reshape(B, T, C)
        return student.o_proj(vals), attn
    return forward


_MHA_GUESSES = [
    (dict(scale="embed"),
     "缩放用了 √embed_dim（所有头的总维度 C），应除以 √head_dim：每个头的点积只有 head_dim 项相加，"
     "方差是 head_dim（第 11 周 S2 里\"为什么除以 √d\"的推导）"),
    (dict(permute_in=False),
     "拆头时直接 reshape 成 (B, H, T, ·) 了：qkv 的内存顺序是 (B, T, ·)，要先 reshape 成 (B, T, H, ·) "
     "再把 H 和 T 两维交换——否则不同 token 的数被混进了同一个头"),
    (dict(permute_out=False),
     "合并多头时没有先把 (B, H, T, d) 换回 (B, T, H, d)：直接 reshape 成 (B, T, C) 会把不同 token 的结果拼在一起"),
]


def multihead_shapes(cls):
    """学生的 MultiheadAttention(embed_dim, num_heads)：forward(x, mask=None, return_attention=False)。"""
    import torch
    with torch.random.fork_rng(devices=[]):          # 不打乱学生 notebook 里的随机数序列
        torch.manual_seed(0)
        return _multihead_shapes(cls)


def _multihead_shapes(cls):
    import torch
    name = "MultiheadAttention"
    try:
        m = cls(16, 4)
    except NotImplementedError:
        return report(name, False, "__init__ 还没有实现（仍是 NotImplementedError）")
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"构造 MultiheadAttention(16, 4) 时出错 {type(e).__name__}: {e}",
                      "构造参数是 (embed_dim, num_heads)")
    m.eval()
    x = torch.randn(2, 5, 16)
    causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    try:
        with torch.no_grad():
            out = m(x)
            out2, attn = m(x, mask=causal, return_attention=True)
    except NotImplementedError:
        return report(name, False, "forward 还没有实现（仍是 NotImplementedError）")
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"前向时出错 {type(e).__name__}: {e}",
                      "逐行 print(t.shape)：qkv (B,T,3C) → (B,T,H,3d) → permute 成 (B,H,T,3d) → chunk 成 q,k,v")
    if tuple(out.shape) != (2, 5, 16):
        return report(name, False, f"输出 shape 应为 (B, T, C) = (2, 5, 16)，实际 {tuple(out.shape)}",
                      "多头的结果要 permute 回 (B, T, H, d) 再 reshape 成 (B, T, C)，最后过 o_proj")
    if tuple(attn.shape) != (2, 4, 5, 5):
        return report(name, False, f"attention 的 shape 应为 (B, H, T, T) = (2, 4, 5, 5)，实际 {tuple(attn.shape)}")
    if not torch.allclose(attn.sum(-1), torch.ones(2, 4, 5), atol=1e-5):
        return report(name, False, "attention 每一行的和应为 1")
    if (attn[..., ~causal].abs() > 1e-7).any():
        return report(name, False, "传入因果 mask 后，attention 的上三角应全为 0",
                      "mask 要原样传给 scaled_dot_product（(T,T) 会自动广播到 (B,H,T,T)）")
    if not (hasattr(m, "qkv_proj") and hasattr(m, "o_proj") and hasattr(m, "num_heads")):
        print("   注意：没有找到 qkv_proj / o_proj / num_heads 这几个属性（例如用了三个独立的 Linear），"
              "下面只做了形状、行和与因果掩码的检查，没有做数值对照。照 UvA 的命名写，才能得到完整检查。")
        return report(name, True)

    def _matches(**kw):
        for layout in ("uva", "chunk"):
            try:
                with torch.no_grad():
                    ref_out, ref_attn = _ref_mha_from(m, layout, **kw)(x, causal)
            except Exception:  # noqa: BLE001
                continue
            if torch.allclose(out2, ref_out, atol=1e-5) and torch.allclose(attn, ref_attn, atol=1e-5):
                return True
        return False

    if _matches():
        return report(name, True)
    hint = next((h for kw, h in _MHA_GUESSES if _matches(**kw)),
                "逐行 print 形状并对照 UvA：qkv (B,T,3C) → (B,T,H,3d) → 交换 H 与 T → chunk 成 q,k,v → "
                "注意力 → 换回 (B,T,H,d) → reshape (B,T,C) → o_proj；也检查 mask 是否传进了 scaled_dot_product")
    return report(name, False, "shape 都对，但数值与按同一组权重重算的结果不一致", hint)


# ---------- 位置编码与置换等变 ----------
def permutation_equivariance(diff_without_pe, diff_with_pe):
    """两个数：打乱输入顺序后 max|f(x[perm]) - f(x)[perm]|，分别在无 / 有位置编码时测得。"""
    name = "置换等变性验证"
    try:
        a, b = float(diff_without_pe), float(diff_with_pe)
    except (TypeError, ValueError):
        return report(name, False, "应传入两个数：无位置编码时的最大差异、有位置编码时的最大差异")
    if not (np.isfinite(a) and np.isfinite(b)):
        return report(name, False, "差异不是有限数")
    if a > 1e-4:
        return report(name, False, f"无位置编码时差异应≈0（浮点误差 < 1e-4），实际 {a:.2e}",
                      "比较的是 f(x[:, perm]) 和 f(x)[:, perm]——输出也要按同一个 perm 重排；"
                      "并且要 model.eval()（dropout 会引入随机差异）")
    if b < 1e-3:
        return report(name, False, f"加上位置编码后差异应明显 > 0，实际 {b:.2e}",
                      "位置编码要加在打乱**之后**的输入上：f(PE(x[:, perm]))，"
                      "而不是先加 PE 再打乱（那样位置信息跟着 token 一起被搬走了）")
    return report(name, True)


# ---------- 序列反转 ----------
def reversal_accuracy_with_pe(acc):
    return _in_range("序列反转（有位置编码）测试准确率", acc, 0.95, 1.0,
                     "UvA 同一设定可到 100%。检查：标签是 x.flip(1)；损失要把 (B, T, 10) 拉平成 (B*T, 10)；"
                     "位置编码确实加在了 embedding 上；训练至少 3 个 epoch")


def reversal_accuracy_without_pe(acc):
    return _in_range("序列反转（无位置编码）测试准确率", acc, 0.18, 0.30,
                     "无位置编码时模型对位置是置换等变的，最多只能猜'除了自己以外出现最多的数字'，"
                     "理论上限约 0.24（本机实测 0.241；'永远猜整串的众数'约 0.237）。若接近 1.0，说明位置信息还在（检查 use_pe 开关）；"
                     "若接近 0.10，说明模型没学到东西")


# ---------- IMDB ----------
def imdb_transformer_accuracy(acc):
    return _in_range("IMDB 小 Transformer 测试准确率", acc, 0.74, 0.86,
                     "本机实测 2 层 / 2 头 / d_model 64、8 个 epoch 约 0.80–0.82（3 个种子）。"
                     "明显偏低时检查：mean-pool 是否按 padding mask 只平均真实 token；"
                     "注意力的 padding mask 形状是否为 (B, 1, 1, T)；是否在 eval 时调用了 model.eval()")
