"""Small helpers for the course's "minimal experiment unit" (week 6 onwards).

课程不教但绕不开的部分：固定随机种子、汇总多种子结果。模型与实验本身由学习者自己写。
"""
import random

import numpy as np


def set_seed(seed: int):
    """Fix Python, NumPy and (if installed) PyTorch random seeds in one call."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def mean_std(values):
    """Return (mean, std) of a list of per-seed results.

    Uses the sample standard deviation (ddof=1), the convention for
    "均值 ± 标准差" across repeated runs; returns std 0.0 for a single value.
    """
    arr = np.asarray(list(values), dtype=float)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def format_result(values, digits=4):
    """Format per-seed results as the course's standard '均值 ± 标准差 (n=k)' string."""
    values = list(values)
    m, s = mean_std(values)
    return f"{m:.{digits}f} ± {s:.{digits}f} (n={len(values)})"
