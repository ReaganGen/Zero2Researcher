"""第6周自查函数。用法：from zero2researcher.checks import week06 as ck; ck.kmeans_ari(ari)

函数类练习把函数本身传进来（不加括号），例如 ck.numerical_gradient(numerical_gradient)；
其余检查直接传入你算出的数值。涉及随机性的结果按合理区间检查。
名字里标着"只核对数值"的检查分辨不出方法错误（例如向量化器在 train + test 上 fit，基线阶梯的分数几乎不变），
划分与拟合是否正确，请对照讲义自查。埋雷练习的 ck.leak_fixed 例外：它逐项检查三处修复，而不只看准确率。
"""
import functools
import math

import numpy as np

from . import report, close, safe_call
from .. import data


# ---------- 参考实现（教师实现） ----------
def _digits():
    from sklearn.datasets import load_digits
    d = load_digits()
    return d.data, d.target


def _digits_pca_evr():
    from sklearn.decomposition import PCA
    X, _ = _digits()
    return PCA().fit(X).explained_variance_ratio_


def _kmeans_ari():
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    X, y = _digits()
    km = KMeans(n_clusters=10, n_init=10, random_state=42).fit(X)
    return adjusted_rand_score(y, km.labels_)


def _pca_dims_accs(dims=(5, 10, 20, 40)):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    X, y = _digits()
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0, stratify=y)
    accs = []
    for d in dims:
        pipe = make_pipeline(PCA(n_components=d, random_state=0),
                             LogisticRegression(max_iter=2000))
        accs.append(pipe.fit(Xtr, ytr).score(Xte, yte))
    return accs


def _num_grad(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


@functools.lru_cache(maxsize=1)
def _leak_data_cached():
    """与 06_exercises 埋雷练习"师兄的数据准备"一格完全相同：返回 (train_df, 污染后的 test_df)。"""
    import pandas as pd
    rng = np.random.default_rng(2026)
    train_df = data.load_imdb("train")
    test_df = data.load_imdb("test")
    src = rng.choice(len(train_df), size=300, replace=False)
    dst = rng.choice(len(test_df), size=300, replace=False)
    keep = np.setdiff1d(np.arange(len(test_df)), dst)
    test_df = pd.concat([test_df.iloc[keep], train_df.iloc[src][["text", "label"]]],
                        ignore_index=True)
    train_df = train_df.sort_values("label", kind="stable").reset_index(drop=True)
    test_df = test_df.sort_values("label", kind="stable").reset_index(drop=True)
    train_df["sample_id"] = np.arange(len(train_df))
    test_df["sample_id"] = np.arange(len(test_df))
    return train_df, test_df


def _leak_data():
    train_df, test_df = _leak_data_cached()
    return train_df.copy(), test_df.copy()


@functools.lru_cache(maxsize=1)
def _leak_reference_acc():
    """三处全修后的参考准确率（本机 sklearn 1.9.1：0.8635，测试样本 1700 条）。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    train_df, test_df = _leak_data_cached()
    clean = test_df[~test_df["text"].isin(set(train_df["text"]))]
    vec = TfidfVectorizer(min_df=2).fit(train_df["text"])
    clf = LogisticRegression(max_iter=1000).fit(vec.transform(train_df["text"]), train_df["label"])
    return float(clf.score(vec.transform(clean["text"]), clean["label"]))


def _in_range(name, value, lo, hi, hint=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(value).__name__}")
    return report(name, lo <= v <= hi,
                  f"期望落在 [{lo}, {hi}]，实际 {v:.4f}", hint)


# ---------- 课堂 ----------
def pca_evr(evr2):
    evr2 = np.asarray(evr2, dtype=float).ravel()
    if evr2.shape != (2,):
        return report("前两个主成分解释方差比", False, f"应传入前 2 个数，实际 shape {evr2.shape}",
                      "PCA().fit(X).explained_variance_ratio_[:2]")
    exp = _digits_pca_evr()[:2]
    return report("前两个主成分解释方差比", np.allclose(evr2, exp, atol=5e-3),
                  f"期望约 {np.round(exp, 3)}，实际 {np.round(evr2, 3)}",
                  "对 digits 的 64 维原始像素（不要标准化）做 PCA().fit(X)")


def kmeans_ari(ari):
    # 上限放宽到 0.80：省略 n_init（sklearn 默认 n_init="auto"，k-means++ 只初始化 1 次）配 random_state=42
    # 时 ARI = 0.726，是合法写法；下限 0.60 仍能把"先标准化"（0.534）判错。
    return _in_range("k-means 与真实标签的 ARI", ari, 0.60, 0.80,
                     "KMeans(n_clusters=10, n_init=10, random_state=42) 作用在原始像素上；"
                     "如果你先做了 StandardScaler，ARI 会掉到 0.53 左右——本课固定用原始像素")


def analytic_derivative(value):
    """06_exercises E1：g(x) = x³ + 2x 在 x = 2 处的手推导数。失败时只给方法，不给答案。"""
    exp = 3 * 2.0 ** 2 + 2
    try:
        v = float(value)
    except (TypeError, ValueError):
        return report("g'(2) 手推", False, f"应传入一个数，实际是 {type(value).__name__}")
    return report("g'(2) 手推", abs(v - exp) < 1e-9, f"实际 {v}，与解析答案不符",
                  "先分别对 x³ 和 2x 求导、相加得到 g'(x)，再代入 x = 2；"
                  "可以拿你写的 numerical_gradient 的结果来核对")


def numerical_gradient(fn):
    cases = [
        (lambda x: x ** 2, 3.0, 6.0),
        (lambda x: (3 * x + 2) ** 2, 1.0, 30.0),
        (math.sin, 0.5, math.cos(0.5)),
    ]
    for f, x, exp in cases:
        ok, got = safe_call("numerical_gradient", fn, f, x)
        if not ok:
            return False
        if not close(got, exp, tol=1e-3):
            return report("numerical_gradient", False,
                          f"在 x={x} 处期望约 {exp:.4f}，实际 {got}",
                          "中心差分：(f(x + h) - f(x - h)) / (2 * h)，h 取 1e-5 左右")
    return report("numerical_gradient", True)


def baseline_ladder(majority, bow, tfidf):
    checks = [
        ("多数类基线", majority, 0.49, 0.51,
         '预测训练集里最多的那一类，再到测试集上打分：DummyClassifier(strategy="most_frequent")'),
        ("词袋 + 逻辑回归", bow, 0.84, 0.87,
         "CountVectorizer(min_df=2) 只在训练集上 fit；LogisticRegression(max_iter=1000)"),
        ("TF-IDF + 逻辑回归", tfidf, 0.85, 0.88,
         "TfidfVectorizer(min_df=2) 只在训练集上 fit"),
    ]
    for name, v, lo, hi, hint in checks:
        if not _in_range(name, v, lo, hi, hint):
            return False
    return report("基线阶梯（只核对数值）", float(tfidf) >= float(bow) - 0.005,
                  f"TF-IDF（{float(tfidf):.4f}）不应明显低于词袋（{float(bow):.4f}）",
                  "检查两级是否用了同一份训练/测试划分与同一个分类器")


def leak_inflated_accuracy(acc):
    return _in_range("污染流水线的测试准确率", acc, 0.95, 1.0,
                     "直接运行给出的流水线即可——它应高得不像话（约 0.99），这正是问题所在")


# ---------- 埋雷排查：逐项检查三处修复 ----------
# 只看准确率不够：只去掉 sample_id、另外两处都不修，准确率 0.8785，也落在"合理"区间里；
# 向量化器在 train + test 上 fit 这一处，修与不修的准确率几乎相同（0.8618 / 0.8635 vs 0.8635）。
# 所以 leak_fixed 分别检查：① 分类器的输入只有词表特征；② 评估用的测试集与训练集没有重复、且没有多删；
# ③ 向量化器的词表与 idf 与"同样参数、只在训练集上 fit"一致；④ fixed_acc 确由这三个部件算出、且接近参考值。
# 提示只指向"症状 → 下一步该查什么"，不直接说出修法（答案在教师版的参考答案格）。
# 同一项第 1 次没通过给第 1 级提示，之后给第 2 级（一个诊断实验）。
_LEAK_HINTS = {
    "features": ("除了影评文本，模型还在用别的信息。把 make_features 拼出来的每一列都列出来，逐列问自己："
                 "部署时来了一条新影评，这一列的值从哪里来？它和标签之间有没有不该有的关系？",
                 "诊断实验：只用那一列额外特征训练一个逻辑回归，看它单独能拿到多少测试准确率；"
                 "再回到数据准备那一格，看这一列是在什么操作之后生成的"),
    "test": ("评估用的测试集还不'干净'：合格的测试集里，不应有模型训练时已经见过的样本。"
             "回到数据准备那一格，逐行读测试集是怎么拼出来的",
             "诊断实验：数一数 test_ok 里有多少条影评在 train_df 里一字不差地出现过（两个 set 求交集）。"
             "这些样本上的'测试成绩'衡量的是什么？"),
    "vec": ("仍有测试集的信息流进了训练流程：检查流水线里每一个调用了 fit 的地方，它看到的是哪些数据？"
            "（第 5 周的底线：凡是从数据里学参数的步骤，都只能看训练数据）",
            "诊断实验：把 len(vec_ok.vocabulary_) 和一个同样参数、只在 train_df[\"text\"] 上 fit 的向量化器的词表大小比一比"),
}
_leak_failures = {}


def _leak_hint(which):
    level = min(_leak_failures.get(which, 0), 1)
    _leak_failures[which] = _leak_failures.get(which, 0) + 1
    return f"（第 {level + 1} 级提示）" + _LEAK_HINTS[which][level]


def leak_fixed(test_ok, vec_ok, clf_ok, fixed_acc):
    """埋雷练习任务 2：ck.leak_fixed(test_ok, vec_ok, clf_ok, fixed_acc)。

    test_ok：最终用来评估的测试集（DataFrame，含 text、label 两列）；vec_ok：修复后流水线里的向量化器；
    clf_ok：修复后的分类器；fixed_acc：clf_ok 在 test_ok 上的测试准确率。
    """
    name = "埋雷修复"
    from sklearn.base import clone
    train_df, test_df = _leak_data_cached()
    if not (hasattr(test_ok, "columns") and {"text", "label"} <= set(test_ok.columns)):
        return report(name, False, "test_ok 应是最终用来评估的测试集 DataFrame（含 text 和 label 两列）")
    vocab = getattr(vec_ok, "vocabulary_", None)
    if not isinstance(vocab, dict):
        return report(name, False, "vec_ok 应是已经 fit 过的向量化器（有 vocabulary_ 属性）")
    n_feat = getattr(clf_ok, "n_features_in_", None)
    if n_feat is None:
        return report(name, False, "clf_ok 应是已经 fit 过的分类器")
    try:
        acc = float(fixed_acc)
    except (TypeError, ValueError):
        return report(name, False, f"fixed_acc 应是一个数，实际是 {type(fixed_acc).__name__}")

    # ① 特征
    if n_feat != len(vocab):
        return report(f"{name} · 检查 1/3", False,
                      f"分类器的输入有 {n_feat} 列，比向量化器的词表（{len(vocab)} 个词）多 {n_feat - len(vocab)} 列",
                      _leak_hint("features"))
    report(f"{name} · 检查 1/3", True)

    # ② 测试集
    texts = test_ok["text"]
    foreign = int((~texts.isin(set(test_df["text"]))).sum())
    if foreign:
        return report(f"{name} · 检查 2/3", False, f"test_ok 里有 {foreign} 条影评不属于给定的测试集",
                      "只能在给定的测试集上修改，不要换成别的数据")
    train_texts = set(train_df["text"])
    overlap = int(texts.isin(train_texts).sum())
    if overlap:
        return report(f"{name} · 检查 2/3", False,
                      f"test_ok 的 {len(test_ok)} 条样本中，仍有 {overlap} 条在训练集 train_df 里出现过",
                      _leak_hint("test"))
    n_expected = int((~test_df["text"].isin(train_texts)).sum())
    if len(test_ok) != n_expected:
        return report(f"{name} · 检查 2/3", False,
                      f"test_ok 只剩 {len(test_ok)} 条样本——删多了",
                      "只去掉真正有问题的测试样本，其余测试样本都要保留")
    report(f"{name} · 检查 2/3", True)

    # ③ 向量化器
    try:
        ref_vec = clone(vec_ok).fit(train_df["text"])
    except Exception as e:  # noqa: BLE001
        return report(f"{name} · 检查 3/3", False, f"无法复制并重新拟合 vec_ok：{type(e).__name__}: {e}",
                      "vec_ok 请用 sklearn 的 TfidfVectorizer / CountVectorizer")
    same = vocab == ref_vec.vocabulary_
    if same and hasattr(ref_vec, "idf_"):
        same = hasattr(vec_ok, "idf_") and np.allclose(vec_ok.idf_, ref_vec.idf_)
    if not same:
        return report(f"{name} · 检查 3/3", False,
                      f"vec_ok 学到的词表 / idf 与'只看训练数据'时应得到的不一致"
                      f"（{len(vocab)} 个词 vs {len(ref_vec.vocabulary_)} 个词）",
                      _leak_hint("vec"))
    report(f"{name} · 检查 3/3", True)

    # ④ 准确率
    try:
        actual = float(clf_ok.score(vec_ok.transform(texts), test_ok["label"]))
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"用 clf_ok、vec_ok 在 test_ok 上重算准确率时出错 {type(e).__name__}: {e}")
    if abs(actual - acc) > 1e-6:
        return report(name, False, f"fixed_acc（{acc:.4f}）与 clf_ok 在 test_ok 上的实际准确率（{actual:.4f}）不一致",
                      "fixed_acc = clf_ok.score(vec_ok.transform(test_ok[\"text\"]), test_ok[\"label\"])")
    ref = _leak_reference_acc()
    return report(name, abs(acc - ref) <= 0.01,
                  f"三处都修好了，但准确率 {acc:.4f} 与参考实现相差超过 0.01",
                  "只修泄漏，模型设置保持师兄原来的：TfidfVectorizer(min_df=2)、LogisticRegression(max_iter=1000)")


# ---------- 作业 ----------
def pca_dims_accuracy(accs):
    accs = np.asarray(accs, dtype=float).ravel()
    if accs.shape != (4,):
        return report("PCA 维数 vs 准确率", False, f"应是 4 个准确率（维数 5/10/20/40），实际 shape {accs.shape}")
    if not np.all(np.diff(accs) >= -0.005):
        return report("PCA 维数 vs 准确率", False, f"保留的维数越多准确率应不降，实际 {np.round(accs, 3)}",
                      "检查四次实验是否用了同一份 train_test_split(random_state=0, stratify=y) 的划分")
    exp = np.array(_pca_dims_accs())
    return report("PCA 维数 vs 准确率", np.allclose(accs, exp, atol=0.03),
                  f"期望约 {np.round(exp, 3)}（容差 ±0.03），实际 {np.round(accs, 3)}",
                  "Pipeline(PCA(n_components=d, random_state=0), LogisticRegression(max_iter=2000))，"
                  "划分固定 train_test_split(random_state=0, stratify=y)")
