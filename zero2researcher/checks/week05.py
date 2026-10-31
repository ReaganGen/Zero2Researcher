"""第5周自查函数。用法：from zero2researcher.checks import week05 as ck; ck.titanic_lr_f1(lr_f1)

多数检查直接传入你算出的数值（如 ck.wrong_cv_error(wrong_err)）。
涉及随机性的实验（交叉验证、ESL 重演）按合理区间检查，不要求和参考实现一字不差。
名字里标着"只核对数值"的检查分辨不出方法错误（例如预处理放在 CV 外，在 Titanic 上几乎不改变分数），
划分与拟合是否正确，请对照讲义自查。ck.titanic_cv_scores 例外：它同时检查你的 Pipeline 和交给 CV 的 X。
"""
import numpy as np

from . import report, close, safe_call
from .. import data


# ---------- 参考实现（教师实现） ----------
_Y_TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
_Y_PRED = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]


def _confusion():
    tp = sum(1 for t, p in zip(_Y_TRUE, _Y_PRED) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(_Y_TRUE, _Y_PRED) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(_Y_TRUE, _Y_PRED) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(_Y_TRUE, _Y_PRED) if t == 0 and p == 0)
    return tp, fp, fn, tn


# 与 05_lecture "准确率什么时候骗人" 一节完全相同的不平衡数据。class_sep=1.0 时两类有重叠：
# 逻辑回归的测试准确率 0.950 恰好等于多数类基线，少数类召回率只有 0.28（class_sep=1.5 时两类几乎线性可分，
# 少数类召回率 0.96，演示不出"准确率失效"）。
_IMB_KW = dict(n_samples=2000, n_features=2, n_informative=2, n_redundant=0,
               n_clusters_per_class=1, weights=[0.95, 0.05], flip_y=0, class_sep=1.0,
               random_state=42)


def _imbalanced_data():
    from sklearn.datasets import make_classification
    return make_classification(**_IMB_KW)


def _majority_acc():
    _, y = _imbalanced_data()
    return max(np.bincount(y)) / len(y)


def _imbalanced_lr_metrics():
    """讲义里那一格的参考结果：(准确率, 少数类召回率, 多数类召回率, macro-F1, ROC-AUC)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    X, y = _imbalanced_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42, stratify=y)
    clf = LogisticRegression().fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return (accuracy_score(yte, pred), recall_score(yte, pred, pos_label=1),
            recall_score(yte, pred, pos_label=0), f1_score(yte, pred, average="macro"),
            roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def _titanic_Xy():
    df = data.load_titanic()
    X = df[["pclass", "age", "fare"]].copy()
    X["sex"] = (df["sex"] == "female").astype(int)
    return X, df["survived"]


def _lr_pipe():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000))


def _rf_pipe():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=100, random_state=42))


def _cv():
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def _flat_steps(est):
    """把 Pipeline（可嵌套、可含 ColumnTransformer）展开成步骤对象的列表。"""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    if isinstance(est, Pipeline):
        return [s for _, step in est.steps for s in _flat_steps(step)]
    if isinstance(est, ColumnTransformer):
        return [s for _, tr, _ in est.transformers for s in _flat_steps(tr)]
    return [est]


def _is_imputer(step):
    return type(step).__module__.startswith("sklearn.impute")


def _is_scaler(step):
    return type(step).__module__.startswith("sklearn.preprocessing") and type(step).__name__.endswith("Scaler")


# GroupKFold 练习（05_exercises E5）：30 只小鼠 × 8 个技术重复孔，标签按小鼠分配、与读数无关
def _mice_data(seed=2027):
    rng = np.random.default_rng(seed)
    mouse_label = rng.permutation(np.array([0, 1] * 15))
    mouse_profile = rng.normal(0, 1, size=(30, 4))
    mouse_id = np.repeat(np.arange(30), 8)
    X = mouse_profile[mouse_id] + rng.normal(0, 0.3, size=(240, 4))
    return X, mouse_label[mouse_id], mouse_id


def _group_split_accs():
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    X, y, groups = _mice_data()
    knn = KNeighborsClassifier(n_neighbors=5)
    by_record = cross_val_score(knn, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    by_group = cross_val_score(knn, X, y, cv=GroupKFold(5, shuffle=True, random_state=0),
                               groups=groups).mean()
    return by_record, by_group


def _titanic_cv_scores(pipe=None, scoring="accuracy"):
    from sklearn.model_selection import cross_val_score
    X, y = _titanic_Xy()
    return cross_val_score(pipe if pipe is not None else _lr_pipe(), X, y,
                           cv=_cv(), scoring=scoring)


def _in_range(name, value, lo, hi, hint=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(value).__name__}")
    return report(name, lo <= v <= hi,
                  f"期望落在 [{lo}, {hi}]，实际 {v:.4f}", hint)


# ---------- 课堂 ----------
def imbalanced_majority_acc(acc):
    # 课上要求"先想一想再算"，所以失败时只给方法、不打印期望值
    return report("多数类基线准确率", close(acc, _majority_acc(), tol=5e-3),
                  f"实际 {acc}，与参考答案不符",
                  "什么都不学、全部预测多数类时，准确率就是多数类样本占全部样本的比例")


def confusion_values(tp, fp, fn, tn):
    exp = _confusion()
    got = (tp, fp, fn, tn)
    wrong = [n for n, g, e in zip(("TP", "FP", "FN", "TN"), got, exp) if g != e]
    return report("混淆矩阵手算", got == exp,
                  f"{'、'.join(wrong)} 数得不对（实际传入 (TP, FP, FN, TN) = {got}）",
                  "逐个对比 y_true 和 y_pred：真实为 1 且预测为 1 的是 TP，真实为 0 但预测为 1 的是 FP；"
                  "四个数加起来应等于样本总数")


def titanic_cv_scores(scores, pipe=None, X=None):
    """检查 Titanic 5 折 CV：ck.titanic_cv_scores(scores, pipe, X)。

    只看分数抓不到"先在全部数据上填充/标准化、再做 CV"——这种泄漏在 Titanic 上几乎不改变分数
    （0.789 vs 0.788）。所以还要检查：① 交给 CV 的 pipe 是 Pipeline，且填充和缩放都在它里面；
    ② 交给 CV 的 X 仍是含缺失值的原始特征（没有在 CV 外先填充）；③ 这 5 个分数确实由 pipe 在 X 上、
    按 StratifiedKFold(5, shuffle=True, random_state=42) 算出。
    """
    name = "Titanic 5 折 CV"
    cv_hint = "cross_val_score(pipe, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))"
    try:
        scores = np.asarray(scores, dtype=float)
    except (TypeError, ValueError):
        return report(name, False, "scores 应是 cross_val_score 返回的 5 个分数", cv_hint)
    if scores.shape != (5,):
        return report(name, False, f"应是 5 个分数，实际 shape {scores.shape}", cv_hint)
    if pipe is None or X is None:
        return report(name, False, "请把交给 cross_val_score 的 Pipeline 和 X 也传进来：ck.titanic_cv_scores(scores, pipe, X)",
                      "只看分数分辨不出预处理是否放在了 CV 外，所以自查要看你的 Pipeline 和 X")
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    leak_note = "分数可能看起来没问题，但这正是本节要防的泄漏——它在 Titanic 上几乎不改变分数，只看分数抓不到"
    if not isinstance(pipe, Pipeline):
        return report(name, False, f"pipe 应是 Pipeline（make_pipeline(...) 的结果），实际是 {type(pipe).__name__}",
                      "把 SimpleImputer、StandardScaler 和 LogisticRegression 一起装进 make_pipeline，"
                      "再把整个 Pipeline 交给 cross_val_score。" + leak_note)
    steps = _flat_steps(pipe)[:-1]   # 最后一步是模型
    if not any(_is_imputer(s) for s in steps) or not any(_is_scaler(s) for s in steps):
        missing = [n for n, ok in (("填充（SimpleImputer）", any(_is_imputer(s) for s in steps)),
                                   ("标准化（StandardScaler）", any(_is_scaler(s) for s in steps))) if not ok]
        return report(name, False, f"Pipeline 里缺少 {'、'.join(missing)} 这一步",
                      "填充和标准化都要放进 Pipeline、排在模型前面，才能保证每一折只用训练折去 fit 它们。" + leak_note)
    Xref, yref = _titanic_Xy()
    try:
        n_rows, n_nan = len(X), int(np.isnan(np.asarray(X, dtype=float)).sum())
    except (TypeError, ValueError):
        return report(name, False, "X 应是交给 cross_val_score 的特征表（DataFrame 或数组）")
    if n_rows != len(yref):
        return report(name, False, f"X 应有 {len(yref)} 行（全部乘客），实际 {n_rows} 行",
                      "不要删行；缺失值交给 Pipeline 里的 SimpleImputer")
    if n_nan == 0:
        return report(name, False, "交给 CV 的 X 里已经没有缺失值了——说明在 CV 之外先做了填充",
                      "把原始的 X（age 有 177 个缺失）直接交给 cross_val_score，填充只在 Pipeline 里做。" + leak_note)
    try:
        redo = cross_val_score(clone(pipe), X, yref, cv=_cv())
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"用你的 pipe 和 X 重算 CV 时出错 {type(e).__name__}: {e}")
    if not np.allclose(scores, redo, atol=1e-6):
        return report(name, False, "这 5 个分数不是用你传入的 pipe 在这个 X 上、按指定的分层 5 折算出来的",
                      "确认 cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)，"
                      "并且传给自查的 pipe、X 就是交给 cross_val_score 的那两个")
    exp = _titanic_cv_scores().mean()
    return report(name, abs(scores.mean() - exp) < 0.02,
                  f"平均准确率与参考实现相差超过 0.02（实际 {scores.mean():.3f}）",
                  "特征用 pclass / age / fare / sex（female=1）；模型用 LogisticRegression(max_iter=1000)")


def wrong_cv_error(err):
    return _in_range("错误流程的 CV 误差", err, 0.0, 0.15,
                     "先在全部 50 个样本上选特征、再做 CV——误差应该低得离谱（这正是泄漏的味道）")


def right_cv_error(err):
    return _in_range("正确流程的 CV 误差", err, 0.35, 0.65,
                     "特征选择放进每一折内部：每折只用训练折的样本选特征，误差应回到 50% 左右")


def group_split_accs(acc_record, acc_group):
    """05_exercises E5：按孔拆（StratifiedKFold）与按小鼠拆（GroupKFold）的 5 折平均准确率。"""
    name = "按组划分（GroupKFold）"
    try:
        a, b = float(acc_record), float(acc_group)
    except (TypeError, ValueError):
        return report(name, False, "应传入两个数：acc_record, acc_group")
    if not 0.80 <= a <= 1.0:
        return report(name, False, f"按孔随机拆的准确率应高得可疑（≥ 0.80），实际 {a:.3f}",
                      "按孔拆时同一只小鼠的孔会同时落进训练折和验证折；"
                      "用 StratifiedKFold(5, shuffle=True, random_state=0)——数据按小鼠顺序排列，不打乱的 cv=5 几乎等于按组拆")
    if not 0.30 <= b <= 0.70:
        return report(name, False, f"按小鼠拆的准确率应回到 0.5 附近（[0.30, 0.70]），实际 {b:.3f}",
                      "用 GroupKFold(5, shuffle=True, random_state=0)，并把 groups=mouse_id 传给 cross_val_score")
    return report(name, a - b >= 0.2, f"两种拆法的差距应明显（≥ 0.2），实际 {a - b:.3f}",
                  "检查两次是否用了同一个模型 KNeighborsClassifier(n_neighbors=5)")


# ---------- 作业 ----------
def titanic_lr_f1(mean):
    return _in_range("逻辑回归 macro-F1（只核对数值）", mean, 0.72, 0.82,
                     'cross_validate(lr_pipe, X, y, cv=cv, scoring="f1_macro")，取 test_score 的均值')


def titanic_rf_f1(mean):
    return _in_range("随机森林 macro-F1（只核对数值）", mean, 0.76, 0.88,
                     "随机森林用同一套特征与同一个 cv；n_estimators=100, random_state=42")


def leak_outside_acc(acc):
    return _in_range("CV 外选特征的准确率", acc, 0.65, 1.0,
                     "SelectKBest 先在全部数据上 fit，再把选出的特征交给 cross_val_score——结果应明显虚高")


def leak_inside_acc(acc):
    return _in_range("CV 内选特征的准确率", acc, 0.35, 0.65,
                     "把 SelectKBest 放进 Pipeline，再把整个 Pipeline 交给 cross_val_score——随机数据上应接近 0.5")
