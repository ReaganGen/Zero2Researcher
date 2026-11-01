"""第3周自查函数。用法：from zero2researcher.checks import week03 as ck; ck.survival_by_sex(result)

数值类练习把计算结果传进来（dict 或 groupby 得到的 Series 均可）；
函数/类练习把函数或类本身（不加括号）传进来，例如 ck.bank_account(BankAccount)。
需要自己数、自己找的题（如头等舱女性人数），期望值在检查函数内部计算，失败时只给方法，不打印答案。
"""
import numpy as np

from . import report, close, safe_call
from .. import data


# ---------- 参考实现 ----------
def _titanic():
    return data.load_titanic()


def _penguins():
    return data.load_penguins()


def _as_dict(obj):
    """把 dict / pandas Series 统一转换成普通 dict；转不了返回 None。"""
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj.items())
    except AttributeError:
        return None


def _compare_dict(name, got, exp, hint, tol=1e-3):
    if got is None:
        return report(name, False, "应传入 dict 或 groupby 得到的 Series", hint)
    if set(got) != set(exp):
        return report(name, False, f"键应为 {sorted(exp)}，实际 {sorted(got)}", hint)
    bad = [k for k in exp if not close(got[k], exp[k], tol=tol)]
    if bad:
        return report(name, False, f"{bad[0]!r} 的数值不对（期望约 {exp[bad[0]]:.3f}）", hint)
    return report(name, True)


# ---------- 讲义/作业：Pandas ----------
def survival_by_sex(obj):
    exp = _titanic().groupby("sex")["survived"].mean().to_dict()
    return _compare_dict("按性别的生还率", _as_dict(obj), exp,
                         'titanic.groupby("sex")["survived"].mean()；survived 列取均值就是生还率')


def survival_by_class(obj):
    exp = {int(k): v for k, v in _titanic().groupby("pclass")["survived"].mean().items()}
    got = _as_dict(obj)
    if got is not None:
        try:
            got = {int(k): v for k, v in got.items()}
        except (TypeError, ValueError):
            got = None
    return _compare_dict("按舱位的生还率", got, exp,
                         'titanic.groupby("pclass")["survived"].mean()')


def survival_by_sex_class(obj):
    exp = {(str(k[0]), int(k[1])): v
           for k, v in _titanic().groupby(["sex", "pclass"])["survived"].mean().items()}
    got = _as_dict(obj)
    if got is not None:
        try:
            got = {(str(k[0]), int(k[1])): v for k, v in got.items()}
        except (TypeError, ValueError, IndexError):
            got = None
    return _compare_dict("按性别×舱位的生还率", got, exp,
                         'titanic.groupby(["sex", "pclass"])["survived"].mean()，键是 (性别, 舱位) 元组')


def first_class_female(n):
    name = "头等舱女性人数"
    t = _titanic()
    exp = len(t.loc[(t["sex"] == "female") & (t["pclass"] == 1)])
    try:
        got = int(n)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个整数，实际是 {type(n).__name__}")
    return report(name, got == exp, f"实际 {got}，与参考答案不符",
                  "两个条件各自加括号，用 & 连接，再用 len() 数行数")


def most_expensive_town(town):
    name = "平均票价最高的港口"
    exp = _titanic().groupby("embark_town")["fare"].mean().idxmax()
    return report(name, town == exp, f"实际 {town!r}，与参考答案不符",
                  'groupby("embark_town")["fare"].mean() 之后用 .idxmax() 拿到最大值对应的组名（不是最大值本身）')


def best_who_group(group):
    name = "生还率最高的组"
    exp = _titanic().groupby("who")["survived"].mean().idxmax()
    return report(name, group == exp, f"实际 {group!r}，与参考答案不符",
                  'groupby("who")["survived"].mean() 之后用 .idxmax()')


def age_filled(s):
    name = "填充后的年龄列"
    t = _titanic()
    exp = t["age"].fillna(t["age"].median()).to_numpy()
    try:
        arr = np.asarray(s, dtype=float)
    except (TypeError, ValueError):
        return report(name, False, "应传入填充后的 age 列（Series 或数组）")
    if arr.shape != (891,):
        return report(name, False, f"长度应为 891，实际 {arr.shape}", "填充一列，不要丢掉任何行")
    if np.isnan(arr).any():
        return report(name, False, f"仍有 {int(np.isnan(arr).sum())} 个缺失值",
                      'titanic["age"].fillna(titanic["age"].median())')
    return report(name, bool(np.allclose(arr, exp)), "数值不对",
                  "用中位数（28.0）填充：fillna(titanic[\"age\"].median())")


# ---------- 讲义/练习：类 ----------
def bank_account(cls):
    name = "BankAccount"
    ok, acct = safe_call(name, cls, "Ada", 100)
    if not ok:
        return False
    try:
        acct.deposit(50)
        if not close(acct.balance, 150):
            return report(name, False, f"存入 50 后余额应为 150，实际 {acct.balance}",
                          "deposit 里 self.balance += amount")
        acct.withdraw(30)
        if not close(acct.balance, 120):
            return report(name, False, f"取出 30 后余额应为 120，实际 {acct.balance}",
                          "withdraw 里 self.balance -= amount")
        acct.withdraw(10_000)
        if not close(acct.balance, 120):
            return report(name, False, "余额不足时不应扣款",
                          "withdraw 里先比较 amount 和 self.balance")
        if not close(acct.get_balance(), 120):
            return report(name, False, "get_balance() 应返回当前余额")
        other = cls("Bob")
        if not close(other.balance, 0):
            return report(name, False, "不传初始余额时应默认为 0", "def __init__(self, owner, balance=0)")
        if not close(acct.balance, 120):
            return report(name, False, "两个对象的余额互相干扰了",
                          "余额要存在 self.balance 上，而不是全局变量")
    except NotImplementedError:
        return report(name, False, "还有方法没有实现（仍是 NotImplementedError）")
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"运行时出错 {type(e).__name__}: {e}",
                      "先读报错信息的最后一行，再用 print() 检查中间变量")
    return report(name, True)


# ---------- 作业：TabularDataset ----------
def tabular_dataset(cls):
    name = "TabularDataset"
    X = np.arange(12).reshape(6, 2)
    y = np.array([0, 10, 20, 30, 40, 50])
    ok, ds = safe_call(name, cls, X, y)
    if not ok:
        return False
    try:
        n = len(ds)
        if n != 6:
            return report(name, False, f"len(ds) 应为 6，实际 {n}",
                          "__len__ 返回样本个数 len(self.X)")
        item = ds[2]
        if not (isinstance(item, tuple) and len(item) == 2):
            return report(name, False, "ds[i] 应返回 (x_i, y_i) 元组",
                          "__getitem__ 里 return self.X[i], self.y[i]")
        xi, yi = item
        if not (np.allclose(np.asarray(xi), X[2]) and close(yi, y[2])):
            return report(name, False, f"ds[2] 应为 ({X[2]}, {y[2]})，实际 ({xi}, {yi})")
    except NotImplementedError:
        return report(name, False, "还有方法没有实现（仍是 NotImplementedError）")
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"运行时出错 {type(e).__name__}: {e}",
                      "先读报错信息的最后一行，再用 print() 检查中间变量")
    return report(name, True)


# ---------- 讲义：第一个分类器 ----------
def iris_accuracy(acc):
    name = "Iris 准确率"
    try:
        acc = float(acc)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(acc).__name__}")
    return report(name, 0.9 < acc <= 1.0, f"应大于 0.9，实际 {acc:.4f}",
                  "model.fit(X, y) 之后用 model.score(X, y) 打分")


def titanic_accuracy(acc):
    name = "Titanic 准确率"
    try:
        acc = float(acc)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(acc).__name__}")
    return report(name, 0.75 <= acc <= 0.85, f"应在 0.75–0.85 之间，实际 {acc:.4f}",
                  "四个特征：sex（female=1）、pclass、age（中位数填充）、fare；"
                  "train_test_split(..., test_size=0.25, random_state=42, stratify=target)")


# ---------- 阶段检查：Palmer Penguins ----------
def penguins_shape(df):
    name = "企鹅数据读取"
    try:
        shape = tuple(df.shape)
    except AttributeError:
        return report(name, False, "应传入 DataFrame", "penguins = data.load_penguins()")
    return report(name, shape == (344, 7), f"shape 应为 (344, 7)，实际 {shape}",
                  "data.load_penguins()，不要提前删行")


def penguins_missing(n):
    name = "含缺失值的行数"
    exp = int(_penguins().isna().any(axis=1).sum())
    try:
        n = int(n)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个整数，实际是 {type(n).__name__}")
    return report(name, n == exp, f"实际 {n}，与参考答案不符",
                  "先按行问“这一行有没有缺失”（isna().any(axis=1)），再求和；不要把各列的缺失数直接相加")


def penguins_clean(df):
    name = "清洗后的企鹅数据"
    exp_rows = len(_penguins().dropna())
    try:
        n_na = int(df.isna().sum().sum())
        shape = tuple(df.shape)
    except AttributeError:
        return report(name, False, "应传入 DataFrame")
    if n_na:
        return report(name, False, f"仍有 {n_na} 个缺失值", "dropna() 整行删除")
    return report(name, shape == (exp_rows, 7), f"shape 应为 ({exp_rows}, 7)，实际 {shape}",
                  "penguins.dropna() 即可：删含缺失的行，不要删列")


def mass_by_species(obj):
    name = "各物种平均体重"
    got = _as_dict(obj)
    p = _penguins()
    refs = [p.dropna().groupby("species")["body_mass_g"].mean().to_dict(),
            p.groupby("species")["body_mass_g"].mean().to_dict()]
    hint = 'clean.groupby("species")["body_mass_g"].mean()'
    if got is None:
        return report(name, False, "应传入 dict 或 groupby 得到的 Series", hint)
    for exp in refs:
        if set(got) == set(exp) and all(close(got[k], exp[k], tol=1e-3) for k in exp):
            return report(name, True)
    rounded = {k: round(v, 1) for k, v in refs[0].items()}
    return report(name, False, f"期望约 {rounded}", hint)
