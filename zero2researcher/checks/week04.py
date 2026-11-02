"""第4周自查函数。用法：from zero2researcher.checks import week04 as ck; ck.gd_step(gd_step)

函数类练习把你写好的函数本身（不加括号）传进来，例如 ck.numerical_derivative(num_derivative)；
数值类练习把算出的数传进来，例如 ck.calhousing_r2_all(r2_all)。
"先笔算、再验证"的题（shape 侦探、纸上走一步梯度下降）失败时只给方法，不打印期望值。
名字里标着"只核对数值"的检查分辨不出方法错误（例如在训练集上打分），划分与拟合是否正确，请对照讲义自查。
讲义 S4 "同一个指标记了两次"那一格（np.linalg.solve 遇到重复列）没有自查：它的结果取决于电脑上的
线性代数库（有的报错，有的给出任意解），两种情况都讲得通。
"""
import functools

import numpy as np

from . import report, close, safe_call


# ---------- 参考实现 ----------
def _synthetic_data():
    """与讲义/作业完全相同的合成数据：y = 3x + 2 + 噪声，种子 42。"""
    rng = np.random.default_rng(42)
    X = rng.uniform(-2, 2, size=(200, 1))
    y = 3 * X[:, 0] + 2 + rng.normal(0, 1, size=200)
    return X, y


def _small_data():
    X = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.0], [-2.0, 1.5], [1.0, 1.0]])
    y = np.array([4.0, 2.0, 1.0, -3.0, 2.5])
    return X, y


def _gradients(X, y, w, b):
    err = X @ w + b - y
    return 2 / len(y) * X.T @ err, 2 / len(y) * err.sum()


def _gd_step(X, y, w, b, lr):
    grad_w, grad_b = _gradients(X, y, w, b)
    return w - lr * grad_w, b - lr * grad_b


def _sklearn_params():
    from sklearn.linear_model import LinearRegression
    X, y = _synthetic_data()
    m = LinearRegression().fit(X, y)
    return float(m.coef_[0]), float(m.intercept_)


def _ols_cases():
    """正规方程的两组检查数据：一组 5×2；一组 40×3 且各列均值不为 0（忘了拼 1 列时解会明显不同）。"""
    rng = np.random.default_rng(0)
    X1, y1 = _small_data()
    X2 = rng.normal(3.0, 2.0, size=(40, 3))
    y2 = X2 @ np.array([1.5, -2.0, 0.5]) + 4.0 + rng.normal(0, 0.1, 40)
    return [(X1, y1), (X2, y2)]


def _lstsq_wb(X, y):
    A = np.column_stack([X, np.ones(len(X))])
    theta = np.linalg.lstsq(A, y, rcond=None)[0]
    return theta[:-1], theta[-1]


def _ridge_wb(X, y, lam):
    """参考答案：中心化后解 (XcᵀXc + λI) w = Xcᵀ(y − ȳ)，截距不惩罚。与 sklearn Ridge(alpha=λ) 一致。"""
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc = X - x_mean
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - y_mean))
    return w, y_mean - x_mean @ w


@functools.lru_cache(maxsize=1)
def _calhousing_std_cached():
    """与讲义 S4 相同：75/25 划分（random_state=42），用训练集的均值和标准差标准化。返回 (X_std, y_raw)。"""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    housing = fetch_california_housing(as_frame=True)
    X_train, _, y_train, _ = train_test_split(housing.data, housing.target, test_size=0.25, random_state=42)
    X_raw, y_raw = X_train.to_numpy(), y_train.to_numpy()
    X_std = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
    X_std.flags.writeable = False
    y_raw.flags.writeable = False
    return X_std, y_raw


def _gd_train(X, y, lr, n_iters):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(n_iters):
        w, b = _gd_step(X, y, w, b, lr)
    return w, b


def _as_wb(got, name):
    """把学习者返回的 (w, b) 转成 (一维数组, 浮点数)；失败时返回 None。"""
    try:
        w, b = got
        return np.ravel(np.asarray(w, dtype=float)), float(np.ravel(np.asarray(b, dtype=float))[0])
    except (TypeError, ValueError, IndexError):
        report(name, False, "应返回 (w, b) 两个值：w 是长度为特征数的数组，b 是一个数")
        return None


# ---------- S1 · 数学工作坊 ----------
def dot(fn):
    name = "my_dot"
    rng = np.random.default_rng(0)
    cases = [(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))]
    cases += [(rng.normal(size=5), rng.normal(size=5)) for _ in range(2)]
    for a, b in cases:
        ok, got = safe_call(name, fn, a, b)
        if not ok:
            return False
        if not close(got, float(np.dot(a, b)), tol=1e-8):
            return report(name, False, f"my_dot({np.round(a, 2)}, {np.round(b, 2)}) "
                          f"期望 {np.dot(a, b):.4f}，实际 {got}",
                          "对应位置相乘再求和：total += x * y")
    return report(name, True)


def numerical_derivative(fn):
    name = "num_derivative"
    cases = [(lambda t: t ** 2, 3.0, 6.0),
             (np.sin, 0.5, float(np.cos(0.5))),
             (lambda t: 5 * t + 1, -2.0, 5.0)]
    for f, x, exp in cases:
        ok, got = safe_call(name, fn, f, x)
        if not ok:
            return False
        if not close(got, exp, tol=1e-3):
            return report(name, False, f"在 x={x} 处期望 ≈{exp:.4f}，实际 {got}",
                          "(f(x + h) - f(x - h)) / (2 * h)，h 取 1e-5 左右")
    return report(name, True)


def shape_Xw(shape):
    """练习 1.1：(6, 3) 的 X 乘 (3,) 的 w，结果的 shape。"""
    name = "shape 侦探"
    rng = np.random.default_rng(0)
    exp = (rng.normal(size=(6, 3)) @ rng.normal(size=3)).shape
    try:
        got = tuple(int(d) for d in shape)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个 shape 元组，实际是 {shape!r}", "(X @ w).shape")
    return report(name, got == exp, f"实际 {got}，与正确答案不符",
                  "对照讲义里的 shape 规则：A @ B 要求 A 的最后一维等于 B 的第一维，这一维在结果里被'消掉'")


# ---------- S2 · 损失 ----------
def mse_loss(fn):
    name = "mse_loss"
    cases = [(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0]), 5 / 3),
             (np.array([0.0, 0.0]), np.array([2.0, -2.0]), 4.0),
             (np.array([1.5]), np.array([1.5]), 0.0)]
    for y_hat, y, exp in cases:
        ok, got = safe_call(name, fn, y_hat, y)
        if not ok:
            return False
        if not close(got, exp, tol=1e-8):
            return report(name, False, f"mse_loss({y_hat}, {y}) 期望 {exp:.4f}，实际 {got}",
                          "np.mean((y_hat - y) ** 2)：残差平方的平均")
    return report(name, True)


# ---------- S3 · 梯度下降 ----------
def gd_step(fn):
    name = "gd_step"
    X, y = _small_data()
    w0, b0, lr = np.array([0.1, -0.2]), 0.3, 0.05
    ok, got = safe_call(name, fn, X.copy(), y.copy(), w0.copy(), b0, lr)
    if not ok:
        return False
    try:
        w1, b1 = got
        w1 = np.ravel(np.asarray(w1, dtype=float))
        b1 = float(b1)
    except (TypeError, ValueError):
        return report(name, False, "应返回更新后的 (w_new, b_new) 两个值")
    exp_w, exp_b = _gd_step(X, y, w0, b0, lr)
    wrong_w, wrong_b = _gd_step(X, y, w0, b0, -lr)
    if np.allclose(w1, wrong_w, atol=1e-6) and close(b1, wrong_b, tol=1e-6):
        return report(name, False, "参数朝梯度的正方向走了：越更新离最优越远",
                      "更新是减号：w - lr * grad_w，b - lr * grad_b")
    ok2 = w1.shape == exp_w.shape and np.allclose(w1, exp_w, atol=1e-6) and close(b1, exp_b, tol=1e-6)
    return report(name, ok2, "数值与参考实现不一致（容差 1e-6）",
                  "y_hat = X @ w + b；grad_w = 2/n * X.T @ (y_hat - y)；grad_b = 2/n * np.sum(y_hat - y)")


def one_gd_step(grad, w1):
    """练习 S2：x = [1, 2, 3]、y = [2, 4, 6]、y_hat = w * x，从 w = 0、lr = 0.1 走一步。"""
    name = "一步梯度下降"
    x, y, w0, lr = np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]), 0.0, 0.1
    exp_grad = float(np.mean(2 * (w0 * x - y) * x))
    exp_w1 = w0 - lr * exp_grad
    if not close(grad, exp_grad):
        hint = "w = 0 时 y_hat 全是 0，残差 y_hat - y 就是 -y；逐项乘 2、乘 x，再取平均"
        try:
            if close(-float(grad), exp_grad):
                hint = "符号反了：残差是 y_hat - y（预测减真实），不是 y - y_hat"
        except (TypeError, ValueError):
            pass
        return report(name, False, f"梯度 {grad} 不对", hint)
    return report(name, close(w1, exp_w1), f"梯度对了，但一步后的 w = {w1} 不对",
                  "更新规则：w1 = w0 - lr * grad（往梯度的反方向走）")


def fitted_params(w, b):
    name = "训练得到的 (w, b)"
    exp_w, exp_b = _sklearn_params()
    try:
        w_val = float(np.ravel(np.asarray(w, dtype=float))[0])
        b_val = float(np.ravel(np.asarray(b, dtype=float))[0])
    except (TypeError, ValueError, IndexError):
        return report(name, False, "应传入训练后的 w 和 b（数或长度为 1 的数组）")
    # 教案要求：梯度下降的解与 sklearn 相差 < 1e-3（绝对误差）。lr=0.1 训练 200 步时实测相差 8.9e-16。
    ok = abs(w_val - exp_w) < 1e-3 and abs(b_val - exp_b) < 1e-3
    return report(name, ok, f"与 sklearn 的解（w≈{exp_w:.4f}, b≈{exp_b:.4f}）相差超过 1e-3，"
                  f"实际 w={w_val:.4f}, b={b_val:.4f}",
                  "用讲义的合成数据（种子 42）从 w=0, b=0 出发，lr=0.1 训练 200 步；"
                  "迭代步数不够（还没收敛）也会不通过")


def normal_equation(fn):
    """讲义 S3：normal_equation(X, y) 解 (AᵀA)θ = Aᵀy（A = [X, 1]），返回 (w, b)。"""
    name = "normal_equation"
    for X, y in _ols_cases():
        ok, got = safe_call(name, fn, X.copy(), y.copy())
        if not ok:
            return False
        wb = _as_wb(got, name)
        if wb is None:
            return False
        w, b = wb
        exp_w, exp_b = _lstsq_wb(X, y)
        if w.shape == (exp_w.shape[0] + 1,):
            return report(name, False, f"w 的长度应为 {exp_w.shape[0]}（特征数），实际 {w.shape[0]}",
                          "解出的 theta 最后一个元素是截距：w = theta[:-1]，b = theta[-1]")
        if w.shape != exp_w.shape:
            return report(name, False, f"w 的长度应为 {exp_w.shape[0]}（特征数），实际 {w.shape[0]}")
        if np.allclose(w, exp_w, atol=1e-8) and abs(b - exp_b) < 1e-8:
            continue
        hint = "A = np.column_stack([X, np.ones(len(X))])；theta = np.linalg.solve(A.T @ A, A.T @ y)"
        if np.allclose(w, np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-6):
            hint = "解出的是'没有截距'的直线：在 X 右边拼一列 1，截距才会和 w 一起被解出来"
        return report(name, False, "与最小二乘的解不一致（容差 1e-8）", hint)
    return report(name, True)


# ---------- S4 · California Housing ----------
def calhousing_r2_single(r2):
    name = "CalHousing 单特征 R²（只核对数值）"
    try:
        r2 = float(r2)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(r2).__name__}")
    return report(name, 0.40 <= r2 <= 0.52, f"应在 0.40–0.52 之间，实际 {r2:.4f}",
                  "只用 MedInc 一列；75/25 划分（random_state=42），训练集拟合、测试集打分")


def calhousing_r2_all(r2):
    name = "CalHousing 全特征 R²（只核对数值）"
    try:
        r2 = float(r2)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(r2).__name__}")
    return report(name, 0.55 <= r2 <= 0.63, f"应在 0.55–0.63 之间，实际 {r2:.4f}",
                  "全部 8 个特征；75/25 划分（random_state=42），训练集拟合、测试集打分")


def calhousing_normal_eq(w, b):
    """讲义 S4 三方对账：在标准化后的训练集 X_std 上用正规方程解出的 (w, b)。"""
    name = "CalHousing 正规方程"
    wb = _as_wb((w, b), name)
    if wb is None:
        return False
    w, b = wb
    if w.shape != (8,):
        return report(name, False, f"w 应有 8 个系数（每个特征一个），实际 {w.shape[0]} 个",
                      "w_ne_h, b_ne_h = normal_equation(X_std, y_raw)")
    from sklearn.linear_model import LinearRegression
    X_std, y_raw = _calhousing_std_cached()
    ref = LinearRegression().fit(X_std, y_raw)
    gap = max(np.abs(w - ref.coef_).max(), abs(b - ref.intercept_))
    if gap < 1e-6:
        return report(name, True)
    hint = "在标准化后的训练集上解：normal_equation(X_std, y_raw)（X_std 用训练集的均值和标准差）"
    w_gd, b_gd = _gd_train(X_std, y_raw, 0.1, 200)
    if max(np.abs(w - w_gd).max(), abs(b - b_gd)) < 1e-3:
        hint = ("这是还没收敛的梯度下降（lr=0.1、200 步）给出的系数：测试 R² 已经一样，系数还差约 0.06。"
                "这一格要的是正规方程直接解出的 (w, b)")
    else:
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        housing = fetch_california_housing(as_frame=True)
        X_train, _, y_train, _ = train_test_split(housing.data, housing.target, test_size=0.25, random_state=42)
        raw = LinearRegression().fit(X_train, y_train)
        if np.allclose(w, raw.coef_, atol=1e-6):
            hint = "这是在原始（未标准化）特征上拟合的系数；这一格用标准化后的 X_std"
    return report(name, False, f"与 LinearRegression 在 X_std 上的解相差 {gap:.2e}（容差 1e-6）", hint)


def ridge_closed_form(fn):
    """讲义 S4：ridge_closed_form(X, y, lam) 先中心化（截距不惩罚），返回 (w, b)，应与 sklearn Ridge(alpha=lam) 一致。"""
    from sklearn.linear_model import Ridge
    name = "ridge_closed_form"
    rng = np.random.default_rng(1)
    X = rng.normal(2.0, 1.5, size=(30, 4))                      # 均值不为 0：截距处理错了会被看出来
    y = X @ np.array([1.0, 0.0, -2.0, 0.5]) + 3.0 + rng.normal(0, 0.5, 30)
    X_dup = np.column_stack([X, X[:, 0]])                       # 含重复列：XᵀX 不可逆，Ridge 仍有唯一解
    for XX, lam in [(X, 0.5), (X, 10.0), (X_dup, 1.0)]:
        ok, got = safe_call(name, fn, XX.copy(), y.copy(), lam)
        if not ok:
            return False
        wb = _as_wb(got, name)
        if wb is None:
            return False
        w, b = wb
        ref = Ridge(alpha=lam).fit(XX, y)
        if w.shape == ref.coef_.shape and np.allclose(w, ref.coef_, atol=1e-6) and abs(b - ref.intercept_) < 1e-6:
            continue
        if w.shape != ref.coef_.shape:
            return report(name, False, f"w 的长度应为 {ref.coef_.shape[0]}（特征数），实际 {w.shape[0]}",
                          "截距单独返回：return w, b")
        hint = ("先中心化：Xc = X - X.mean(axis=0)；w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ (y - y.mean()))；"
                "b = y.mean() - X.mean(axis=0) @ w")
        A = np.column_stack([XX, np.ones(len(XX))])
        theta_pen = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
        Xc = XX - XX.mean(axis=0)
        n = len(y)
        w_mean = np.linalg.solve(Xc.T @ Xc / n + lam * np.eye(XX.shape[1]), Xc.T @ (y - y.mean()) / n)
        if np.allclose(w, theta_pen[:-1], atol=1e-6):
            hint = ("截距也被惩罚了：拼一列 1 再加 lam·I，会把 b 往 0 拉。sklearn 的 Ridge 不惩罚截距——"
                    "先中心化，最后单独算 b")
        elif np.allclose(w, w_mean, atol=1e-6):
            hint = "你用的是'平均'损失（相当于把 λ 放大了 n 倍）；sklearn 的 Ridge 用平方和：Xc.T @ Xc + lam * I"
        return report(name, False, f"与 sklearn Ridge(alpha={lam}) 不一致（容差 1e-6）", hint)
    return report(name, True)


def ridge_path(lams, norms):
    """练习 S4（选做）：California Housing 标准化特征上，Ridge 系数的范数 ‖w‖ 随 λ 的变化。

    只核对 ‖w‖：X_std 的每列已经中心化，"截距也被惩罚"（A.T @ A + lam * I）的写法只改变 b、不改变 w，
    这里抓不到；截距的错误由讲义 S4 的 ck.ridge_closed_form 负责（它专门识别"截距也加了 λI"）。"""
    name = "Ridge 系数路径"
    try:
        lams = np.ravel(np.asarray(lams, dtype=float))
        norms = np.ravel(np.asarray(norms, dtype=float))
    except (TypeError, ValueError):
        return report(name, False, "应传入 lams 和对应的 ‖w‖ 两个数组：ck.ridge_path(lams, norms)")
    if lams.shape != norms.shape or lams.size < 2:
        return report(name, False, f"lams 与 norms 应一样长（至少 2 个），实际 {lams.shape} 与 {norms.shape}",
                      "norms = np.linalg.norm(coefs, axis=1)：每个 λ 一个范数")
    if np.any(np.diff(lams) <= 0):
        return report(name, False, "lams 应从小到大排列", "lams = np.logspace(-2, 5, 50)")
    if np.any(np.diff(norms) > 1e-12 * max(1.0, norms.max())):
        return report(name, False, "‖w‖ 应随 λ 增大而单调不增，实际有上升的地方",
                      "检查每个 λ 是否都用同一份 X_std、y_raw，以及惩罚项是否写成 lam * np.eye(d)")
    X_std, y_raw = _calhousing_std_cached()
    ref = np.array([np.linalg.norm(_ridge_wb(X_std, y_raw, lam)[0]) for lam in lams])
    bad = np.where(~np.isclose(norms, ref, rtol=1e-6, atol=1e-10))[0]
    if bad.size:
        i = bad[0]
        return report(name, False, f"λ = {lams[i]:.3g} 时 ‖w‖ 应约为 {ref[i]:.4f}，实际 {norms[i]:.4f}",
                      "数据用讲义的 X_std（训练集标准化）与 y_raw；Ridge 用平方和损失、截距不惩罚")
    return report(name, True)
