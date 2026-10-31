"""第4周自查函数。用法：from zero2researcher.checks import week04 as ck; ck.gd_step(gd_step)

函数类练习把你写好的函数本身（不加括号）传进来，例如 ck.numerical_derivative(num_derivative)；
数值类练习把算出的数传进来，例如 ck.calhousing_r2_all(r2_all)。
"先笔算、再验证"的题（shape 侦探、纸上走一步梯度下降）失败时只给方法，不打印期望值。
名字里标着"只核对数值"的检查分辨不出方法错误（例如在训练集上打分），划分与拟合是否正确，请对照讲义自查。
"""
import numpy as np

from . import report, close, safe_call


# ---------- 参考实现（教师实现） ----------
def _synthetic_data():
    """与课上/作业完全相同的合成数据：y = 3x + 2 + 噪声，种子 42。"""
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
    """课堂练习 1.1：(6, 3) 的 X 乘 (3,) 的 w，结果的 shape。"""
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
    """课堂练习 S2：x = [1, 2, 3]、y = [2, 4, 6]、y_hat = w * x，从 w = 0、lr = 0.1 走一步。"""
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
                  "用课上的合成数据（种子 42）从 w=0, b=0 出发，lr=0.1 训练 200 步；"
                  "迭代步数不够（还没收敛）也会不通过")


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
