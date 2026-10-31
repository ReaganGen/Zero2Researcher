"""第7周自查函数。用法：from zero2researcher.checks import week07 as ck; ck.value_add(Value)

本周的练习是"实现一个类"：把你写好的 Value 类本身（类名，不加括号）传进来，
例如 ck.value_mul(Value)。检查会用你的类搭一个小表达式、调用 backward()，
再和参考实现的梯度逐个比对。数值类检查直接传入你算出的数。

参考实现照抄 Andrej Karpathy 的 micrograd（MIT License, (c) 2020 Andrej Karpathy，
https://github.com/karpathy/micrograd ），并补充了课程用到的 tanh 与 exp。
"""
import math
import random

from . import report, close, safe_call


# ---------- 参考实现（教师实现，micrograd engine.py + tanh/exp） ----------
class _Value:
    """标量自动微分节点（参考答案，结构与 micrograd 的 engine.py 一致）。"""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, _Value) else _Value(other)
        out = _Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, _Value) else _Value(other)
        out = _Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "指数只支持 int/float"
        out = _Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = _Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = _Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = _Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


# ---------- 参考实现（micrograd nn.py 结构，激活用课程口径 tanh） ----------
class _Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class _Neuron(_Module):
    def __init__(self, nin, nonlin=True):
        self.w = [_Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = _Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class _Layer(_Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [_Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class _MLP(_Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [_Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
                       for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ---------- 通用：用学生的类搭表达式，与参考实现比对 ----------
def _run_expr(name, Value, build, inputs, hint):
    ref_xs = [_Value(v) for v in inputs]
    ref_out = build(*ref_xs)
    ref_out.backward()
    try:
        xs = [Value(v) for v in inputs]
        out = build(*xs)
        out.backward()
    except NotImplementedError:
        return report(name, False, "还有方法没有实现（仍是 NotImplementedError）")
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"运行时出错 {type(e).__name__}: {e}", hint)
    try:
        out_data, grads = float(out.data), [float(x.grad) for x in xs]
    except (TypeError, ValueError, AttributeError):
        return report(name, False, "data 或 grad 不是数——检查每个运算是否返回新的 Value 节点", hint)
    if not close(out_data, ref_out.data):
        return report(name, False, f"前向值期望 {ref_out.data:.6f}，实际 {out_data}",
                      "先别管梯度：逐步 print 每个中间节点的 data")
    for i, (g, ref_x) in enumerate(zip(grads, ref_xs), start=1):
        if not close(g, ref_x.grad):
            return report(name, False,
                          f"第 {i} 个输入的梯度期望 {ref_x.grad:.6f}，实际 {g}", hint)
    return report(name, True)


# ---------- 课堂 / 作业：Value 的各个运算 ----------
# 有新导数的运算（mul / tanh / pow / relu / exp）一律放在表达式**中间**：后面还要再乘一个不是 1 的数，
# 这样它收到的 out.grad ≠ 1，"_backward 里忘了乘 out.grad"的实现才会算错梯度、被抓出来。
# value_add 例外：它只用加法，保证 __mul__ 写错时不会连累加法的检查（加法的局部导数恒为 1）。
# 提示只讲原理、不直接给出代码行（作业 A 考的正是这几行）。
_CHAIN = "链式法则：局部导数还要再乘下游传回来的梯度 out.grad，并用 += 累加"


def value_add(Value):
    return _run_expr("__add__ 的梯度", Value,
                     lambda a, b: (a + b) + (a + 1.0), [2.0, -3.0],
                     "加法的局部导数是 1：两个输入各自原样收到 out.grad（用 += 累加）；"
                     "遇到 int/float 要先包成 Value；同一节点用两次时梯度必须累加")


def value_mul(Value):
    return _run_expr("__mul__ 的梯度", Value,
                     lambda a, b: (a * b + a * a) * 3.0, [2.0, -3.0],
                     "乘法的局部导数是\"另一个因子的值\"：a 的梯度看 b 的 data，b 的梯度看 a 的 data。"
                     + _CHAIN)


def value_tanh(Value):
    return _run_expr("tanh 的梯度", Value,
                     lambda x: (x * 2.0 + 1.0).tanh() * 3.0, [0.5],
                     "tanh 的局部导数是 1 − t²，t 是前向算出的 tanh 值（不是输入 x）。" + _CHAIN)


def value_pow(Value):
    return _run_expr("__pow__ 的梯度", Value,
                     lambda x: (x ** 2 + x ** -1) * 3.0, [3.0],
                     "幂法则：d(xⁿ)/dx = n·xⁿ⁻¹（n 是指数 other，x 是 self 的 data）。" + _CHAIN)


def value_relu(Value):
    return _run_expr("relu 的梯度", Value,
                     lambda a, b: a.relu() * 3.0 + b.relu(), [2.0, -1.5],
                     "relu 的局部导数：输出 > 0 时为 1，否则为 0（正半边放行、负半边拦下）。" + _CHAIN)


def value_exp(Value):
    return _run_expr("exp 的梯度", Value,
                     lambda x: (x * 2.0).exp() * 3.0, [0.7],
                     "eˣ 的导数就是 eˣ 自己——前向已经算好、存在输出节点的 data 里（不是输入的 data）。" + _CHAIN)


def value_sub(Value):
    return _run_expr("__neg__ / __sub__ 的梯度", Value,
                     lambda a, b: (a - b) + (1.0 - b), [5.0, 3.0],
                     "不用新写导数：取负 = 乘以 -1，减法 = 加上取负；"
                     "1.0 - b 走的是 __rsub__，注意方向是 other 减 self")


def value_div(Value):
    return _run_expr("除法的梯度", Value,
                     lambda a, b: a / b + 6.0 / a, [4.0, 2.0],
                     "除法经 __pow__ 实现：a / b = a · b⁻¹；6.0 / a 走 __rtruediv__（方向是 other 除以 self）。"
                     "先确认 ck.value_pow 已通过")


def gradient_reuse(Value):
    """专查 "= 而不是 +=" 的经典 bug：b = a + a 时 a 的梯度应为 2。"""
    name = "节点复用的梯度累加"
    try:
        a = Value(3.0)
        b = a + a
        b.backward()
        g = float(a.grad)
    except NotImplementedError:
        return report(name, False, "还有方法没有实现（仍是 NotImplementedError）")
    except Exception as e:  # noqa: BLE001
        return report(name, False, f"运行时出错 {type(e).__name__}: {e}")
    return report(name, close(g, 2.0),
                  f"b = a + a 后 a.grad 应为 2，实际 {g}",
                  "_backward 里用 = 赋值会把另一条路径传来的梯度覆盖掉——"
                  "同一个节点被使用多次时，每条路径的梯度都要累加（+=）")


# ---------- 课堂练习：纸上反向传播 ----------
def paper_backprop(grad_a, grad_b, grad_c):
    """L = (a*b + c)**2，a=3, b=-1, c=5 时的三个梯度。"""
    name = "纸上反向传播 L=(a·b+c)²"
    exp = (-4.0, 12.0, 4.0)
    got = (grad_a, grad_b, grad_c)
    for g, e, who in zip(got, exp, "abc"):
        if not close(g, e):
            return report(name, False, f"∂L/∂{who} 期望 {e}，实际 {g}",
                          "先算 d = a·b + c = 2，∂L/∂d = 2d = 4；"
                          "再用链式法则往回乘：∂L/∂a = 4·b，∂L/∂b = 4·a，∂L/∂c = 4")
    return report(name, True)


# ---------- 梯度检查（CS231n 口径） ----------
def numerical_grad(fn):
    """学生的数值梯度函数：在 f(x) = x³ + 2x 上与解析值 3x² + 2 比对。"""
    cases = [(2.0, 14.0), (-1.0, 5.0)]
    f = lambda x: x ** 3 + 2 * x  # noqa: E731
    for x, exp in cases:
        ok, got = safe_call("numerical_grad", fn, f, x)
        if not ok:
            return False
        if not close(got, exp, tol=1e-4):
            return report("numerical_grad", False,
                          f"f(x)=x³+2x 在 x={x} 处期望约 {exp}，实际 {got}",
                          "中心差分：(f(x + h) - f(x - h)) / (2 * h)，h 取 1e-5")
    return report("numerical_grad", True)


def torch_agreement(rel_err):
    """与 PyTorch autograd 对照的相对误差应 < 1e-6。"""
    name = "与 PyTorch autograd 对照"
    try:
        v = float(rel_err)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数（相对误差），实际是 {type(rel_err).__name__}")
    return report(name, 0 <= v < 1e-6,
                  f"相对误差应 < 1e-6，实际 {v:.3g}",
                  "相对误差 = |a - b| / max(|a|, |b|)；torch 张量要用 dtype=torch.float64 "
                  "并设 requires_grad=True，取梯度用 .grad.item()")


# ---------- moons 训练 ----------
def mlp_accuracy(acc):
    name = "moons 训练准确率"
    try:
        v = float(acc)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(acc).__name__}")
    return report(name, 0.95 <= v <= 1.0,
                  f"期望 ≥ 0.95（实测可到 1.00），实际 {v:.4f}",
                  "常见原因：忘了在每步开头 zero_grad；标签没有变成 ±1；"
                  "学习率没有按 1.0 → 0.1 衰减；步数不够 100")
