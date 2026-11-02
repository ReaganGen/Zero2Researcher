"""第18周自查函数。用法：from zero2researcher.checks import week18 as ck; ck.incremental_mean(incremental_mean)

函数类练习把函数本身传进来（不加括号），例如 ck.reward_to_go(reward_to_go)；其余检查直接传入你算出的数。
含随机性的结果按合理区间检查，区间依据本机实测 ± 余量：
- 10 臂测试平台（Sutton & Barto 第 2 版第 2 章的设定：2000 个任务 × 1000 步，q*(a) ~ N(0, 1)，奖励 ~ N(q*(a), 1)）；
- 伯努利 10 臂测试平台（课程自造：q*(a) ~ U(0, 1)，拉一次得 0/1；Thompson 采样的后验 Beta(1 + 成功, 1 + 失败)，
  写法照 Russo et al. 2017 Algorithm 3.2，代码自写）；
- FrozenLake-v1 4×4 表格 Q-learning（HF Deep RL Course Unit 2 的超参数），以及用 env.unwrapped.P 做的价值迭代
  （精确解；先在课程自造的"两个房间"小 MDP 上对闭式解）；
- CartPole-v1 REINFORCE（策略网络 4→16→2、Adam 1e-2、γ = 1.0；"达标" = 最近 100 回合平均回报首次 ≥ 475，
  475 是 Gymnasium 注册表里 CartPole-v1 的 reward_threshold）；作业 D 的价值网络基线（4→16→1、输出 ×100、Adam 1e-2）；
- MountainCar-v0 离散化表格 Q-learning 的三种奖励设计（势函数塑形函数本身另用构造的状态直接比对数值）。
本模块的大部分检查只用 NumPy；学习者的函数返回 torch 张量时也能检查（用 .detach() 转换）。
只有 bellman_optimal_values（要 FrozenLake 的转移表）与 fit_value（要一个小网络）在函数内部导入 gymnasium / torch。
价值迭代的学习者函数叫 bellman_optimal_values(R, C, gamma, tol)，接收讲义给定的 model_arrays 整理出的数组；
不用 CS234 作业里的 value_iteration / policy_iteration 这类函数名与 (P, 状态数, 动作数, γ) 的接口。
"""
import numpy as np

from . import report, safe_call

CARTPOLE_THRESHOLD = 475.0     # Gymnasium 1.4 注册表中 CartPole-v1 的 reward_threshold
CARTPOLE_MAX_RETURN = 500.0    # CartPole-v1 的 max_episode_steps
SOLVE_WINDOW = 100
REINFORCE_SETTINGS = ("A", "B", "C", "D")
N_SEEDS_RL = 5
# FrozenLake 4×4、γ = 0.95 的精确解（价值迭代 tol 1e-10，与策略迭代的线性方程解差 < 2e-9；Gymnasium 1.4）
FROZENLAKE_V0 = {False: 0.95 ** 5, True: 0.180472}             # 起点的 V*：不滑 / 滑
FROZENLAKE_OPT_SUCCESS = {99: 0.728185, 100: 0.729766, 200: 0.778133, None: 0.780488}   # 滑冰面最优策略 H 步内的成功率


def _in_range(name, value, lo, hi, hint=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return report(name, False, f"应传入一个数，实际是 {type(value).__name__}")
    if not np.isfinite(v):
        return report(name, False, f"数值不是有限数（{v}）", hint)
    return report(name, lo <= v <= hi, f"期望落在 [{lo}, {hi}]，实际 {v:.4f}", hint)


def _np(x):
    """torch 张量 / numpy 数组 / 列表 → float64 numpy 数组。"""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _floats(name, values, n):
    try:
        vals = [float(v) for v in values]
    except (TypeError, ValueError):
        return None, report(name, False, f"应传入 {n} 个数")
    if len(vals) != n or not all(np.isfinite(v) for v in vals):
        return None, report(name, False, f"应传入 {n} 个有限的数，实际 {values}")
    return vals, True


# ---------- S1 · 多臂老虎机 ----------
def _ref_incremental_mean(q, n, r):
    return q + (r - q) / n


def incremental_mean(fn):
    """fn(q, n, r)：q 是前 n−1 个奖励的平均 Q_n，r 是第 n 个奖励 R_n，返回 Q_{n+1}（前 n 个奖励的平均）。"""
    name = "增量均值更新"
    rewards = [2.0, -1.0, 3.0, 0.5, 0.5, -2.25, 4.0]
    steps = []                                   # 每一步都喂"正确的"旧估计，错误不会逐步累积
    for n, r in enumerate(rewards, start=1):
        q = float(np.mean(rewards[:n - 1])) if n > 1 else 0.0
        ok, got = safe_call(name, fn, q, n, r)
        if not ok:
            return False
        try:
            got = float(got)
        except (TypeError, ValueError):
            return report(name, False, f"应返回一个数，实际是 {type(got).__name__}",
                          "函数最后要 return 新的估计值")
        steps.append((q, n, r, got, _ref_incremental_mean(q, n, r)))   # = 前 n 个奖励的平均
    wrong = [st for st in steps if abs(st[3] - st[4]) > 1e-9]
    if not wrong:
        return report(name, True)
    # 常见错误：只有在**每一步**都与某个错误公式吻合时才给出对应提示
    mistakes = [
        (lambda q, n, r: q + (r - q) / (n + 1),
         "分母多了 1：r 已经是第 n 个奖励，步长是 1/n（S&B 式 2.3：Q_{n+1} = Q_n + (R_n − Q_n)/n）"),
        (lambda q, n, r: q + (r - q) / max(n - 1, 1), "分母少了 1：步长是 1/n，n 是算上这个奖励之后的次数"),
        (lambda q, n, r: (q + r) / 2, "(q + r) / 2 让最新的奖励永远占一半权重；样本平均里每个奖励的权重都是 1/n"),
        (lambda q, n, r: r, "你直接返回了最新的奖励；要在旧估计上走一小步：q + (r − q) / n"),
        (lambda q, n, r: q + r, "忘了乘步长 1/n，而且括号里应是 (r − q)"),
    ]
    hint = "新估计 = 旧估计 + 步长 × (新奖励 − 旧估计)，步长 = 1/n"
    for f, h in mistakes:
        if all(abs(got - f(q, n, r)) < 1e-9 for q, n, r, got, _ in steps):
            hint = h
            break
    q, n, r, got, want = wrong[0]
    return report(name, False,
                  f"奖励序列 {rewards[:n]} 的第 {n} 步（旧估计 q = {q:.4f}）：期望 {want:.4f}（前 {n} 个奖励的平均），"
                  f"实际 {got:.4f}", hint)


def epsilon_greedy(fn):
    """fn(Q, epsilon, rng)：Q 是一维数组；以概率 ε 从全部 k 个动作中均匀随机选，否则选 argmax(Q)；返回动作下标。"""
    name = "ε-greedy 选动作"
    Q = np.array([0.1, 0.5, 0.2, 0.0])
    Q_before = Q.copy()
    k, best = len(Q), 1
    rng = np.random.default_rng(0)

    def counts(eps, n):
        c = np.zeros(k)
        for _ in range(n):
            ok, a = safe_call(name, fn, Q, eps, rng)
            if not ok:
                return None
            try:
                a = int(a)
            except (TypeError, ValueError):
                report(name, False, f"应返回一个动作下标（整数），实际是 {type(a).__name__}",
                       "探索时 int(rng.integers(len(Q)))，利用时 int(np.argmax(Q))")
                return None
            if not 0 <= a < k:
                report(name, False, f"返回的动作 {a} 不在 0..{k - 1} 之内")
                return None
            c[a] += 1
        return c / n

    f0 = counts(0.0, 300)
    if f0 is None:
        return False
    if not np.array_equal(Q, Q_before):
        return report(name, False, "函数改动了传进来的 Q", "选动作只读 Q，不要原地修改它")
    if f0[best] < 1.0:
        return report(name, False, f"ε = 0 时应永远选估计值最大的动作 {best}，实际只有 {f0[best]:.0%} 选中",
                      "检查比较方向：rng.random() < epsilon 时才探索")
    f1 = counts(1.0, 8000)
    if f1 is None:
        return False
    if np.abs(f1 - 1 / k).max() > 0.03:
        hint = ("你在探索时把贪心动作排除了。本课程的约定（S&B 与 HF 的 env.action_space.sample() 相同）："
                "探索时从全部 k 个动作里均匀选：rng.integers(len(Q))") if f1[best] < 0.01 else \
            "探索时从全部动作里均匀选：rng.integers(len(Q))"
        return report(name, False, f"ε = 1 时应完全均匀随机（每个约 25%），实际频率 {np.round(f1, 3)}", hint)
    f = counts(0.2, 20000)
    if f is None:
        return False
    want_best = 1 - 0.2 + 0.2 / k
    if abs(f[best] - want_best) > 0.015:
        if abs(f[best] - 0.8) < 0.015:
            hint = ("你在探索时只从非贪心动作里选。本课程的约定（S&B 与 HF 的 env.action_space.sample() 相同）："
                    "探索时从全部 k 个动作里均匀选，所以贪心动作的总概率是 1 − ε + ε/k")
        elif abs(f[best] - 0.2) < 0.03:
            hint = "比较方向反了：应是以概率 ε 探索、以概率 1 − ε 利用"
        else:
            hint = f"Q = {Q.tolist()}、ε = 0.2 时，选中动作 {best} 的概率应为 1 − ε + ε/k = {want_best:.2f}"
        return report(name, False, f"ε = 0.2 时动作 {best} 的频率应约为 {want_best:.2f}，实际 {f[best]:.3f}", hint)
    others = np.delete(f, best)
    if np.abs(others - 0.2 / k).max() > 0.01:
        return report(name, False, f"ε = 0.2 时每个非贪心动作应各约 {0.2 / k:.2f}，实际 {np.round(others, 3)}",
                      "探索时要均匀随机，不要偏向某个动作")
    return report(name, True)


def _check_choose_eps_greedy(fn):
    name = "ε-greedy 选臂 choose_eps_greedy"
    n, k, best = 20000, 4, 1
    rng = np.random.default_rng(0)
    Q = np.tile([0.1, 0.5, 0.2, 0.0], (n, 1))
    Q_before = Q.copy()
    no_noise = np.zeros((n, k))

    def call(Q_, eps, noise):
        ok, a = safe_call(name, fn, Q_, eps, rng, noise)
        if not ok:
            return None
        a = _arms(a)
        if a.shape != (n,) or not np.all(np.isin(a, np.arange(k))):
            report(name, False, f"{n} 个任务、每个任务 {k} 个臂时，应返回 {n} 个 0–{k - 1} 之间的臂下标，实际形状 {a.shape}",
                   "Q 的每一行是一个任务：对每一行取 argmax 用 axis=1；探索时每个任务各抽一个臂（size=任务数）")
            return None
        return a

    a = call(Q, 0.0, no_noise)
    if a is None:
        return False
    if not np.array_equal(Q, Q_before):
        return report(name, False, "函数改动了传进来的 Q", "选臂只读 Q；更新 Q 是 update_mean 的事")
    if not np.all(a == best):
        hint = "利用：np.argmax(Q + tie_noise, axis=1)；以概率 ε 才探索"
        if abs(np.mean(a == best) - 1 / k) < 0.02:
            hint = "ε = 0 时却几乎全在随机选：比较方向反了——rng.random(任务数) < epsilon 的任务才探索"
        return report(name, False, f"ε = 0 时每个任务都应选估计值最大的臂 {best}，实际只有 {np.mean(a == best):.0%}", hint)
    noise = 1e-9 * rng.random((n, k))
    a = call(np.zeros((n, k)), 0.0, noise)
    if a is None:
        return False
    if not np.array_equal(a, np.argmax(noise, axis=1)):
        hint = "利用时取 argmax(Q + tie_noise)：Q 全部并列时，靠这点噪声随机打破平局"
        if np.all(a == 0):
            hint = "Q 全部并列时你总是选臂 0——这是 np.argmax 直接用 Q 的结果；要用 np.argmax(Q + tie_noise, axis=1)"
        return report(name, False, "Q 全部并列（全 0）、ε = 0 时，选的臂应是 argmax(Q + tie_noise)", hint)
    a = call(Q, 1.0, no_noise)
    if a is None:
        return False
    freq = np.bincount(a.astype(int), minlength=k) / n
    if np.abs(freq - 1 / k).max() > 0.02:
        hint = "探索时从全部 k 个臂里均匀随机选：rng.integers(0, 臂数, size=任务数)"
        if freq[best] < 0.01:
            hint = "你在探索时把估计最好的臂排除了；本课程的约定（S&B、练习 E2 相同）是从全部 k 个臂里均匀选"
        return report(name, False, f"ε = 1 时应完全均匀随机（每个臂约 {1 / k:.0%}），实际频率 {np.round(freq, 3).tolist()}", hint)
    a = call(Q, 0.2, no_noise)
    if a is None:
        return False
    want = 1 - 0.2 + 0.2 / k
    got = float(np.mean(a == best))
    if abs(got - want) > 0.015:
        hint = f"ε = 0.2 时选中臂 {best} 的概率应为 1 − ε + ε/k = {want:.2f}"
        if got > 0.99 or abs(got - 1 / k) < 0.02:
            hint = ("这一次调用里要么所有任务都在利用、要么都在探索：每个任务要各自决定是否探索——"
                    "rng.random(任务数) < epsilon，而不是用一个 rng.random() 管所有任务")
        elif abs(got - 0.8) < 0.015:
            hint = "探索时只从非贪心的臂里选了；要从全部 k 个臂里选，所以贪心臂的总概率是 1 − ε + ε/k"
        elif abs(got - (0.2 + 0.8 / k)) < 0.02:
            hint = "比较方向反了：rng.random(任务数) < epsilon 的任务才探索"
        return report(name, False, f"ε = 0.2 时臂 {best} 被选中的比例应约为 {want:.2f}，实际 {got:.3f}", hint)
    return report(name, True)


def _check_choose_ucb(fn):
    name = "UCB 选臂 choose_ucb"
    rng = np.random.default_rng(1)
    n, k, t, c = 500, 5, 200, 1.5
    Q = rng.normal(0.0, 1.0, (n, k))
    N = rng.integers(1, 50, (n, k)).astype(float)
    noise = 1e-9 * rng.random((n, k))
    Q_before, N_before = Q.copy(), N.copy()

    def call(Q_, N_, t_, c_, noise_):
        with np.errstate(divide="ignore", invalid="ignore"):
            ok, a = safe_call(name, fn, Q_, N_, t_, c_, noise_)
        if not ok:
            return None
        a = _arms(a)
        if a.shape != (len(Q_),):
            report(name, False, f"{len(Q_)} 个任务时应返回 {len(Q_)} 个臂下标，实际形状 {a.shape}",
                   "得分的形状与 Q 相同 (任务数, 臂数)：对每一行取 argmax，axis=1")
            return None
        return a

    def ucb_argmax(score):
        return np.argmax(score + noise, axis=1)

    a = call(Q, N, t, c, noise)
    if a is None:
        return False
    if not (np.array_equal(Q, Q_before) and np.array_equal(N, N_before)):
        return report(name, False, "函数改动了传进来的 Q 或 N", "选臂只读 Q、N；把 +∞ 写进一个新的 score 数组，不要写进 N")
    bonus = np.sqrt(np.log(t + 1) / N)
    if not np.array_equal(a, ucb_argmax(Q + c * bonus)):
        guesses = [(Q + 2.0 * bonus, "你用了 c = 2；要用传进来的 c（伯努利平台上用的是 √2）"),
                   (Q + c * np.log(t + 1) / N, "加分项忘了开平方：c·sqrt(ln(t + 1) / N)"),
                   (c * bonus, "得分 = 估计值 Q + 不确定性加分；你只用了加分"),
                   (Q - c * bonus, "加分是加上去的：拉得少的臂更值得试"),
                   (Q, "你只用了 Q（那是贪心）；还要加上 c·sqrt(ln(t + 1) / N)")]
        hint = "得分 = Q + c·sqrt(ln(t + 1) / N)，再对每一行取 argmax(得分 + tie_noise)"
        for score, h in guesses:
            if np.array_equal(a, ucb_argmax(score)):
                hint = h
                break
        if np.array_equal(a, np.argmin(Q + c * bonus + noise, axis=1)):
            hint = "用了 argmin；应选得分最高的臂"
        return report(name, False, f"t = {t}、c = {c}、所有臂都拉过时，有 {np.mean(a != ucb_argmax(Q + c * bonus)):.0%} 的任务"
                      "选的不是得分最高的臂", hint)
    # 得分并列时靠 tie_noise 打破
    a = call(np.zeros((n, k)), np.full((n, k), 3.0), t, c, noise)
    if a is None:
        return False
    if not np.array_equal(a, np.argmax(noise, axis=1)):
        return report(name, False, "所有臂的 Q 与 N 都相同（得分并列）时，应选 argmax(得分 + tie_noise)",
                      "最后一步是 np.argmax(score + tie_noise, axis=1)：用 tie_noise 随机打破平局")
    # 没拉过的臂优先：第 0 步与第 7 步
    N0 = N.copy()
    N0[:, 3:] = 0
    for t_ in (0, 7):
        a = call(Q, N0, t_, c, noise)
        if a is None:
            return False
        if not np.all(a >= 3):
            hint = "N = 0 的臂得分记为 +∞（score[N == 0] = np.inf），保证没拉过的臂先被试"
            if t_ == 0:
                hint = ("第 t 步（从 0 数）用 ln(t + 1)：t = 0 时 ln 0 = −∞，得分成了 NaN；"
                        "再把 N = 0 的臂得分记为 +∞")
            return report(name, False, f"t = {t_}、臂 3、4 还没拉过时，每个任务都应先拉它们，实际只有 {np.mean(a >= 3):.0%}", hint)
    return report(name, True)


def _check_update_mean(fn):
    name = "增量均值更新 update_mean"
    rng = np.random.default_rng(2)
    n, k = 8, 3
    Q = rng.normal(0.0, 1.0, (n, k))
    N = rng.integers(1, 6, (n, k)).astype(float)
    rows = np.arange(n)
    a = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    N[0, 0], Q[0, 0] = 0.0, 0.0                  # 任务 0 第一次拉臂 0：步长应是 1
    r = rng.normal(0.0, 1.0, n)
    Q0, N0, r0 = Q.copy(), N.copy(), r.copy()
    want_N = N0.copy()
    want_N[rows, a] += 1
    want_Q = Q0.copy()
    want_Q[rows, a] += (r0 - Q0[rows, a]) / want_N[rows, a]
    with np.errstate(divide="ignore", invalid="ignore"):
        ok, ret = safe_call(name, fn, Q, N, rows, a, r)
    if not ok:
        return False
    if np.array_equal(Q, Q0) and np.array_equal(N, N0):
        return report(name, False, "Q 和 N 都没有被原地修改",
                      "run_bandit 不接收返回值，要直接改传进来的数组：N[rows, a] += 1，再更新 Q[rows, a]")
    if not np.array_equal(r, r0):
        return report(name, False, "函数改动了传进来的奖励 r", "只读 r")
    if not np.array_equal(N, want_N):
        hint = "被拉的臂（每个任务一个）次数加 1：N[rows, a] += 1；其余位置不变"
        if np.array_equal(N, N0):
            hint = "N 没有更新：先 N[rows, a] += 1，Q 的步长用加 1 之后的次数"
        return report(name, False, "更新之后的次数 N 不对", hint)
    mask = np.ones_like(Q, dtype=bool)
    mask[rows, a] = False
    if not np.allclose(Q[mask], Q0[mask]):
        return report(name, False, "没被拉的臂的 Q 也变了", "只更新 Q[rows, a]（每个任务被拉的那一个臂）")
    if np.allclose(Q, want_Q, atol=1e-12):
        return report(name, True)
    with np.errstate(divide="ignore", invalid="ignore"):
        old_step = Q0[rows, a] + (r0 - Q0[rows, a]) / N0[rows, a]
    got = Q[rows, a]
    if not np.all(np.isfinite(got)) or np.allclose(got[1:], old_step[1:]):
        hint = "步长要用加 1 之后的次数：先 N[rows, a] += 1，再 Q[rows, a] += (r − Q[rows, a]) / N[rows, a]（第一次拉时步长是 1）"
    elif np.allclose(got, r0):
        hint = "你直接把 Q 换成了新奖励；增量均值只往 r 走 1/N 步"
    else:
        hint = "Q[rows, a] += (r − Q[rows, a]) / N[rows, a]，N 是加 1 之后的次数"
    return report(name, False, f"更新之后被拉的臂的 Q 不对：期望 {np.round(want_Q[rows, a], 4).tolist()}，"
                  f"实际 {np.round(got, 4).tolist()}", hint)


def bandit_rules(eps_fn, ucb_fn, update_fn):
    """讲义 S1 的三个向量化小函数（一次处理很多个任务；Q、N 的形状都是 (任务数, 臂数)）：
    eps_fn(Q, epsilon, rng, tie_noise) → 每个任务选的臂 (任务数,)：各自以概率 ε 从全部臂里均匀选，否则 argmax(Q + tie_noise)；
    ucb_fn(Q, N, t, c, tie_noise) → 每个任务 argmax(Q + c·sqrt(ln(t + 1)/N) + tie_noise)，N = 0 的臂得分 +∞；
    update_fn(Q, N, rows, a, r)：原地把 N[rows, a] 加 1，Q[rows, a] 按增量均值往 r 走 1/N 步。
    三个函数分别检查，各打印一行结果；全部通过才返回 True。"""
    results = [_check_choose_eps_greedy(eps_fn), _check_choose_ucb(ucb_fn), _check_update_mean(update_fn)]
    return all(results)


def bandit_final_reward(eps0, eps01, eps1, ucb):
    """四种方法在 10 臂测试平台上最后 100 步的平均奖励（2000 个任务平均）：ε = 0、0.01、0.1 与 UCB c = 2。"""
    name = "10 臂测试平台的最终平均奖励"
    vals, ok = _floats(name, (eps0, eps01, eps1, ucb), 4)
    if vals is None:
        return ok
    e0, e01, e1, u = vals
    hint = ("2000 个任务 × 1000 步、样本平均更新、Q 初值 0，取最后 100 步的平均奖励。本机实测（种子 0）："
            "ε = 0 → 1.02，ε = 0.01 → 1.29，ε = 0.1 → 1.37，UCB c = 2 → 1.49；"
            "这个测试平台上能拿到的最好平均奖励约为 E[max q*] ≈ 1.54")
    bands = [("ε = 0（贪心）", e0, 0.90, 1.15), ("ε = 0.01", e01, 1.18, 1.38),
             ("ε = 0.1", e1, 1.30, 1.44), ("UCB c = 2", u, 1.42, 1.56)]
    for label, v, lo, hi in bands:
        if not lo <= v <= hi:
            return report(name, False, f"{label} 应落在 [{lo}, {hi}]，实际 {v:.4f}", hint)
    if not (u > e1 > e0 and e01 > e0):
        return report(name, False,
                      f"排序应为 UCB > ε = 0.1 > ε = 0（且 ε = 0.01 > ε = 0），实际 {u:.3f} / {e1:.3f} / {e01:.3f} / {e0:.3f}",
                      hint)
    return report(name, True)


# ---------- S1 · Thompson 采样（伯努利 10 臂测试平台） ----------
THOMPSON_WIN_PROB = 113 / 126      # P(θ0 > θ1)，θ0 ~ Beta(4, 2)、θ1 ~ Beta(2, 4)：3 成 1 败 vs 1 成 3 败，先验 Beta(1, 1)


def _arms(a):
    """学习者返回的臂下标 → 一维数组；无法转换时返回空数组（由调用方报告形状不对）。"""
    if a is None:
        return np.array([])
    try:
        return _np(a).reshape(-1)
    except (TypeError, ValueError):
        return np.array([])


def thompson_choose(fn):
    """fn(successes, failures, rng)：successes、failures 的形状是 (任务数, 臂数)，记每个任务里每个臂成功、失败的次数；
    返回每个任务选中的臂，形状 (任务数,)。Thompson 采样：每个臂从后验 Beta(1 + 成功数, 1 + 失败数) 里抽一个 θ，
    选抽到最大的臂——所以一个臂被选中的概率 = 后验下"它是最优臂"的概率（概率匹配）。"""
    name = "Thompson 采样选臂 thompson_choose"
    rng = np.random.default_rng(0)
    n = 20000
    # 情形 1：臂 0 成功 3 次、失败 1 次；臂 1 成功 1 次、失败 3 次 → 后验 Beta(4, 2) vs Beta(2, 4)
    S, F = np.tile([3.0, 1.0], (n, 1)), np.tile([1.0, 3.0], (n, 1))
    S0, F0 = S.copy(), F.copy()
    ok, a = safe_call(name, fn, S, F, rng)
    if not ok:
        return False
    if not (np.array_equal(S, S0) and np.array_equal(F, F0)):
        return report(name, False, "函数改动了传进来的成功 / 失败次数", "选臂只读这两个数组；更新计数是训练循环的事")
    a = _arms(a)
    if a.shape != (n,) or not np.all((a == 0) | (a == 1)):
        return report(name, False, f"{n} 个任务、每个任务 2 个臂时，应返回 {n} 个臂下标（0 或 1），实际形状 {a.shape}",
                      "theta 的形状与 successes 相同 (任务数, 臂数)：对每一行取 argmax，即 np.argmax(theta, axis=1)")
    got = float((a == 0).mean())
    if got > 0.995:
        return report(name, False, f"后验 Beta(4, 2) vs Beta(2, 4) 时几乎总是选臂 0（{got:.3f}）——这是贪心（选后验均值最大的臂），"
                      "不是 Thompson 采样",
                      "要从每个臂的 Beta 后验里**抽样**一个 θ（rng.beta），再对抽到的 θ 取 argmax；不要用均值")
    if abs(got - THOMPSON_WIN_PROB) > 0.02:
        hint = "后验是 Beta(1 + 成功数, 1 + 失败数)：第一个参数配成功、第二个配失败，两个都要加先验的 1"
        if abs(got - (1 - THOMPSON_WIN_PROB)) < 0.02:
            hint = "成功与失败放反了：rng.beta(1 + successes, 1 + failures)——第一个参数越大，θ 越接近 1"
        elif abs(got - 0.95) < 0.02:
            hint = "漏了先验的 1：后验是 Beta(1 + 成功数, 1 + 失败数)，不是 Beta(成功数, 失败数)（没拉过的臂还会让 rng.beta 报错）"
        elif abs(got - 0.5) < 0.03:
            hint = "两个臂被选中的机会几乎相同：抽样时有没有用到 successes 与 failures？"
        return report(name, False, f"臂 0 三成一败、臂 1 一成三败时，选臂 0 的比例应 ≈ P(θ0 > θ1)（约 0.90），实际 {got:.3f}", hint)
    # 情形 2：4 个臂都没拉过 → 后验相同，应均匀选
    Z = np.zeros((n, 4))
    ok, a = safe_call(name, fn, Z, Z.copy(), rng)
    if not ok:
        return False
    a = _arms(a)
    if a.shape != (n,) or not np.all(np.isin(a, [0, 1, 2, 3])):
        return report(name, False, f"4 个臂时应返回 {n} 个 0–3 之间的臂下标，实际形状 {a.shape}")
    freq = np.bincount(a.astype(int), minlength=4) / n
    if np.abs(freq - 0.25).max() > 0.02:
        return report(name, False, f"4 个都没拉过的臂后验相同，应被均匀选中（各约 0.25），实际 {np.round(freq, 3)}",
                      "从 Beta(1, 1) 里各抽一个 θ 再取 argmax，四个臂的机会均等；如果总是选第一个，说明没有抽样")
    return report(name, True)


def bernoulli_bandit_regret(eps01, ucb1, thompson):
    """伯努利 10 臂测试平台（q*(a) ~ U(0, 1)，2000 个任务 × 1000 步）上 1000 步累计期望遗憾，5 组任务（种子 0–4）的平均：
    ε = 0.1、UCB1（c = √2）、Thompson 采样。"""
    name = "伯努利 10 臂测试平台的累计遗憾"
    vals, ok = _floats(name, (eps01, ucb1, thompson), 3)
    if vals is None:
        return ok
    e, u, t = vals
    hint = ("2000 个任务 × 1000 步，传入 5 组任务（种子 0–4）1000 步累计期望遗憾的平均。本机实测："
            "ε = 0.1 → 68.2 ± 0.7，UCB1（c = √2）→ 130.3 ± 0.4，Thompson → 28.2 ± 0.2")
    if 170 <= u <= 190:
        return report(name, False, f"UCB1 应落在 [120, 140]，实际 {u:.1f}",
                      "这正是把讲义 S1 高斯平台上的 c = 2 搬过来的结果（本机 179.9）。c = 2 是给方差 1 的奖励定的；"
                      "0/1 奖励这里用 [0, 1] 奖励的常用值 c = √2——探索系数要按奖励尺度调")
    bands = [("ε = 0.1", e, 60, 76, "ε-greedy 用样本平均更新 Q、Q 初值 0、平局随机打破"),
             ("UCB1（c = √2）", u, 120, 140, "UCB 得分 Q + c·sqrt(ln(t + 1) / N)，没拉过的臂得分 +∞，c = √2"),
             ("Thompson", t, 24, 33, "Thompson 偏离时先查：后验 Beta(1 + 成功, 1 + 失败)、失败数 = N − S、每一步都重新抽样"
                                    "（先用 ck.thompson_choose 单独检查选臂函数）")]
    for label, v, lo, hi, h in bands:
        if not lo <= v <= hi:
            return report(name, False, f"{label} 应落在 [{lo}, {hi}]，实际 {v:.1f}", f"{h}。{hint}")
    return report(name, t < e < u, f"排序应为 Thompson < ε = 0.1 < UCB1，实际 {t:.1f} / {e:.1f} / {u:.1f}", hint)


def beta_win_probability(p):
    """练习 E7：臂 0 三成一败、臂 1 一成三败、先验 Beta(1, 1) 时，P(θ0 > θ1) 的抽样估计（10⁶ 对样本）。"""
    name = "E7 · 后验下臂 0 比臂 1 好的概率"
    vals, ok = _floats(name, (p,), 1)
    if vals is None:
        return ok
    v = vals[0]
    if abs(v - THOMPSON_WIN_PROB) <= 0.005:
        return report(name, True)
    hint = "臂 0 的后验是 Beta(1 + 3, 1 + 1)，臂 1 是 Beta(1 + 1, 1 + 3)；各抽 10⁶ 个样本，估计 = (θ0 样本 > θ1 样本) 的比例"
    if abs(v - (1 - THOMPSON_WIN_PROB)) < 0.01:
        hint = "成功与失败放反了（或者算的是 P(θ1 > θ0)）：Beta 的第一个参数配成功"
    elif abs(v - 0.95) < 0.01:
        hint = "漏了先验的 1：后验是 Beta(1 + 成功, 1 + 失败) = Beta(4, 2) 与 Beta(2, 4)"
    elif v > 0.999:
        hint = "比较的是两个后验的均值（2/3 > 1/3，永远为真）；要从两个后验里各抽样，逐对比较"
    elif abs(v - 0.5) < 0.01:
        hint = "两个臂用了同一个后验？"
    elif not 0.0 <= v <= 1.0:
        hint = "概率应在 0 与 1 之间：(样本0 > 样本1).mean()"
    return report(name, False, f"估计值 {v:.4f} 与精确值相差太多（10⁶ 个样本时抽样误差约 0.0003）", hint)


# ---------- S2 · MDP 与 Q-learning ----------
def _bellman_guess(fn, Q0, cases):
    """更新值不对时猜是哪种常见错误。在**全部**用例上重跑学习者的函数，只有每个用例都与某个错误公式吻合时
    才给出对应提示——单个用例可能同时符合好几种错误（只有 2 个动作时，Q[s', a] 不是 max 就是 min）。"""
    got = []
    for s, a, r, s2, term, alpha, gamma in cases:
        Q = Q0.copy()
        try:
            ret = fn(Q, s, a, r, s2, term, alpha, gamma)
        except Exception:  # noqa: BLE001  （出错的情况主循环已经报告过）
            return None
        table = Q
        if ret is not None and np.array_equal(Q, Q0):
            try:
                arr = _np(ret)
            except (TypeError, ValueError):
                arr = None
            if arr is not None and arr.shape == Q0.shape:
                table = arr
        got.append(float(table[s, a]))

    def step(s, a, alpha, target):
        return Q0[s, a] + alpha * (target - Q0[s, a])

    mistakes = [
        (lambda s, a, r, s2, term, al, g: step(s, a, al, r + g * Q0[s2].max()),
         "terminated 为 True 时没有明天：目标就是 r，不再加 γ · max Q[s']"),
        (lambda s, a, r, s2, term, al, g: step(s, a, al, r if term else r + g * Q0[s2, a]),
         "用了 Q[s', a]（同一个动作）；Q-learning 的目标取下一状态里**最好**的动作：Q[s'].max()"),
        (lambda s, a, r, s2, term, al, g: step(s, a, al, r if term else r + g * Q0[s2].min()),
         "用了 min；应该是 max"),
        (lambda s, a, r, s2, term, al, g: r if term else r + g * Q0[s2].max(),
         "直接把目标赋给了 Q[s, a]，忘了学习率 α：Q[s, a] += α · (目标 − Q[s, a])"),
        (lambda s, a, r, s2, term, al, g: step(s, a, al, r if term else r + Q0[s2].max()),
         "忘了折扣 γ：目标 = r + γ · max Q[s']"),
        (lambda s, a, r, s2, term, al, g: step(s, a, al, r if term else r + g * Q0[s].max()),
         "用了当前状态 s 的 max；目标里的 max 取的是**下一状态** s' 的那一行"),
    ]
    for f, hint in mistakes:
        if all(abs(v - f(*case)) < 1e-9 for v, case in zip(got, cases)):
            return hint
    return None


def bellman_update(fn):
    """fn(Q, s, a, r, s_next, terminated, alpha, gamma)：原地更新 Q[s, a]，并返回更新后的值。
    目标 = r + γ · max_a' Q[s_next, a']；terminated 为 True 时目标 = r（终止之后没有"明天"）。"""
    name = "Q-learning 的一步 Bellman 更新"
    Q0 = np.array([[0.0, 0.5],
                   [1.0, 2.0],
                   [-0.5, 0.25]])
    cases = [  # (s, a, r, s_next, terminated, alpha, gamma)
        (0, 0, 1.0, 1, False, 0.5, 0.9),
        (2, 0, 0.0, 1, False, 0.1, 0.99),
        (1, 0, 0.0, 2, True, 0.5, 0.9),
        (1, 1, -1.0, 0, False, 1.0, 0.5),
    ]
    for s, a, r, s2, term, alpha, gamma in cases:
        Q = Q0.copy()
        ok, ret = safe_call(name, fn, Q, s, a, r, s2, term, alpha, gamma)
        if not ok:
            return False
        target = r if term else r + gamma * Q0[s2].max()
        want = Q0[s, a] + alpha * (target - Q0[s, a])
        desc = (f"Q[{s}, {a}] = {Q0[s, a]}，r = {r}，s' = {s2}，terminated = {term}，α = {alpha}，γ = {gamma}")
        ret_arr = None
        if ret is not None:
            try:
                ret_arr = _np(ret)
            except (TypeError, ValueError):
                return report(name, False, "返回值应是更新后的 Q[s, a]（一个数）", "最后 return Q[s, a]")
        table = Q
        if ret_arr is not None and ret_arr.shape == Q0.shape and np.array_equal(Q, Q0):
            table = ret_arr                     # 返回了一张新表、没有原地改：也按新表检查数值
        mask = np.ones_like(Q0, dtype=bool)
        mask[s, a] = False
        if not np.allclose(table[mask], Q0[mask]):
            return report(name, False, f"{desc}：除了 Q[s, a] 之外，表里的其他格也被改了",
                          "一次更新只改 Q[s, a] 这一格")
        got = table[s, a]
        if abs(got - Q0[s, a]) < 1e-12 and abs(want - Q0[s, a]) > 1e-12:
            if ret_arr is not None and ret_arr.size == 1 and abs(float(ret_arr) - want) < 1e-9:
                return report(name, False, f"{desc}：返回值对了，但 Q 表本身没有被更新",
                              "要原地修改：Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])，再 return Q[s, a]")
            return report(name, False, f"{desc}：Q[s, a] 没有变化", "要原地修改 Q[s, a]")
        if abs(got - want) > 1e-9:
            return report(name, False, f"{desc}：期望新值 {want:.4f}，实际 {got:.4f}",
                          _bellman_guess(fn, Q0, cases)
                          or "目标 = r + γ · max Q[s']（终止时 = r）；新值 = 旧值 + α · (目标 − 旧值)")
        if ret_arr is not None and ret_arr.size == 1 and abs(float(ret_arr) - want) > 1e-9:
            return report(name, False, f"{desc}：Q 表更新对了，但返回值 {float(ret_arr):.4f} 不是更新后的 Q[s, a]",
                          "最后 return Q[s, a]")
    return report(name, True)


def frozenlake_success(rate, slippery=False):
    """贪心策略在 FrozenLake 4×4 上的成功率。slippery=False：不滑冰面（HF Unit 2 设定）；
    slippery=True：滑冰面、lr 0.1（理论上限约 0.73，见下面的提示）。"""
    if not slippery:
        return _in_range("FrozenLake（不滑）贪心策略成功率", rate, 0.95, 1.0,
                         "HF Unit 2 超参数：10000 回合、lr 0.7、γ 0.95、max_steps 99、ε 从 1.0 按 "
                         "exp(−0.0005·回合) 衰减到 0.05。不滑冰面上训练好的贪心策略应每次都到终点（本机 5/5 个种子 1.00）。"
                         "接近 0 时先查：argmax 平局有没有随机打破（Q 表全 0 时 np.argmax 永远选 0 = 向左）")
    return _in_range("FrozenLake（滑）贪心策略成功率", rate, 0.50, 0.77,
                     "滑冰面、lr 0.1、10000 回合：本机 5 个种子各 0.40–0.72、均值约 0.61。"
                     "γ = 0.95 的最优策略在 99 步内的成功率是 0.728（100 步 0.730；讲义 S2 末尾用价值迭代精确算出），"
                     "所以超过 0.77 说明评估的环境或步数上限不对"
                     "（例如评估时用了不滑的冰面、或者去掉了 99 步的上限）；"
                     "明显偏低时检查评估是否还在探索（评估要用纯贪心策略，本机评估时保留 ε = 0.1 会从约 0.61 降到约 0.35）、"
                     "是否把训练回合也算了进去，以及学习率是否过大（随机转移下 lr 0.7 太大）")


def argmax_random_tiebreak(fn):
    """fn(values, rng)：返回最大值的下标；有多个最大值时在它们之间均匀随机选。"""
    name = "随机打破平局的 argmax"
    rng = np.random.default_rng(0)
    cases = [np.zeros(4), np.array([1.0, 3.0, 3.0, 0.0]), np.array([5.0, 1.0, 2.0]),
             np.array([-1.0, -1.0, -3.0]), np.array([0.2, 0.7, 0.1, 0.7, 0.7])]
    for vals in cases:
        before = vals.copy()
        n = 4000
        c = np.zeros(len(vals))
        for _ in range(n):
            ok, i = safe_call(name, fn, vals, rng)
            if not ok:
                return False
            try:
                i = int(i)
            except (TypeError, ValueError):
                return report(name, False, f"应返回一个下标（整数），实际是 {type(i).__name__}")
            if not 0 <= i < len(vals):
                return report(name, False, f"返回的下标 {i} 越界（长度 {len(vals)}）")
            c[i] += 1
        if not np.array_equal(vals, before):
            return report(name, False, "函数改动了传进来的数组", "只读，不要原地修改")
        winners = np.flatnonzero(vals == vals.max())
        freq = c / n
        if freq[np.setdiff1d(np.arange(len(vals)), winners)].sum() > 0:
            return report(name, False, f"输入 {vals.tolist()}：选到了不是最大值的位置",
                          "只能在最大值所在的位置里随机：np.flatnonzero(values == values.max())")
        if len(winners) > 1:
            if freq[winners[0]] == 1.0:
                return report(name, False, f"输入 {vals.tolist()}：平局时你总是返回第一个最大值（下标 {winners[0]}）",
                              "这正是 np.argmax 的行为——FrozenLake 里 Q 表全 0 时它永远选 0（向左）。"
                              "要在所有最大值里均匀随机选：rng.choice(np.flatnonzero(values == values.max()))")
            if np.abs(freq[winners] - 1 / len(winners)).max() > 0.04:
                return report(name, False,
                              f"输入 {vals.tolist()}：{len(winners)} 个最大值应各约 {1 / len(winners):.0%}，"
                              f"实际 {np.round(freq[winners], 3)}",
                              "要**均匀**随机：rng.choice(候选下标)")
    return report(name, True)


# ---------- S2 · 价值迭代：Q-learning 的阳性对照 ----------
def _two_rooms():
    """课程自造的小 MDP（格式同 Gymnasium 的 env.unwrapped.P）：状态 0 = 走廊、1 = 门口；动作 0 = 原地、1 = 往前。
    走廊原地：+0.05，留在走廊；走廊往前：0.8 到门口、0.2 留在走廊，奖励 0；
    门口原地：+0，留在门口；门口往前：+1，回合结束（terminated=True，下一状态写成 0——专门检查"终止时不再加 γV(s')"）。"""
    return {0: {0: [(1.0, 0, 0.05, False)], 1: [(0.8, 1, 0.0, False), (0.2, 0, 0.0, False)]},
            1: {0: [(1.0, 1, 0.0, False)], 1: [(1.0, 0, 1.0, True)]}}


def _two_rooms_solution(gamma):
    """闭式解：门口 V1 = max(γ·V1, 1) = 1；走廊"往前"V0 = 0.8γ·V1 / (1 − 0.2γ)，"原地"0.05 / (1 − γ)。
    γ = 0.9 与 0.5 时最优动作都是"往前"（只用来检查备份写得对不对）。"""
    v1 = 1.0
    return np.array([max(0.8 * gamma * v1 / (1 - 0.2 * gamma), 0.05 / (1 - gamma)), v1])


def _model_arrays(P, n_states, n_actions):
    """与讲义给定的 model_arrays 相同：R[s, a] = Σ p·r（期望即时奖励）；C[s, a, s'] = 到达 s' 且回合没有结束的概率
    （terminated 的转移不进 C，所以终止之后不会再加 γ·V(s')）。"""
    R, C = np.zeros((n_states, n_actions)), np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            for p, s2, r, done in P[s][a]:
                R[s, a] += p * r
                if not done:
                    C[s, a, s2] += p
    return R, C


def _ref_bellman_optimal_values(R, C, gamma, tol=1e-10, variant="ok", max_sweeps=10_000):
    """参考实现与几种常见错误写法（只用来给提示）。"""
    V = np.zeros(R.shape[0])
    Q = np.zeros_like(R)
    for sweep in range(1, max_sweeps + 1):
        if variant == "current_state":                     # 用了当前状态的 V(s)，不是下一状态的 V(s')
            Q = R + gamma * C.sum(axis=2) * V[:, None]
        else:
            Q = R + gamma * np.einsum("ijk,k->ij", C, V)
        V_new = Q.mean(axis=1) if variant == "mean" else Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q, sweep
        V = V_new
    return V, Q, max_sweeps


def _vi_output(name, out, n_states, n_actions):
    """把 (V, Q, 扫描次数) 拆开并检查形状；不合格时返回 (None, 报告结果)。"""
    try:
        V, Q, sweeps = out
    except (TypeError, ValueError):
        return None, report(name, False, "应返回三个值 (V, Q, 扫描次数)", "停止时 return V_new, Q, sweeps")
    try:
        V, Q, sweeps = _np(V), _np(Q), int(sweeps)
    except (TypeError, ValueError):
        return None, report(name, False, "V、Q 应是数组，扫描次数应是整数")
    if V.shape != (n_states,) or Q.shape != (n_states, n_actions):
        return None, report(name, False, f"V 的形状应为 (状态数,) = ({n_states},)、Q 为 (状态数, 动作数) = ({n_states}, {n_actions})，"
                            f"实际 {V.shape}、{Q.shape}",
                            "Q 与 R 同形状 (状态数, 动作数)；V 是对动作（axis=1）取 max")
    if not (np.all(np.isfinite(V)) and np.all(np.isfinite(Q))):
        return None, report(name, False, "V 或 Q 里有非有限数（发散了？）", "别忘了乘 γ；0 ≤ γ < 1 时价值迭代一定收敛")
    return (V, Q, sweeps), True


def _vi_consistent(name, V, Q):
    if np.allclose(Q.max(axis=1), V, atol=1e-6):
        return True
    return report(name, False, "V 的数值对了，但返回的 Q 与它对不上：应有 V(s) = max_a Q(s, a)",
                  "返回的 V 与 Q 要来自同一次扫描：V_new = Q.max(axis=1)，停止时 return V_new, Q, sweeps")


def _call_vi(name, fn, P, n_states, n_actions, gamma):
    """在 P 上调 model_arrays 再把数组交给学习者的函数；检查没有改动 R、C，返回 ((V, Q, 扫描次数), True) 或 (None, 结果)。"""
    R, C = _model_arrays(P, n_states, n_actions)
    R0, C0 = R.copy(), C.copy()
    ok, out = safe_call(name, fn, R, C, gamma, 1e-10)
    if not ok:
        return None, False
    if not (np.array_equal(R, R0) and np.array_equal(C, C0)):
        return None, report(name, False, "函数改动了传进来的 R 或 C", "价值迭代只读 R、C；每一遍算出新的 Q 与 V_new")
    return _vi_output(name, out, n_states, n_actions)


def bellman_optimal_values(fn):
    """fn(R, C, gamma, tol) → (V, Q, 扫描次数)。R、C 是讲义给定的 model_arrays 的输出：R[s, a] = 期望即时奖励，
    C[s, a, s'] = 到达 s' 且回合没有结束的概率（terminated 的转移不在 C 里）。
    备份 Q(s, a) = R[s, a] + γ·Σ_s' C[s, a, s']·V(s')、V(s) = max_a Q(s, a)，停止条件 max|V_new − V| < tol。
    检查时先对 P 调 model_arrays 再传数组：先在课程自造的"两个房间"小 MDP（γ = 0.9、0.5；门口"往前"得 +1 后
    回合结束，价值应恰好是 1——终止的转移不再自举）上对闭式解，再在 FrozenLake 4×4（γ = 0.95，滑与不滑）上
    对 V*(起点) 与滑冰面的扫描次数。"""
    name = "价值迭代 bellman_optimal_values"
    P = _two_rooms()
    for gamma in (0.9, 0.5):
        res, ok = _call_vi(name, fn, P, 2, 2, gamma)
        if res is None:
            return ok
        V, Q, _ = res
        want = _two_rooms_solution(gamma)
        if np.allclose(V, want, atol=1e-6):
            if _vi_consistent(name, V, Q) is not True:
                return False
            continue
        R, C = _model_arrays(P, 2, 2)
        hint = ("备份：Q(s, a) = R[s, a] + γ·Σ_s' C[s, a, s']·V(s')，V(s) = max_a Q(s, a)；"
                "停止条件：这一遍与上一遍 V 的最大变化 < tol")
        guesses = [("current_state", "Σ_s' C[s, a, s']·V(s') 里用的是**下一状态** s' 的价值：对 C 的最后一维（s'）和 V 相乘再求和"),
                   ("mean", "你对动作取了平均（那是'均匀随机策略'的价值）；最优价值要对动作取 max")]
        for variant, h in guesses:
            if np.allclose(V, _ref_bellman_optimal_values(R, C, gamma, variant=variant)[0], atol=1e-4):
                hint = h
                break
        else:
            if np.allclose(V, want, atol=1e-2):
                hint = "数值接近但精度不够：停止条件要用传进来的 tol（这里是 1e-10），不要写死别的值（比如 1e-3）"
        return report(name, False, f"自造小 MDP（走廊 → 门口，γ = {gamma}）：期望 V = {np.round(want, 6).tolist()}，"
                      f"实际 {np.round(V, 6).tolist()}", hint)
    import gymnasium as gym
    for slippery in (False, True):
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery)
        res, ok = _call_vi(name, fn, env.unwrapped.P, 16, 4, 0.95)
        if res is None:
            return ok
        V, Q, sweeps = res
        label = "滑" if slippery else "不滑"
        v0 = FROZENLAKE_V0[slippery]
        if abs(V[0] - v0) > 1e-4:
            return report(name, False, f"FrozenLake 4×4（{label}，γ = 0.95）起点的 V* 应为 {v0:.6f}，实际 {V[0]:.6f}",
                          "Σ_s' C[s, a, s']·V(s') 要对**所有**下一状态求和（滑冰面每个动作有 3 个可能的结果）；"
                          "停止条件用传进来的 tol")
        if _vi_consistent(name, V, Q) is not True:
            return False
        if slippery and not 150 <= sweeps <= 400:
            return report(name, False, f"滑冰面 tol = 1e-10 时本机需要 241 次扫描，实际 {sweeps}",
                          "停止条件：这一遍与上一遍 V 的最大变化 < tol；每扫描一遍（所有状态备份一次）计数加 1")
    return report(name, True)


def greedy_as_policy(fn):
    """fn(Q) → 与 Q 同形状的概率表 pi：每一行在并列最大的动作上平分概率，其余为 0
    （与讲义 greedy_policy 的"随机打破平局"是同一个策略）。"""
    name = "贪心策略写成概率表 greedy_as_policy"
    Q = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.1, 0.5, 0.5, 0.2],
                  [1.0, 0.2, 0.3, 0.0],
                  [-1.0, -2.0, -1.0, -1.0],
                  [0.3, 0.1, 0.2, 0.3]])
    Q_before = Q.copy()
    best = (Q == Q.max(axis=1, keepdims=True)).astype(float)
    want = best / best.sum(axis=1, keepdims=True)
    ok, pi = safe_call(name, fn, Q)
    if not ok:
        return False
    if not np.array_equal(Q, Q_before):
        return report(name, False, "函数改动了传进来的 Q", "只读 Q，新建一张概率表返回")
    if pi is None:
        return report(name, False, "函数没有返回值（返回了 None）", "最后 return 概率表")
    try:
        pi = _np(pi)
    except (TypeError, ValueError):
        return report(name, False, f"应返回一个与 Q 同形状的数组，实际是 {type(pi).__name__}")
    if pi.shape != Q.shape:
        return report(name, False, f"应返回与 Q 同形状 {Q.shape} 的概率表，实际 {pi.shape}",
                      "每一行对应一个状态、每一列对应一个动作")
    if np.allclose(pi, want):
        return report(name, True)
    if np.allclose(pi, np.eye(Q.shape[1])[Q.argmax(axis=1)]):
        hint = ("并列时只给了第一个动作（np.argmax 的行为）——讲义的贪心策略在并列最大的动作之间均匀随机选，"
                "所以要平分：best = (Q == Q.max(axis=1, keepdims=True))，再按行除以并列的个数")
    elif np.allclose(pi, best):
        hint = "每一行要除以并列的个数，概率之和才是 1：best / best.sum(axis=1, keepdims=True)"
    elif np.allclose(pi.sum(axis=1), 1) and pi.min() > 0:
        hint = "非最大的动作概率应为 0——贪心策略不会选它们（你也许用了 softmax）"
    elif not np.allclose(pi.sum(axis=1), 1):
        hint = f"每一行的概率之和应为 1，实际 {np.round(pi.sum(axis=1), 3).tolist()}"
    else:
        hint = "best = (Q == Q.max(axis=1, keepdims=True)).astype(float)；pi = best / best.sum(axis=1, keepdims=True)"
    row = int(np.flatnonzero(~np.isclose(pi, want).all(axis=1))[0])
    return report(name, False, f"Q 的第 {row} 行 {Q[row].tolist()}：期望 {np.round(want[row], 3).tolist()}，"
                  f"实际 {np.round(pi[row], 3).tolist()}", hint)


def optimal_success(success, v_start):
    """滑冰面 4×4、γ = 0.95：价值迭代得到的最优策略在 99 步之内到达 G 的精确概率，与起点的最优价值 V*(0)。"""
    name = "滑冰面的最优值与最优成功率"
    vals, ok = _floats(name, (success, v_start), 2)
    if vals is None:
        return ok
    s, v = vals
    if abs(v - FROZENLAKE_V0[True]) > 1e-3:
        hint = "传入滑冰面价值迭代的 V[0]（γ = 0.95）"
        if abs(v - FROZENLAKE_V0[False]) < 1e-3:
            hint = "这是不滑冰面的 V*（0.95⁵）；第二个数要传滑冰面的 V*(起点)"
        elif abs(v - FROZENLAKE_OPT_SUCCESS[99]) < 1e-3:
            hint = "两个数传反了：第一个是成功率，第二个是 V*(起点)"
        return report(name, False, f"V*(起点) 应约为 0.1805，实际 {v:.4f}", hint)
    if abs(s - v) < 1e-3:
        return report(name, False, "成功率与 V*(起点) 一样——V* 是打折后（γ = 0.95）的价值，不是成功率",
                      "成功率要另算：用 success_probability(P, 策略, 99)，不打折、只走 99 步")
    if abs(s - FROZENLAKE_OPT_SUCCESS[99]) <= 5e-4:
        return report(name, True)
    hint = "success_probability(P, greedy_as_policy(Q_slip), 99)[0]：起点（状态 0）在 99 步内到达 G 的概率"
    for horizon, h in ((100, "这是 100 步内的成功率；讲义 evaluate_agent 每局最多走 99 步，horizon 用 99"),
                       (200, "这是 200 步内的成功率；讲义的口径是 99 步"),
                       (None, "这是不限步数的成功率；讲义 evaluate_agent 每局最多走 99 步，horizon 用 99")):
        if abs(s - FROZENLAKE_OPT_SUCCESS[horizon]) < 5e-4:
            hint = h
            break
    return report(name, False, f"99 步内的最优成功率应约为 0.728，实际 {s:.4f}", hint)


# ---------- S3 · REINFORCE ----------
def _ref_reward_to_go(rewards, gamma):
    G = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def reward_to_go(fn):
    """fn(rewards, gamma)：返回与 rewards 等长的数组，G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …（只看 t 及之后的奖励）。"""
    name = "reward-to-go（从每一步往后的折扣回报）"
    rng = np.random.default_rng(0)
    cases = [([1.0, 1.0, 1.0, 1.0], 1.0), ([1.0, 0.0, 2.0], 0.9), ([3.0, -1.0, 2.0], 0.0),
             ([5.0], 0.99), (list(np.round(rng.normal(size=30), 2)), 0.95), ([1.0] * 50, 0.99)]
    for rewards, gamma in cases:
        r_before = list(rewards)
        ok, got = safe_call(name, fn, list(rewards), gamma)
        if not ok:
            return False
        if rewards != r_before:
            return report(name, False, "函数改动了传进来的奖励列表")
        try:
            got = _np(got).reshape(-1)
        except (TypeError, ValueError):
            return report(name, False, f"应返回一个数组，实际是 {type(got).__name__}")
        want = _ref_reward_to_go(rewards, gamma)
        short = rewards if len(rewards) <= 6 else f"长度 {len(rewards)} 的序列"
        if got.shape != want.shape:
            return report(name, False, f"rewards = {short}：应返回长度 {len(want)} 的数组，实际长度 {got.size}",
                          "每个时间步都有自己的 G_t")
        if np.allclose(got, want, atol=1e-6):
            continue
        r = np.asarray(rewards)
        disc = gamma ** np.arange(len(r))
        if np.allclose(got, np.cumsum(r * disc), atol=1e-6) or np.allclose(got, np.cumsum(r), atol=1e-6):
            hint = "方向反了：你算的是从开头累加到 t；G_t 要从 t 往**后**加（倒着循环最方便）"
        elif np.allclose(got, np.full(len(r), (r * disc).sum()), atol=1e-6):
            hint = "每一步都用了整局的回报；reward-to-go 只算 t 及之后的奖励（过去的奖励与现在的动作无关）"
        elif np.allclose(got, _ref_reward_to_go(rewards, 1.0), atol=1e-6):
            hint = "忘了折扣 γ：倒着循环时 running = r_t + γ · running"
        elif np.allclose(got, np.array([(r[t:] * disc[t:]).sum() for t in range(len(r))]), atol=1e-6):
            hint = "折扣的指数要从 t 重新数：G_t = r_t + γ r_{t+1} + …，r_t 本身不打折"
        else:
            hint = "倒着循环：running = 0；for t in reversed(range(T)): running = rewards[t] + gamma * running; G[t] = running"
        return report(name, False,
                      f"rewards = {short}、γ = {gamma}：期望 {np.round(want[:6], 4).tolist()}…，实际 {np.round(got[:6], 4).tolist()}…",
                      hint)
    return report(name, True)


def moving_baseline(fn):
    """fn(b, G, beta)：b 是长度 MAX_T 的数组，b[t] 记录过去各回合在时间步 t 的 reward-to-go 的指数滑动平均；
    G 是这一回合的 reward-to-go（长度 T ≤ MAX_T）。只更新 b[:T]：b[t] ← b[t] + β · (G_t − b[t])，
    并返回更新后的 b（原地修改后 return b，或返回一个新数组，都可以）。"""
    name = "跨回合的滑动平均基线 b[t]"
    max_t = 12
    episodes = [np.array([4.0, 3.0, 2.0, 1.0]),
                np.arange(9, 0, -1, dtype=float),
                np.array([3.0, 2.0, 1.0]),
                np.arange(12, 0, -1, dtype=float),
                _ref_reward_to_go([1.0] * 7, 0.9)]
    for beta in (0.05, 0.3):
        b = np.zeros(max_t)
        ref = np.zeros(max_t)
        for k, G in enumerate(episodes, start=1):
            T = len(G)
            G_before = G.copy()
            ok, out = safe_call(name, fn, b, G.copy(), beta)
            if not ok:
                return False
            if out is None:
                return report(name, False, "函数没有返回值（返回了 None）",
                              "训练循环里写的是 b = update_time_baseline(b, G, beta)，所以最后要 return b")
            try:
                new = _np(out).reshape(-1)
            except (TypeError, ValueError):
                return report(name, False, f"应返回更新后的数组 b，实际是 {type(out).__name__}")
            if new.shape != (max_t,):
                return report(name, False, f"b 的长度应保持 {max_t}（MAX_T），实际 {new.size}",
                              "b 覆盖所有可能的时间步；这一回合只更新前 T 个：b[:T]")
            prev = ref.copy()
            ref[:T] += beta * (G_before - ref[:T])
            if not np.allclose(new, ref, atol=1e-9):
                if not np.allclose(new[T:], prev[T:]):
                    hint = "这一回合只走了 T 步，只能更新 b[:T]；更后面的时间步要保持原样"
                elif np.allclose(new[:T], G_before):
                    hint = "你把 b[:T] 直接换成了这一回合的 G；滑动平均要只走一小步：b[:T] += beta * (G − b[:T])"
                elif np.allclose(new[:T], prev[:T] + beta * G_before):
                    hint = "少减了旧值：b[:T] += beta * (G − b[:T])"
                elif np.allclose(new[:T], beta * prev[:T] + (1 - beta) * G_before):
                    hint = "新旧权重放反了：旧值占 1 − β，新的 G 占 β"
                elif np.allclose(new[:T], prev[:T] + beta * (G_before.mean() - prev[:T])) or \
                        np.allclose(new[:T], np.full(T, G_before.mean())):
                    hint = ("用了本回合 G 的平均——b 要按时间步 t 分别记录（b[t] 只和过去回合在第 t 步的 G_t 比）；"
                            "用回合内均值就退化成作业里要讨论的有偏基线")
                else:
                    hint = "b[:T] += beta * (G − b[:T])，其中 T = len(G)"
                return report(name, False,
                              f"β = {beta}，第 {k} 个回合（T = {T}）之后：期望 b = {np.round(ref, 3).tolist()}，"
                              f"实际 {np.round(new, 3).tolist()}", hint)
            b = new.copy()
    return report(name, True)


def _solved_at(returns, threshold=CARTPOLE_THRESHOLD, window=SOLVE_WINDOW):
    """首次"最近 window 回合平均回报 ≥ threshold"的回合编号（从 1 数）；没达到返回 None。"""
    r = np.asarray(returns, dtype=float)
    if len(r) < window:
        return None
    ma = np.convolve(r, np.ones(window) / window, mode="valid")
    hits = np.flatnonzero(ma >= threshold - 1e-9)
    return int(hits[0]) + window if hits.size else None


def _make_value_net(seed=0):
    """与作业 ValueNet 相同的结构（4 → 16 → 1、输出 ×100），固定初始化；不消耗全局随机数。"""
    import torch
    from torch import nn

    class _ValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 16)
            self.fc2 = nn.Linear(16, 1)

        def forward(self, x):
            return 100.0 * self.fc2(torch.relu(self.fc1(x))).squeeze(-1)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _ValueNet()


def _ref_fit_value(value, optimizer, states, G, n_iters):
    for _ in range(n_iters):
        loss = ((value(states) - G) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def fit_value(fn):
    """fn(value, optimizer, states, G, n_iters)：用均方误差把 value(states) 回归到 reward-to-go G，
    做 n_iters 步（每步 zero_grad → backward → step），返回最后一步的损失（loss.item()）。
    在一个固定初始化的价值网络上与参考实现对照：同一个 optimizer 连续调用两次、各 100 步
    （训练循环里每个回合都调用一次，Adam 的状态要跨回合保留）。"""
    import torch
    name = "价值网络回归 fit_value"
    gen = torch.Generator().manual_seed(0)
    states = torch.randn(300, 4, generator=gen)
    # 一个能从状态学出来的目标（量级与 CartPole 的回报相同）：写法不同的错误会让 200 步后的 MSE 明显偏离参考实现
    G = (250 + 60 * states[:, 0] - 40 * states[:, 2] + 30 * states[:, 1] * states[:, 3]).clamp(1, 500)
    states_before, G_before = states.clone(), G.clone()

    def mse(net):
        with torch.no_grad():
            return float(((net(states) - G) ** 2).mean())

    ref = _make_value_net(0)
    ref_opt = torch.optim.Adam(ref.parameters(), lr=1e-2)
    for _ in range(2):
        _ref_fit_value(ref, ref_opt, states, G, 100)
    want = mse(ref)

    net = _make_value_net(0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    before = mse(net)
    params_before = [p.detach().clone() for p in net.parameters()]
    loss = None
    for _ in range(2):
        ok, loss = safe_call(name, fn, net, opt, states, G, 100)
        if not ok:
            return False
    if not (torch.equal(states, states_before) and torch.equal(G, G_before)):
        return report(name, False, "函数改动了传进来的 states 或 G", "回归只读数据")
    after = mse(net)
    if not np.isfinite(after):
        return report(name, False, "训练后网络输出了非有限数", "学习率用传进来的 optimizer 里的（1e-2），不要自己放大")
    if after > 0.5 * before:
        if all(torch.equal(p, q) for p, q in zip(net.parameters(), params_before)):
            hint = "参数一点都没变：每一步要 loss.backward() 之后 optimizer.step()"
        else:
            hint = "下降得太少：要循环 n_iters 次，每次 zero_grad → backward → step"
        return report(name, False, f"两次调用、各 100 步之后，MSE 应明显下降（之前 {before:.0f}，参考实现之后 {want:.0f}），"
                      f"实际 {after:.0f}", hint)
    if abs(after - want) > 0.1 * want:
        return report(name, False, f"MSE 下降了，但与参考实现差得太多（参考 {want:.1f}，实际 {after:.1f}）",
                      "常见原因：①忘了 optimizer.zero_grad()，梯度一步步累加；②在函数里新建了优化器——"
                      "要用传进来的 optimizer，否则 Adam 的动量每个回合都被清零；③没有循环 n_iters 次；"
                      "④损失不是均方误差 ((value(states) − G)²).mean()")
    if loss is None:
        return report(name, False, "函数没有返回值（返回了 None）", "最后 return loss.item()（最后一步的损失，一个数）")
    try:
        lf = float(loss.item() if hasattr(loss, "item") else loss)
    except (TypeError, ValueError, RuntimeError):
        return report(name, False, f"应返回最后一步的损失（一个数），实际是 {type(loss).__name__}", "return loss.item()")
    return report(name, bool(np.isfinite(lf)), f"返回的损失不是有限数（{lf}）", "return loss.item()")


def baseline_explained(explained):
    """研究实验 5 的诊断量：explained = {"B": [5 个种子], "D": [5 个种子], …}，每个数是"基线解释掉的比例"
    1 − Σ(G_t − b_t)² / ΣG_t²——从第 51 回合起，把所有回合、所有时间步加在一起算；b_t 是**更新前**的基线
    （B：滑动平均 b[t]；D：价值网络 V(s_t)）。只检查 B 与 D，其他键可以有。"""
    name = "基线解释掉的比例（B vs D）"
    if not isinstance(explained, dict) or not {"B", "D"} <= set(map(str, explained)):
        return report(name, False, "explained 应是 dict，至少含 'B' 与 'D' 两个键，每个键是 5 个种子的数",
                      "train_reinforce 返回 (returns, explained)；主循环里 explained[setting] = [每个种子的比例]")
    vals = {}
    for k in ("B", "D"):
        v, ok = _floats(name, explained[k], N_SEEDS_RL)
        if v is None:
            return ok
        vals[k] = np.array(v)
    b, d = vals["B"], vals["D"]
    if not np.all((b >= 0.95) & (b <= 1.0)):
        return report(name, False, f"B 每个种子都应 ≥ 0.95（本机 0.967–0.978），实际 {np.round(b, 3).tolist()}",
                      "从第 51 回合起（episode >= 50）累计 Σ(G − b[:T])² 与 ΣG²，b 用**更新前**的那个；"
                      "比例 = 1 − 两者之比")
    if np.any(d > 0.95):
        return report(name, False, f"D 每个种子应在 [0.6, 0.95]（本机 0.79–0.91），实际 {np.round(d, 3).tolist()}",
                      "D 的价值网络只看状态 (x, ẋ, θ, θ̇)——如果把时间 t 也放进了输入，那是选做 Dt，不是 D"
                      "（本机 Dt 约 0.97）；另外检查基线用的是**更新前**的 V（先在 torch.no_grad() 下算，再 fit_value）")
    if np.any(d < 0.6):
        return report(name, False, f"D 每个种子应在 [0.6, 0.95]（本机 0.79–0.91），实际 {np.round(d, 3).tolist()}",
                      "检查 ValueNet 的输出 ×100、每回合 fit_value 做 V_ITERS = 5 步、V_LR = 1e-2，"
                      "以及 baseline 是 value(states) 而不是别的量")
    return report(name, bool(d.mean() < b.mean()), f"D 的均值（{d.mean():.3f}）应低于 B（{b.mean():.3f}）")


def reinforce_results(results):
    """研究实验 5 的结果：results = {"A": [...], "B": [...], "C": [...], "D": [...]}，每个设置是 5 个种子的列表，
    每个元素是该种子每个回合的回报（回合内奖励之和，CartPole-v1 中在 1–500 之间）。
    只查结构；数值上只要求 B（滑动平均基线）与 D（价值网络基线）5 个种子的达标回合数中位数都不超过回合预算。"""
    name = "研究实验 5 结果结构（A/B/C/D × 5 种子）"
    if not isinstance(results, dict):
        return report(name, False, f"results 应是 dict，实际是 {type(results).__name__}")
    keys = set(map(str, results))
    if keys != set(REINFORCE_SETTINGS):
        return report(name, False, f"键应恰好是 {list(REINFORCE_SETTINGS)}，实际 {sorted(keys)}")
    budget = None
    curves = {}
    for setting in REINFORCE_SETTINGS:
        runs = results[setting]
        try:
            runs = [_np(r).reshape(-1) for r in runs]
        except (TypeError, ValueError):
            return report(name, False, f"results['{setting}'] 应是 5 个种子的回报序列（列表或数组）")
        if len(runs) != N_SEEDS_RL:
            return report(name, False, f"results['{setting}'] 应有 {N_SEEDS_RL} 个种子，实际 {len(runs)} 个",
                          "RL 种子间差异很大，本实验每个设置跑种子 0–4")
        for i, r in enumerate(runs):
            if budget is None:
                budget = len(r)
            if len(r) != budget:
                return report(name, False, f"results['{setting}'][{i}] 有 {len(r)} 个回合，其他有 {budget} 个",
                              "四种设置、所有种子要用同一个回合预算，比较才公平")
            if budget < SOLVE_WINDOW:
                return report(name, False, f"每个种子至少要跑 {SOLVE_WINDOW} 个回合才能算'最近 100 回合平均'，实际 {budget}")
            if not np.all(np.isfinite(r)):
                return report(name, False, f"results['{setting}'][{i}] 里有非有限数")
            if r.min() < 1 or r.max() > CARTPOLE_MAX_RETURN + 1e-6:
                return report(name, False,
                              f"results['{setting}'][{i}] 的回报应在 [1, 500] 之间，实际范围 [{r.min():.1f}, {r.max():.1f}]",
                              "记录的应是每回合的奖励之和（= 这一局坚持了多少步），不是 reward-to-go 之和或损失")
        if all(np.array_equal(runs[0], r) for r in runs[1:]):
            return report(name, False, f"results['{setting}'] 的 5 个种子曲线完全相同，种子没有起作用",
                          "每个种子在建网络之前 experiments.set_seed(seed)，并 env.reset(seed=seed)")
        curves[setting] = runs
    checks = [("B", "B（滑动平均基线）",
               "本机实测 B 在 500 回合内 10/10 个种子达标（中位约 230 回合）。检查：b 只依赖过去的回合、"
               "按时间步 t 记录（不要用本回合 G 的平均）；β = 0.05；γ = 1.0；Adam lr 1e-2；每回合更新一次"),
              ("D", "D（价值网络基线）",
               "本机实测 D 在 500 回合内 10/10 个种子达标（中位 258、最慢 403）。检查：先在 torch.no_grad() 下用**旧的** V "
               "算基线、权重与 B 相同（G − V(s)），再 fit_value；ValueNet 输出 ×100；V_LR 1e-2、V_ITERS 5；"
               "价值网络在策略网络之后创建")]
    for setting, label, hint in checks:
        solved = [_solved_at(r) for r in curves[setting]]
        med = float(np.median([s if s is not None else np.inf for s in solved]))
        if med > budget:
            shown = [s if s is not None else f">{budget}" for s in solved]
            return report(name, False, f"{label}的达标回合数中位数应不超过预算 {budget}，实际各种子 {shown}", hint)
    return report(name, True)


# ---------- S4 · 奖励设计 ----------
def potential_shaping(fn, potential, gamma):
    """fn(obs, obs_next, terminated)：讲义 S4 的 reward_potential = 稀疏奖励 + 势函数塑形项，

        r = (1 if terminated else 0) + γ·Φ(s') − Φ(s)，**终止时 Φ(s') 记为 0**

    （Ng, Harada & Russell 1999 的形式）。potential 是讲义给出的 Φ(obs)，gamma 是 Q-learning 用的折扣
    （讲义的 GAMMA_MC）。不训练，直接在几组构造的 (obs, obs_next, terminated) 上比对数值——
    训练结果层面的 reward_hacking_demo 抓不到"漏了 γ""终止时没把 Φ 置 0"这类写法错误
    （漏了 γ 时到达率反而更高）。"""
    name = "势函数塑形 reward_potential"
    try:
        g = float(gamma)
    except (TypeError, ValueError):
        return report(name, False, f"gamma 应是一个数，实际是 {type(gamma).__name__}", "传入讲义里的 GAMMA_MC")
    if not 0.0 < g < 1.0:
        return report(name, False, f"gamma 应在 (0, 1) 之间，实际 {g}", "传入 Q-learning 用的折扣 GAMMA_MC")
    # (x, v) → (x', v')：谷底、左坡、右坡、高速冲坡；最后两组越过 x = 0.5（到达旗子，terminated = True）
    samples = [((-0.52, 0.0), (-0.521, -0.001), False),
               ((-1.0, -0.03), (-1.03, -0.02), False),
               ((0.2, 0.045), (0.245, 0.05), False),
               ((-0.3, 0.06), (-0.24, 0.065), False),
               ((0.47, 0.04), (0.51, 0.04), True),
               ((0.49, 0.02), (0.512, 0.022), True)]
    rows = []                                       # (描述, terminated, Φ(s), Φ(s'), 学习者的值)
    for (x, v), (x2, v2), term in samples:
        obs = np.array([x, v], dtype=np.float32)    # 与 MountainCar 的观测同类型
        obs_next = np.array([x2, v2], dtype=np.float32)
        try:
            phi_s, phi_next = float(potential(obs.copy())), float(potential(obs_next.copy()))
        except Exception as e:  # noqa: BLE001
            return report(name, False, f"potential(obs) 运行出错 {type(e).__name__}: {e}",
                          "第二个参数传讲义里的 potential 函数本身（不加括号）")
        if not (np.isfinite(phi_s) and np.isfinite(phi_next)):
            return report(name, False, "potential(obs) 返回了非有限数")
        ok, got = safe_call(name, fn, obs.copy(), obs_next.copy(), term)
        if not ok:
            return False
        if got is None:
            return report(name, False, "函数没有返回值（返回了 None）", "最后 return 这一步的奖励（一个数）")
        try:
            got = float(_np(got))
        except (TypeError, ValueError):
            return report(name, False, f"应返回一个数（这一步的奖励），实际是 {type(got).__name__}")
        desc = f"s = (x {x}, v {v}) → s' = (x {x2}, v {v2})，terminated = {term}"
        rows.append((desc, term, phi_s, phi_next, got))
    if np.ptp([r[2] for r in rows] + [r[3] for r in rows]) < 1e-3:
        return report(name, False, "potential 在这些状态上几乎是常数，无法检查塑形项",
                      "第二个参数传讲义里的 potential（机械能 1000·(½v² + 0.0025·sin(3x)/3)）")

    def matches(f):
        return all(abs(got - f(term, ps, pn)) < 1e-5 for _, term, ps, pn, got in rows)

    def correct(term, ps, pn):
        return (1.0 if term else 0.0) + g * (0.0 if term else pn) - ps

    if matches(correct):
        return report(name, True)
    sparse = lambda term: 1.0 if term else 0.0                      # noqa: E731
    mistakes = [
        (lambda term, ps, pn: sparse(term) + (0.0 if term else pn) - ps,
         "漏了折扣 γ：塑形项是 γ·Φ(s') − Φ(s)。少了 γ 就多出一项 (1 − γ)·Φ(s')——它在奖励'处在高能量状态'本身，"
         "不再是势能差，沿轨迹求和也不会相消，'最优策略不变'的保证就没有了"),
        (lambda term, ps, pn: sparse(term) + g * pn - ps,
         "终止状态的势能要记为 0：phi_next = 0.0 if terminated else potential(obs_next)。"
         "否则到达旗子的那一步白白多拿 γ·Φ(s')，塑形项沿轨迹求和就不再只取决于起点"),
        (lambda term, ps, pn: sparse(term) + pn - ps,
         "两处错误：漏了折扣 γ（应是 γ·Φ(s')），而且终止时没有把 Φ(s') 记为 0"),
        (lambda term, ps, pn: sparse(term) + ps - g * (0.0 if term else pn),
         "符号反了：F = γ·Φ(s') − Φ(s)（新状态的势能减去旧状态的），攒到能量才得正奖励"),
        (lambda term, ps, pn: sparse(term) + (0.0 if term else pn) - g * ps,
         "γ 乘错了位置：它乘在新状态的势能上，F = γ·Φ(s') − Φ(s)"),
        (lambda term, ps, pn: g * (0.0 if term else pn) - ps,
         "只返回了塑形项 F，忘了加上稀疏奖励（到达旗子 +1）：reward_sparse(obs, obs_next, terminated) + F"),
    ]
    hint = ("r = reward_sparse(obs, obs_next, terminated) + γ·Φ(s') − Φ(s)，"
            "其中 Φ(s') = 0.0 if terminated else potential(obs_next)，Φ(s) = potential(obs)")
    for f, h in mistakes:
        if matches(f):
            hint = h
            break
    desc, term, ps, pn, got = next(r for r in rows if abs(r[4] - correct(*r[1:4])) >= 1e-5)
    return report(name, False, f"{desc}：期望 {correct(term, ps, pn):.5f}，实际 {got:.5f}", hint)


def reward_hacking_demo(success_sparse, success_shaped, success_velocity):
    """MountainCar 三种奖励下"到达终点的比例"（训练回合中到达终点的比例，或训练后贪心策略的到达率；三者口径一致即可）：
    只在终点 +1（稀疏）、终点 +1 + 势函数塑形、终点 +1 + 每步 10·|v| 速度奖励。"""
    name = "MountainCar：稀疏奖励 / 势函数塑形 / 速度奖励"
    vals, ok = _floats(name, (success_sparse, success_shaped, success_velocity), 3)
    if vals is None:
        return ok
    sp, sh, ve = vals
    if not all(0.0 <= v <= 1.0 for v in vals):
        return report(name, False, f"三个数都应是比例（0–1），实际 {vals}")
    if sp > 0.01:
        return report(name, False, f"只在终点 +1 时几乎不可能到达终点（应 ≤ 0.01），实际 {sp:.3f}",
                      "稀疏版本只在 terminated 为 True 时给 1、其余每步给 0；不要沿用环境默认的每步 −1"
                      "（Q 零初始化配上 −1 等于乐观初始化，反而能学会）")
    if sh < 0.05 or sh < sp + 0.05:
        return report(name, False, f"势函数塑形后的到达率应明显高于稀疏奖励（≥ 0.05），实际 {sh:.3f}（稀疏 {sp:.3f}）",
                      "塑形项 F = γ·Φ(s') − Φ(s)，Φ 用机械能 1000·(½v² + 0.0025·sin(3x)/3)，终止状态 Φ = 0"
                      "（先用 ck.potential_shaping 单独检查塑形函数）；本机 3 个种子 × 500 回合，到达率约 0.23–0.56")
    if ve > 0.02:
        return report(name, False, f"速度奖励下智能体应几乎从不到达终点（≤ 0.02），实际 {ve:.3f}",
                      "速度奖励是 reward_sparse + 每步 10·|v'|（v' 是下一状态的速度），终点仍是 +1，"
                      "不要再叠加环境默认的每步 −1（那等于乐观初始化，会驱动它去探索、碰到旗子）。"
                      "γ = 0.99 时一直摆下去的折扣回报有十几分，远大于旗子的 +1，所以它应几乎从不到旗子")
    return report(name, True)
