"""第18周自查函数。用法：from zero2researcher.checks import week18 as ck; ck.incremental_mean(incremental_mean)

函数类练习把函数本身传进来（不加括号），例如 ck.reward_to_go(reward_to_go)；其余检查直接传入你算出的数。
含随机性的结果按合理区间检查，区间依据本机实测 ± 余量：
- 10 臂测试平台（Sutton & Barto 第 2 版第 2 章的设定：2000 个任务 × 1000 步，q*(a) ~ N(0, 1)，奖励 ~ N(q*(a), 1)）；
- FrozenLake-v1 4×4 表格 Q-learning（HF Deep RL Course Unit 2 的超参数）；
- CartPole-v1 REINFORCE（策略网络 4→16→2、Adam 1e-2、γ = 1.0；"达标" = 最近 100 回合平均回报首次 ≥ 475，
  475 是 Gymnasium 注册表里 CartPole-v1 的 reward_threshold）；
- MountainCar-v0 离散化表格 Q-learning 的三种奖励设计（势函数塑形函数本身另用构造的状态直接比对数值）。
本模块只用 NumPy；学习者的函数返回 torch 张量时也能检查（用 .detach() 转换，无需 import torch）。
"""
import numpy as np

from . import report, safe_call

CARTPOLE_THRESHOLD = 475.0     # Gymnasium 1.4 注册表中 CartPole-v1 的 reward_threshold
CARTPOLE_MAX_RETURN = 500.0    # CartPole-v1 的 max_episode_steps
SOLVE_WINDOW = 100
REINFORCE_SETTINGS = ("A", "B", "C")
N_SEEDS_RL = 5


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
                     "γ = 0.95 的最优策略在 99 步内的成功率约 0.73，所以超过 0.77 说明评估的环境或步数上限不对"
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
                              "训练循环里写的是 b = update_baseline(b, G, beta)，所以最后要 return b")
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


def reinforce_results(results):
    """研究实验 5 的结果：results = {"A": [...], "B": [...], "C": [...]}，每个设置是 5 个种子的列表，
    每个元素是该种子每个回合的回报（回合内奖励之和，CartPole-v1 中在 1–500 之间）。
    只查结构；数值上只要求 B（滑动平均基线）5 个种子的达标回合数中位数不超过回合预算。"""
    name = "研究实验 5 结果结构（A/B/C × 5 种子）"
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
                              "三种设置、所有种子要用同一个回合预算，比较才公平")
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
    solved = [_solved_at(r) for r in curves["B"]]
    med = float(np.median([s if s is not None else np.inf for s in solved]))
    shown = [s if s is not None else f">{budget}" for s in solved]
    return report(name, med <= budget,
                  f"B（滑动平均基线）的达标回合数中位数应不超过预算 {budget}，实际各种子 {shown}",
                  "本机实测 B 在 500 回合内 10/10 个种子达标（中位约 230 回合）。检查：b 只依赖过去的回合、"
                  "按时间步 t 记录（不要用本回合 G 的平均）；β = 0.05；γ = 1.0；Adam lr 1e-2；每回合更新一次")


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
