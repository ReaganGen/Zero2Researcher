"""第19周自查函数。用法：from zero2researcher.checks import week19 as ck; ck.tool_schema(calculator_tool)

函数练习把函数本身传进来（不加括号），例如 ck.react_loop(run_agent)。
需要跑模型的检查一律在内部新建一个 agents.MockLLM（不联网、不花钱、结果确定），
所以即使你在 Colab 上配了 Claude 的密钥，检查也不会产生费用。

结果字典的约定（课堂与作业通用）：run_agent / run_workflow 返回
    {"answer": 正常结束时最后一条回复的文字（否则 None）, "llm_calls": int, "tool_calls": int,
     "input_tokens": int, "output_tokens": int, "trace": [每次工具调用的记录],
     "stop_reason": 最后一次调用的 stop_reason,
     "status": "answered" | "refused" | "truncated" | "max_steps", ...}
run_agent 另外返回 "messages"（完整的对话历史）。
"""
import inspect
import json
import re
from collections import Counter

from . import report
from .. import agents

RESULT_KEYS = ("answer", "llm_calls", "tool_calls", "input_tokens", "output_tokens", "trace", "status")

STATUSES = ("answered", "refused", "truncated", "max_steps")
"""一次运行怎样结束：正常作答（end_turn）、拒答（refusal）、被截断（max_tokens）、用完 max_steps。"""

CATEGORIES = ("工具选择错误", "参数错误", "提前结束", "幻觉/编造", "格式不匹配")
"""失败分析的五个类别（每条失败只标一个主因）。"""

STOP_CATEGORIES = ("拒答", "截断", "超步数")
"""非正常结束的三种情况：先按 status 单独计数，不进入上面的五类。"""

_LABELS = {**agents.FAULT_CATEGORY, **agents.STOP_CATEGORY}

_DECISION_RULES = (
    "先看这次运行怎样结束：拒答（status 'refused'）、截断（'truncated'）、超步数（'max_steps'）单独计数。"
    "其余的按顺序判断，第一条成立的就是主因：①答案内容其实对了、只是判分器没认出来 → 格式不匹配；"
    "②调用了不该用的工具（算术题去查维基、事实题去用计算器；workflow 里路由走错分支）→ 工具选择错误；"
    "③工具选对了，但传进去的参数错了（表达式列错、检索词查不到所需条目）→ 参数错误；"
    "④还没拿到所需信息（或没做完所需的计算）就给出答案 → 提前结束；"
    "⑤观察结果里已经有正确信息，答案却与之矛盾或凭空捏造 → 幻觉/编造")

_ALIASES = {
    "wrong_tool": ["工具选择错误", "选错工具", "工具选择", "工具错误", "路由错误", "wrongtool", "toolselection",
                   "toolchoice", "wrongtoolselection"],
    "bad_args": ["参数错误", "参数", "参数有误", "输入错误", "badargs", "badarguments", "wrongarguments",
                 "argumenterror", "wrongargs"],
    "early_stop": ["提前结束", "过早结束", "提前停止", "过早停止", "earlystop", "prematurestop", "premature",
                   "earlystopping"],
    "hallucination": ["幻觉编造", "幻觉", "编造", "hallucination", "fabrication", "hallucinated"],
    "format_mismatch": ["格式不匹配", "格式错误", "格式", "标签歧义", "formatmismatch", "format",
                        "labelambiguity", "answerformat"],
    "refusal": ["拒答", "拒绝回答", "拒绝", "refusal", "refused", "refuse"],
    "truncated": ["截断", "被截断", "输出截断", "truncated", "truncation", "maxtokens"],
    "loop": ["超步数", "超出步数", "步数耗尽", "用完步数", "死循环", "循环", "重复调用", "maxsteps", "loop",
             "looping", "推理错误"],
}

_EXERCISE_CASES = {"A": ("f10", "early_stop"), "B": ("a09", "bad_args"), "C": ("f04", "wrong_tool"),
                   "D": ("f05", "hallucination"), "E": ("a02", "format_mismatch")}

_ALLOWED_TOOL_KEYS = {"name", "description", "input_schema", "strict", "cache_control", "type",
                      "input_examples", "eager_input_streaming", "defer_loading", "allowed_callers"}


def _items():
    return {it["id"]: it for it in agents.load_eval()}


def _canon_label(label):
    s = re.sub(r"[\s_\-/、，,（）()：:]+", "", str(label).strip().lower())
    for fault, names in _ALIASES.items():
        if s in names:
            return fault
    return None


# ---------- 工具定义 ----------
def tool_schema(d):
    """检查一个工具定义 dict：name、description、input_schema（type object、properties 里每个参数有
    type 和 description、required、additionalProperties: False）。对工具包里已有的工具名，
    还会核对参数名是否与 Python 函数一致。"""
    name = "工具定义（JSON schema）"
    if not isinstance(d, dict):
        return report(name, False, f"工具定义应是 dict，实际是 {type(d).__name__}")
    extra = set(d) - _ALLOWED_TOOL_KEYS
    if extra:
        return report(name, False, f"出现了 API 不认识的顶层字段 {sorted(extra)}",
                      "顶层只需要 name、description、input_schema 三个字段（参数写在 input_schema 里）")
    try:
        json.dumps(d)
    except (TypeError, ValueError) as e:
        return report(name, False, f"不能序列化成 JSON：{e}", "只用 dict、list、str、数字、True/False")
    tname = d.get("name")
    if not isinstance(tname, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", tname):
        return report(name, False, f"name 应是 1–128 个字母、数字、下划线或连字符，实际 {tname!r}")
    desc = d.get("description")
    if not isinstance(desc, str) or len(desc.strip()) < 20:
        return report(name, False, "description 太短或缺失",
                      "描述是模型决定'何时、怎样'用这个工具的唯一依据：写清它做什么、接受什么、返回什么、"
                      "什么时候该用（HF Agents Course 列的工具组成：文字描述、可调用对象、带类型的参数、"
                      "可选的带类型输出）")
    schema = d.get("input_schema")
    if not isinstance(schema, dict):
        return report(name, False, "缺少 input_schema（一个 JSON Schema 对象）",
                      "注意字段名是 input_schema，不是 parameters（那是别家 API 的写法）")
    if schema.get("type") != "object":
        return report(name, False, f"input_schema 的 type 应为 \"object\"，实际 {schema.get('type')!r}")
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return report(name, False, "input_schema 里缺少 properties（每个参数一项）")
    for pname, spec in props.items():
        if not isinstance(spec, dict) or "type" not in spec:
            return report(name, False, f"参数 {pname!r} 缺少 type", "例如 {\"type\": \"string\", \"description\": ...}")
        if not str(spec.get("description", "")).strip():
            return report(name, False, f"参数 {pname!r} 缺少 description",
                          "给参数写一句说明和一个例子，能显著减少模型传错参数")
    req = schema.get("required")
    if not isinstance(req, list) or not req or not set(req) <= set(props):
        return report(name, False, f"required 应是 properties 里参数名的非空列表，实际 {req!r}")
    if schema.get("additionalProperties") is not False:
        return report(name, False, "input_schema 里应写 \"additionalProperties\": False",
                      "禁止模型编造 schema 之外的参数；这也是 strict 模式的前提（Python 里写 False，不是 false）")
    fn = agents.TOOL_FUNCTIONS.get(tname)
    if fn is not None:
        params = inspect.signature(fn).parameters
        needed = [p for p, v in params.items() if v.default is inspect.Parameter.empty]
        if set(req) != set(needed) or not set(props) <= set(params):
            return report(name, False,
                          f"工具包里的 {tname}() 需要参数 {needed}，你的 schema 声明的是 {sorted(props)}"
                          f"（required {req}）",
                          f"模型给出的 input 会被直接展开成 {tname}(**block.input)，参数名必须一模一样")
    return report(name, True)


# ---------- 执行一个 tool_use 块 ----------
def tool_result_block(fn):
    """fn(block) -> tool_result dict。检查：type、tool_use_id 与 block.id 一致、content 是工具输出的字符串；
    工具出错（不安全的表达式、未知工具名）时不抛异常，而是返回 is_error=True 的结果。"""
    name = "执行 tool_use 并构造 tool_result"
    cases = [
        (agents.ToolUseBlock(type="tool_use", id="toolu_ck_001", name="calculator",
                             input={"expression": "2 * (3 + 4)"}), "14", False),
        (agents.ToolUseBlock(type="tool_use", id="toolu_ck_002", name="wiki_search",
                             input={"query": "Marie Curie Nobel Prize"}), "1911", False),
        (agents.ToolUseBlock(type="tool_use", id="toolu_ck_003", name="calculator",
                             input={"expression": "__import__('os').system('echo hacked')"}), None, True),
        (agents.ToolUseBlock(type="tool_use", id="toolu_ck_004", name="weather",
                             input={"city": "Paris"}), None, True),
    ]
    for block, want_in, want_error in cases:
        try:
            out = fn(block)
        except NotImplementedError:
            return report(name, False, "函数还没有实现（仍是 NotImplementedError）")
        except Exception as e:  # noqa: BLE001
            return report(name, False,
                          f"处理 {block.name}({block.input}) 时把异常 {type(e).__name__}: {e} 抛了出来",
                          "工具出错是正常情况：用 try/except 接住异常，返回一个标了 is_error 的 tool_result"
                          "（错误信息写进 content），让模型看到错误后自己调整")
        if not isinstance(out, dict):
            return report(name, False, f"应返回一个 dict，实际是 {type(out).__name__}")
        if out.get("type") != "tool_result":
            return report(name, False, f"type 应为 'tool_result'，实际 {out.get('type')!r}")
        if out.get("tool_use_id") != block.id:
            return report(name, False, f"tool_use_id 应等于 block.id（{block.id}），实际 {out.get('tool_use_id')!r}",
                          "API 靠这个 id 把结果和对应的那次调用配对；并行调用时尤其重要")
        content = out.get("content")
        if isinstance(content, list):
            content = "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
        if not isinstance(content, str) or not content:
            return report(name, False, f"content 应是非空字符串（工具输出），实际 {out.get('content')!r}",
                          "content=str(结果)")
        if bool(out.get("is_error", False)) != want_error:
            if want_error:
                return report(name, False, f"{block.name}({block.input}) 应该失败，但结果里没有 is_error=True",
                              "出错的工具调用要标 is_error=True（并把错误信息写进 content）")
            return report(name, False, f"{block.name}({block.input}) 正常执行，却被标成了 is_error")
        if want_in is not None and want_in not in content:
            return report(name, False, f"{block.name}({block.input}) 的 content 里应包含 {want_in!r}，实际 {content!r}",
                          "用 agents.TOOL_FUNCTIONS[block.name](**block.input) 调用工具")
    return report(name, True)


# ---------- ReAct 循环 ----------
def _check_result(name, res, llm, item, want_calls, want_tools):
    if not isinstance(res, dict):
        return report(name, False, f"应返回一个 dict，实际是 {type(res).__name__}",
                      f"返回 {{{', '.join(repr(k) for k in RESULT_KEYS)}}} 这些键")
    missing = [k for k in RESULT_KEYS if k not in res]
    if missing:
        return report(name, False, f"返回的 dict 缺少键 {missing}",
                      "status 记录这次运行怎样结束：'answered'、'refused'、'truncated' 或 'max_steps'"
                      if "status" in missing else None)
    if not agents.grade(item, res["answer"]):
        hint = "最后一条回复里应有 'Final answer: ...'；把 response.content 里所有 text 块的文字拼起来作为答案"
        miss = llm.log[-1].get("missing_evidence") if llm.log else None
        if miss:
            hint = (f"mock 在工具结果里没看到应有的 {miss}，所以只能瞎猜——检查工具是否真的执行了、"
                    "结果是否放进了 tool_result 的 content")
        return report(name, False, f"题 {item['id']} 的答案不对：{res['answer']!r}（标准答案 {item['gold']}）", hint)
    if res["status"] != "answered":
        return report(name, False, f"题 {item['id']}：正常作答时 status 应为 'answered'，实际 {res['status']!r}",
                      "stop_reason == 'end_turn' 时记 status = 'answered'")
    if not (res["llm_calls"] == llm.n_calls == want_calls):
        return report(name, False,
                      f"题 {item['id']}：标准轨迹应调用模型 {want_calls} 次；mock 实际被调用 {llm.n_calls} 次，"
                      f"你记录的 llm_calls = {res['llm_calls']}",
                      "每次 create() 记一次；stop_reason 不是 'tool_use' 时就该结束循环")
    if res["tool_calls"] != want_tools or len(res["trace"]) != want_tools:
        return report(name, False,
                      f"题 {item['id']}：应执行 {want_tools} 次工具，你记录 tool_calls = {res['tool_calls']}、"
                      f"trace 里 {len(res['trace'])} 条",
                      "一个回复里可能有多个 tool_use 块（并行调用），每个都要执行、都要记一次")
    if res["input_tokens"] != llm.total_input_tokens or res["output_tokens"] != llm.total_output_tokens:
        return report(name, False,
                      f"题 {item['id']}：token 合计应为 输入 {llm.total_input_tokens} / 输出 {llm.total_output_tokens}，"
                      f"你记录的是 {res['input_tokens']} / {res['output_tokens']}",
                      "每次调用后累加 response.usage.input_tokens 与 response.usage.output_tokens")
    return True


def _history_problem(msgs):
    """对话历史是否按推荐的写法组织。真实 API 对这两种写法都不报错（MockLLM 也接受），但各有代价。"""
    prev = None
    for m in msgs:
        role = m.get("role") if isinstance(m, dict) else None
        if role == prev == "user":
            return ("同一轮的 tool_result 被拆成了几条相邻的 user 消息",
                    "真实 API 会把相邻的 user 消息合并成一轮、不报错，但这样会让模型以后更少并行调用；"
                    "把这一轮的全部 tool_result 放进同一条 user 消息")
        if role == "assistant":
            types = [agents._btype(b) for b in agents._blocks(m.get("content"))]
            if "tool_use" in types and "thinking" not in types:
                return ("assistant 消息没有原样追加：thinking 块被删掉了",
                        "真实 API 不会因此报错，但模型会丢掉那一轮的推理；直接追加 response.content")
        prev = role
    return None


def _run(name, fn, *args, **kwargs):
    try:
        return True, fn(*args, **kwargs)
    except NotImplementedError:
        report(name, False, "函数还没有实现（仍是 NotImplementedError）")
    except agents.MockBadRequestError as e:
        report(name, False, f"请求被 API 拒绝了——{e}")
    except agents.MockLoopGuardError as e:
        report(name, False, str(e), "for step in range(max_steps): ... 循环结束仍没有答案就返回 answer=None")
    except Exception as e:  # noqa: BLE001
        report(name, False, f"运行时出错 {type(e).__name__}: {e}",
               "先读报错的最后一行；常见原因：把 block 当 dict 用（应写 block.type、block.id）、"
               "忘了 block.input 已经是 dict")
    return False, None


def react_loop(fn):
    """fn(question, llm, max_steps=8) -> 结果 dict（见模块说明）。用 MockLLM 跑 3 道题
    （顺序调用、并行调用、文字答案），检查对话历史的写法，再检查三种非正常结束：
    用完 max_steps、拒答（refusal）、被截断（max_tokens）。"""
    name = "ReAct 循环"
    items = _items()
    raw = {it["id"]: it for it in agents._eval_raw()["items"]}
    for qid in ("a03", "f06", "f03"):
        item, llm = items[qid], agents.MockLLM()
        ok, res = _run(name, fn, item["question"], llm, max_steps=8)
        if not ok:
            return False
        want_calls = len(raw[qid]["mock"]["agent"])
        if _check_result(name, res, llm, item, want_calls, item["ref_tool_calls"]) is not True:
            return False
        msgs = res.get("messages")
        if not isinstance(msgs, list) or not msgs or msgs[0].get("role") != "user":
            return report(name, False, "结果里的 messages 应是完整的对话历史（第一条是 user 的问题）")
        problem = _history_problem(msgs)
        if problem:
            return report(name, False, f"题 {qid}：{problem[0]}", problem[1])
    llm = agents.MockLLM(faults="loop")
    ok, res = _run(name, fn, items["a01"]["question"], llm, max_steps=3)
    if not ok:
        return False
    if llm.n_calls != 3:
        return report(name, False, f"模型一直要求调用工具时，max_steps=3 应恰好调用 3 次，实际 {llm.n_calls} 次",
                      "最大步数保护：for step in range(max_steps)")
    if not isinstance(res, dict) or res.get("answer") or agents.grade(items["a01"], res.get("answer")):
        return report(name, False, "达到最大步数时不应该返回答案——这时根本没有答案，answer 应为 None")
    if res.get("status") != "max_steps":
        return report(name, False, f"用完 max_steps 时 status 应为 'max_steps'，实际 {res.get('status')!r}",
                      "循环正常走完（没有 return）之后记 status = 'max_steps'")
    for fault, qid, status, what in (("refusal", "f01", "refused", "模型拒答（stop_reason == 'refusal'）"),
                                     ("truncated", "a05", "truncated", "回复被截断（stop_reason == 'max_tokens'）")):
        llm = agents.MockLLM(faults=fault)
        ok, res = _run(name, fn, items[qid]["question"], llm, max_steps=8)
        if not ok:
            return False
        if llm.n_calls != 1 or not isinstance(res, dict) or res.get("answer"):
            return report(name, False, f"{what}时应立即结束，answer 为 None",
                          "content 里可能只剩中断前的半句话，它不是答案：只在 stop_reason == 'end_turn' 时取文字作答案")
        if res.get("status") != status:
            return report(name, False, f"{what}时 status 应为 {status!r}，实际 {res.get('status')!r}",
                          "按 stop_reason 记录这次运行怎样结束：end_turn → 'answered'，refusal → 'refused'，"
                          "max_tokens → 'truncated'")
    return report(name, True)


# ---------- 固定步骤的 workflow ----------
def workflow(fn):
    """fn(question, llm) -> 结果 dict。检查 3 道题的答案、调用次数与固定的步骤顺序：
    算术题 route → expression → answer；查资料题 route → query → expression → answer。"""
    name = "固定步骤的 workflow"
    items = _items()
    for qid in ("a05", "f02", "f07"):
        item, llm = items[qid], agents.MockLLM()
        ok, res = _run(name, fn, item["question"], llm)
        if not ok:
            return False
        steps = [e.get("step") for e in llm.log]
        want = (["route", "query", "expression", "answer"] if item["type"] == "lookup"
                else ["route", "expression", "answer"])
        if steps != want:
            return report(name, False, f"题 {qid}（{item['type']}）的步骤顺序应为 {want}，实际 {steps}",
                          "每一步用 agents.WORKFLOW_PROMPTS[步骤名] 作 system，不传 tools；"
                          "路由结果含 LOOKUP 才做检索")
        want_tools = {"a05": 1, "f02": 1, "f07": 3}[qid]
        if _check_result(name, res, llm, item, len(want), want_tools) is not True:
            return False
    llm = agents.MockLLM(faults="refusal")
    ok, res = _run(name, fn, items["f02"]["question"], llm)
    if not ok:
        return False
    if not isinstance(res, dict) or res.get("answer") or res.get("status") != "refused":
        return report(name, False, "某一步被拒答（stop_reason == 'refusal'）时，answer 应为 None、status 应为 'refused'"
                                   f"；实际 answer = {res.get('answer') if isinstance(res, dict) else res!r}、"
                                   f"status = {res.get('status') if isinstance(res, dict) else None!r}",
                      "ask() 里检查 stop_reason：不是 'end_turn' 就记下 status，并且不把半截文字当输出")
    return report(name, True)


# ---------- 评测结果表 ----------
_SUMMARY_COLS = {"accuracy": "correct", "mean_llm_calls": "llm_calls", "mean_tool_calls": "tool_calls",
                 "mean_input_tokens": "input_tokens", "mean_output_tokens": "output_tokens"}


def eval_table(results, summary=None):
    """results = {"agent": [记录, ...], "workflow": [...], ...}；每条记录含 id、answer、correct、llm_calls、
    tool_calls、input_tokens、output_tokens。每个系统要覆盖 20 道题，每题重复次数相同（≥1）。
    correct 必须与 agents.grade 一致。重试后仍失败的 API 调用可以记成一条带 "error"（错误信息）的记录：
    它的 answer 应为 None、correct 为 False，并单独计数。可选 summary（DataFrame 或 dict，行=系统）：检查
    accuracy、mean_llm_calls、mean_tool_calls、mean_input_tokens、mean_output_tokens 与记录一致。"""
    name = "评测结果表"
    items = _items()
    if not isinstance(results, dict) or not {"agent", "workflow"} <= set(results):
        return report(name, False, "results 应是至少含 'agent' 与 'workflow' 两个键的 dict")
    means, errors = {}, {}
    for system, recs in results.items():
        if not isinstance(recs, list) or not recs:
            return report(name, False, f"results[{system!r}] 应是非空的记录列表")
        for r in recs:
            if not isinstance(r, dict):
                return report(name, False, f"results[{system!r}] 里的记录应是 dict")
            need = {"id", "answer", "correct", *(_SUMMARY_COLS.values())} - set(r)
            if need:
                return report(name, False, f"results[{system!r}] 的记录缺少字段 {sorted(need)}")
            if r["id"] not in items:
                return report(name, False, f"未知的题号 {r['id']!r}")
            if not isinstance(r["correct"], bool):
                return report(name, False, f"correct 应是 True/False，实际 {r['correct']!r}")
            if r.get("error"):
                if r["answer"] is not None or r["correct"] is not False:
                    return report(name, False, f"{system} / {r['id']}：API 出错的记录应 answer=None、correct=False")
                errors[system] = errors.get(system, 0) + 1
            if r["correct"] != agents.grade(items[r["id"]], r["answer"]):
                return report(name, False, f"{system} / {r['id']}：correct={r['correct']}，但按 agents.grade "
                                           f"判分的结果相反（answer = {r['answer']!r}）",
                              "统一用 agents.grade(item, answer) 判分，不要自己比字符串")
            for k in ("llm_calls", "tool_calls", "input_tokens", "output_tokens"):
                if not isinstance(r[k], int) or isinstance(r[k], bool) or r[k] < 0:
                    return report(name, False, f"{system} / {r['id']}：{k} 应是非负整数，实际 {r[k]!r}")
        counts = Counter(r["id"] for r in recs)
        if set(counts) != set(items) or len(set(counts.values())) != 1:
            missing = sorted(set(items) - set(counts))
            return report(name, False, f"results[{system!r}] 应覆盖全部 20 道题且每题次数相同；"
                                       f"缺 {missing}，次数分布 {sorted(set(counts.values()))}")
        means[system] = {col: sum(float(r[key]) for r in recs) / len(recs) for col, key in _SUMMARY_COLS.items()}
    if summary is not None:
        for system, row in means.items():
            for col, want in row.items():
                try:
                    got = float(summary.loc[system, col]) if hasattr(summary, "loc") else float(summary[system][col])
                except (KeyError, TypeError, ValueError):
                    return report(name, False, f"summary 里找不到 系统 {system!r} 的 {col!r}",
                                  f"summary 的行是系统名，列是 {list(_SUMMARY_COLS)}")
                if abs(got - want) > 1e-6 * max(1.0, abs(want)):
                    return report(name, False, f"summary[{system!r}][{col!r}] = {got:.4f}，按记录算应为 {want:.4f}")
    if errors:
        print("   注意：有 API 调用重试后仍失败的记录（按答错计入准确率，报告里要单独写明）："
              + "，".join(f"{s} {n} 条" for s, n in errors.items()))
    return report(name, True)


# ---------- 失败分析 ----------
def failure_case(case_id):
    """课堂练习用：返回 (题目, 注入了某种故障的 MockLLM)。故障类型请你读轨迹后自己判断
    （这个 MockLLM 的 faults 属性只显示 "custom"，不告诉你注入了什么）。"""
    if case_id not in _EXERCISE_CASES:
        raise KeyError(f"练习里的案例编号是 {sorted(_EXERCISE_CASES)}")
    qid, fault = _EXERCISE_CASES[case_id]
    return _items()[qid], agents.MockLLM(faults={"agent": {qid: fault}})


def _expected_fault(case_id):
    if case_id in _EXERCISE_CASES:
        return _EXERCISE_CASES[case_id][1]
    if isinstance(case_id, str) and "/" in case_id:
        system, qid = case_id.split("/", 1)
        return agents._HOMEWORK_FAULTS.get(system, {}).get(qid)
    return None


def failure_category(label, case_id):
    """label：你判断的失败类别（五类之一，或非正常结束的"拒答 / 截断 / 超步数"之一，中英文都可）；
    case_id：练习里的 "A"–"E"，或作业里 mock 评测（make_backend(mock_faults="homework")）的
    "agent/a01"、"workflow/a10" 这种 "系统/题号"。"""
    name = f"失败类别（{case_id}）"
    fault = _expected_fault(case_id)
    if fault is None:
        return report(name, False, f"{case_id!r} 不是一个注入了故障的案例",
                      "练习里的编号是 A–E；作业里写成 '系统/题号'，而且只标注 mock 评测里真正答错的题")
    got = _canon_label(label)
    if got is None:
        return report(name, False, f"{label!r} 不是五个类别（{'、'.join(CATEGORIES)}）之一，"
                                   f"也不是非正常结束（{'、'.join(STOP_CATEGORIES)}）之一")
    return report(name, got == fault, f"你的判断是「{_LABELS[got]}」，再读一遍轨迹", _DECISION_RULES)


# ---------- 答案规范化 ----------
_MATCH_CASES = [
    ("Final answer: 49", 49, "最简单的情况"),
    ("Final answer: 1,953.", 1953, "去掉数字里的千位分隔逗号，以及句末的句点"),
    ("The answer is 378.0", 378, "数字按数值比较：378.0 == 378"),
    ("1950 and then 1956", 1956, "没有 'Final answer:' 标记时取全文最后一个数"),
    ("I think 1950. Final answer: 1956", 1956, "有标记时只看最后一个 'Final answer:' 之后的部分"),
    ("FINAL ANSWER: alexander fleming.", "Alexander Fleming", "标记大小写不敏感；文字答案要小写、去标点"),
    ("Final answer: The Nature", "Nature", "文字答案要去掉冠词 a / an / the"),
    ("Final answer: 15.3%", 15.3, "数字后面的 % 或单位不影响取数"),
    ("Final answer: 299,792,458 m/s", 299792458, "多个千位逗号"),
    ("Final answer: 21.31", 21, "21.31 不等于 21（容差只吸收浮点误差，默认 1e-6）"),
    ("Final answer: twenty-one", 21, "英文数词不算数字——判错，这正是失败分析里的'格式不匹配'"),
    ("Final answer: about 21.3, so 21 full bags", 21, "有标记时，标记后出现了两个不同的数（21.3 和 21），判错"),
    ("Final answer: 1953, 1954", 1953, "两头下注：标记后有两个不同的数，判错（否则同时写几个候选就能'蒙对'）"),
    ("Final answer: 1950. Wait - Final answer: 1956", 1956, "有两个标记时只看最后一个"),
    ("", 5, "空回答判错"),
    (None, 5, "None（循环没拿到答案）判错，不能报错"),
    ("Final answer: Louis Pasteur", "Alexander Fleming", "不同的名字判错"),
]


def normalized_match(fn):
    """fn(pred, gold) -> bool。与 agents.is_correct 在 17 个边界用例上逐一比对。"""
    name = "答案规范化判分"
    for pred, gold, why in _MATCH_CASES:
        want = agents.is_correct(pred, gold)
        try:
            got = fn(pred, gold)
        except NotImplementedError:
            return report(name, False, "函数还没有实现（仍是 NotImplementedError）")
        except Exception as e:  # noqa: BLE001
            return report(name, False, f"my_is_correct({pred!r}, {gold!r}) 出错 {type(e).__name__}: {e}",
                          "先处理 pred 为 None 或空字符串的情况")
        if not isinstance(got, bool) or got != want:
            return report(name, False, f"my_is_correct({pred!r}, {gold!r}) 应返回 {want}，实际 {got!r}", why)
    return report(name, True)
