"""第19周 AI Agents 的工具包部分：工具、离线语料、评测集、判分，以及一个与 Claude SDK 同构的 MockLLM。

课程要教的（ReAct 循环、固定步骤的 workflow、评测与失败分析）由学习者自己写；
这里只放"课程不教但绕不开"的部分：

- 工具：``calculator``（AST 安全求值）、``wiki_search`` / ``wiki_lookup``（离线维基摘要），
  以及 ``live_wiki_summary``（联网版，必须带 User-Agent；只在学习者自己选择时使用，CI 不调用）。
- 工具定义：``CALCULATOR_TOOL`` 等 JSON 常量（``input_schema`` + ``additionalProperties: False``）。
- 评测：``load_eval()``（20 道题）、``normalize_answer`` / ``is_correct`` / ``grade``（答案规范化后判分）。
- 后端：``MockLLM``（确定性的脚本化"模型"，CI 与无密钥时使用）与 ``make_backend()``。

MockLLM 的 ``create(model=..., max_tokens=..., messages=..., tools=..., system=...)`` 返回的对象与
``anthropic.Anthropic().messages.create(...)`` 的返回值同构：``.content`` 是块列表（``thinking`` /
``text`` / ``tool_use``，tool_use 块有 ``.id`` ``.name`` ``.input``），``.stop_reason``，
``.usage.input_tokens`` / ``.usage.output_tokens``。所以同一个循环既能接 MockLLM，也能接真实 Claude。

MockLLM 对请求的处理尽量与真实 SDK + API 一致（依据官方文档与 SDK 2.3.0，未用真实密钥实测）：

- 真实 SDK 本地就会拒绝的（SDK 2.x 的 ``create()`` 没有 temperature / top_p / top_k，拼错的参数名）
  → 抛 ``TypeError``，请求根本不会发出；
- 真实 API 会返回 HTTP 400 的（经 ``extra_body`` 塞进去的采样参数、强制 tool_choice、关闭思考或
  budget_tokens、assistant 预填、tool_use 缺 tool_result、tool_result 不在最前、改写已发出的历史使
  thinking 块失效）→ 抛 ``MockBadRequestError``；装了 anthropic 时它是 ``anthropic.BadRequestError``
  的子类，所以 ``except anthropic.BadRequestError`` 对两种后端都适用；
- 真实 API 接受、但不推荐的写法（相邻的同角色消息——API 会把它们合并成一轮；同一轮的 tool_result 拆成
  几条 user 消息——模型以后会更少并行调用；回传时删掉 thinking 块——模型丢掉之前的推理）→ 照样接受，
  只发出一次 ``MockLLMWarning`` 提醒。

请求里设了 ``thinking={"type": "adaptive", "display": "summarized"}`` 时，MockLLM 的 thinking 块里会有
一句演示用的推理摘要（对应真实 API 的 summarized 显示）；不设时与真实 API 的默认（omitted）一样是空字符串。

MockLLM 只认识 ``agent_eval.json`` 里的 20 道题：每题有一段预写的轨迹（按对话进行到第几步返回下一个
tool_use 或最终答案），并支持故障注入（提前结束、选错工具、参数错误、幻觉、格式不匹配，以及死循环、
拒答、被截断），供失败分析教学。它的 token 数是按字符数估算的模拟值，
**它的准确率是剧本写出来的，不代表任何真实模型的能力**。
"""
import ast
import copy
import email.utils
import hashlib
import importlib.util
import json
import math
import operator
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from functools import lru_cache
from importlib import resources

if importlib.util.find_spec("anthropic") is not None:     # 只为了让 MockBadRequestError 成为
    import anthropic as _anthropic                         # anthropic.BadRequestError 的子类
    try:
        import httpx2 as _httpx                            # anthropic 1.x / 2.x 基于 httpx2
    except ImportError:                                    # pragma: no cover —— 旧版 SDK 基于 httpx
        import httpx as _httpx
else:                                                      # pragma: no cover —— 没装 anthropic 时退化为普通异常
    _anthropic = _httpx = None

MODEL = "claude-opus-5-5"
"""课程默认模型（Claude Opus 5.5）。请求里不要传 temperature / top_p / top_k，也不要强制 tool_choice。"""

# ---------------------------------------------------------------------------
# 数据文件
# ---------------------------------------------------------------------------


def _data_path(name):
    return resources.files("zero2researcher") / "datasets" / name


@lru_cache(maxsize=None)
def _eval_raw():
    with _data_path("agent_eval.json").open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _wiki_raw():
    with _data_path("wiki_snippets.json").open(encoding="utf-8") as f:
        return json.load(f)


def load_eval():
    """返回 20 道评测题（list of dict）：id、type（"arithmetic" / "lookup"）、question、gold、
    aliases、tol、source（维基条目标题列表）、evidence、min_tool_calls 等字段。

    返回的是副本，不含 MockLLM 的剧本（剧本只在 MockLLM 内部使用）。
    """
    items = copy.deepcopy(_eval_raw()["items"])
    for it in items:
        it.pop("mock", None)
    return items


def load_wiki():
    """离线维基语料：list of {"title", "summary", "url"}（英文维基百科条目的简短改写，按 CC BY-SA 4.0 发布；
    署名与改写说明见 wiki_snippets.json 的 license 字段）。"""
    return copy.deepcopy(_wiki_raw()["articles"])


# ---------------------------------------------------------------------------
# 工具 1：安全计算器
# ---------------------------------------------------------------------------

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_ABS = 1e100
_MAX_LEN = 200


def _check_size(x):
    if abs(x) > _MAX_ABS:
        raise ValueError("result is too large")
    return x


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"only numbers are allowed, found {node.value!r}")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise ValueError("number is too large")                  # 例如 1e999 被解析成 inf
        return _check_size(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 100 or (abs(left) > 1 and abs(right) * math.log10(abs(left)) > 100):
                raise ValueError("power is too large")
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        out = _BINOPS[type(node.op)](left, right)
        if isinstance(out, complex):
            raise ValueError("result is not a real number")
        return _check_size(out)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
        raise ValueError("'^' is not a power operator here; use ** for powers")
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.FloorDiv, ast.Mod)):
        raise ValueError("'//' and '%' are not supported (no integer division or remainder); "
                         "divide with '/' and round the result yourself")
    raise ValueError("only numbers, + - * / ** and parentheses are allowed "
                     f"(found {type(node).__name__})")


def _format_number(x):
    if isinstance(x, int):
        return str(x)
    x = round(x, 10)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return repr(x)


def calculator(expression: str) -> str:
    """安全地计算一个算术表达式，返回结果的字符串（如 "14"、"21.3076923077"）。

    只允许数字、+ - * / **、一元正负号与括号；用 ast 解析后逐节点求值，**从不调用 eval**，
    所以 ``__import__('os')``、变量名、函数调用、属性访问一律被拒绝。任何错误（包括溢出、
    ``0 ** -1`` 这类除零）都统一抛出 ValueError（在 Agent 循环里应捕获它，作为 is_error 的
    tool_result 返回给模型）。
    """
    if not isinstance(expression, str):
        raise ValueError(f"expression must be a string, got {type(expression).__name__}")
    expr = expression.strip().replace("×", "*").replace("÷", "/").replace("−", "-")
    if not expr:
        raise ValueError("empty expression")
    if len(expr) > _MAX_LEN:
        raise ValueError(f"expression is longer than {_MAX_LEN} characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ValueError(f"not a valid arithmetic expression: {expression!r}") from None
    try:
        return _format_number(_eval_node(tree))
    except ArithmeticError as e:                    # OverflowError、ZeroDivisionError 等
        raise ValueError(f"arithmetic error: {e}") from None


# ---------------------------------------------------------------------------
# 工具 2、3：离线维基检索
# ---------------------------------------------------------------------------

_STOPWORDS = set("""a an the of in on at and or for to is was were be been by with from as that this
it its what which who whom when where how many much did does do year years first published""".split())


def _tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t not in _STOPWORDS]


@lru_cache(maxsize=None)
def _wiki_index():
    arts = _wiki_raw()["articles"]
    docs = [(set(_tokens(a["title"])), set(_tokens(a["summary"]))) for a in arts]
    n = len(docs)
    df = {}
    for title_t, summ_t in docs:
        for t in title_t | summ_t:
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(1 + n / c) for t, c in df.items()}
    return arts, docs, idf


def _format_article(a):
    return f"{a['title']}: {a['summary']}"


def wiki_search(query: str, k: int = 3) -> str:
    """在离线维基摘要上做关键词检索，返回最匹配的至多 k 篇（"标题: 摘要"，每篇一段，最相关的在前）。

    打分：查询词（去掉停用词）出现在标题里记 2 倍、出现在摘要里记 1 倍，再乘以该词的 IDF。
    没有任何命中时返回一句 "No results ..."（这不是错误，模型应换关键词再搜）。
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    arts, docs, idf = _wiki_index()
    q = set(_tokens(query))
    scored = []
    for i, (title_t, summ_t) in enumerate(docs):
        s = sum(idf.get(t, 0.0) * (2.0 * (t in title_t) + 1.0 * (t in summ_t)) for t in q)
        if s > 0:
            scored.append((-s, i))
    scored.sort()
    if not scored:
        return f"No results for {query!r}. Try different keywords."
    return "\n\n".join(_format_article(arts[i]) for _, i in scored[:k])


def wiki_lookup(title: str) -> str:
    """按**确切标题**（大小写不敏感）返回一篇条目的摘要；找不到时抛出 ValueError。"""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    want = title.strip().lower().replace("_", " ")
    for a in _wiki_raw()["articles"]:
        if a["title"].lower() == want:
            return _format_article(a)
    raise ValueError(f"No article titled {title!r}. Use wiki_search to find the exact title.")


DEFAULT_USER_AGENT = "Zero2ResearcherCourse/1.0 (educational use)"
"""通用的课程标识，不含任何个人信息。按维基媒体的 User-Agent 政策
（https://meta.wikimedia.org/wiki/User-Agent_policy ），自己联网时请换成含你本人联系方式的字符串。"""

_WIKI_MIN_INTERVAL = 1.0                 # 连续请求之间至少间隔 1 秒
_wiki_last_request = [float("-inf")]


def _retry_after_seconds(value, attempt):
    """解析 429 响应的 Retry-After（秒数或 HTTP 日期）；没有就按 1、2、4 秒指数退避。最多等 60 秒。"""
    if value:
        try:
            return min(max(0.0, float(value)), 60.0)
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(value).timestamp()
                return min(max(0.0, when - time.time()), 60.0)
            except (TypeError, ValueError):
                pass
    return float(2 ** attempt)


def live_wiki_summary(title: str, user_agent: str = DEFAULT_USER_AGENT, timeout: float = 10.0,
                      max_retries: int = 3) -> str:
    """联网版 wiki_lookup：调用维基百科 REST 摘要端点，返回 "标题: 导语"。

    **必须**带一个描述性的 User-Agent——不带、或用 Python 默认 UA（``python-requests/...``、
    ``Python-urllib/...``）都会被维基百科一律返回 403；照抄浏览器的 UA（``Mozilla/5.0 ...``）也是
    API 礼仪明确不允许的。默认的 DEFAULT_USER_AGENT 只是一个通用的课程标识（不含个人信息）；按维基媒体的
    User-Agent 政策（https://meta.wikimedia.org/wiki/User-Agent_policy ），请换成含你自己联系方式的字符串。

    一次只发一个请求、不要并发：本函数让连续请求之间至少间隔 1 秒；遇到 HTTP 429（请求太多）时，
    按响应的 Retry-After 等待后重试，最多 max_retries 次，仍失败就把 HTTPError 抛出来。
    课程 CI 从不调用这个函数；默认用离线的 wiki_search / wiki_lookup。
    """
    if not isinstance(user_agent, str) or not user_agent.strip():
        raise ValueError("pass a descriptive user_agent (Wikipedia rejects requests without one, HTTP 403)")
    ua = user_agent.strip().lower()
    if ua.startswith("python"):
        raise ValueError("Wikipedia rejects default Python user agents (HTTP 403); "
                         "pass a descriptive user_agent")
    if ua.startswith(("mozilla/", "curl/", "wget/")):
        raise ValueError("do not copy a browser or tool user agent; describe your own client, "
                         "see https://meta.wikimedia.org/wiki/User-Agent_policy")
    slug = urllib.parse.quote(title.strip().replace(" ", "_"), safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent.strip(), "Accept": "application/json"})
    for attempt in range(max_retries + 1):
        wait = _WIKI_MIN_INTERVAL - (time.monotonic() - _wiki_last_request[0])
        if wait > 0:
            time.sleep(wait)
        _wiki_last_request[0] = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 (fixed https host)
                data = json.load(resp)
            return f"{data.get('title', title)}: {data.get('extract', '')}"
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == max_retries:
                raise
            time.sleep(_retry_after_seconds(e.headers.get("Retry-After") if e.headers else None, attempt))
    raise RuntimeError("unreachable")      # pragma: no cover


TOOL_FUNCTIONS = {"calculator": calculator, "wiki_search": wiki_search, "wiki_lookup": wiki_lookup}
"""工具名 → Python 函数。Agent 循环里执行 tool_use 块：``TOOL_FUNCTIONS[block.name](**block.input)``。"""

# ---------------------------------------------------------------------------
# 工具定义（传给 messages.create 的 tools=[...]）
# ---------------------------------------------------------------------------

CALCULATOR_TOOL = {
    "name": "calculator",
    "description": ("Evaluate one arithmetic expression and return its numeric value as text. "
                    "Supports numbers, + - * / ** and parentheses only (no variables, functions or units). "
                    "Use it for every arithmetic step instead of computing in your head; "
                    "call it again for each intermediate result you need."),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string",
                           "description": "An arithmetic expression, e.g. '(84 * 2.5) + (60 * 1.5)'."},
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
}

WIKI_SEARCH_TOOL = {
    "name": "wiki_search",
    "description": ("Search a small offline copy of English Wikipedia article summaries by keywords. "
                    "Returns up to 3 matching articles as 'Title: summary' paragraphs, best match first, "
                    "or a 'No results' message. Use it to look up facts such as dates, names and numbers."),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "A few keywords, e.g. 'Marie Curie Nobel Prize'."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

WIKI_LOOKUP_TOOL = {
    "name": "wiki_lookup",
    "description": ("Return the summary of one article from the offline Wikipedia copy, given its exact "
                    "title as shown by wiki_search. Returns an error if no article has that title."),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The exact article title, e.g. 'Marie Curie'."},
        },
        "required": ["title"],
        "additionalProperties": False,
    },
}

TOOLS = [CALCULATOR_TOOL, WIKI_SEARCH_TOOL, WIKI_LOOKUP_TOOL]

# ---------------------------------------------------------------------------
# 提示词（MockLLM 靠这些常量识别 workflow 的每一步；真实 Claude 也直接用它们）
# ---------------------------------------------------------------------------

AGENT_SYSTEM = (
    "You answer questions with the help of tools. Use the calculator for every arithmetic step "
    "and wiki_search to look up facts; do not rely on memory for facts. You may call several tools "
    "in one turn when they do not depend on each other. Before each tool call, state in one short "
    "sentence what you are about to do and why. When you are done, reply with one short "
    "sentence and then a last line 'Final answer: <answer>', where <answer> is only a number "
    "(no units, no thousands separators) or a short name.")

DIRECT_SYSTEM = (
    "Answer the question without using any tools. Reply with one short sentence and then a last "
    "line 'Final answer: <answer>', where <answer> is only a number (no units, no thousands "
    "separators) or a short name.")

WORKFLOW_PROMPTS = {
    "route": ("You are a router. Decide whether answering the question needs facts from an encyclopedia "
              "(LOOKUP) or only arithmetic on numbers given in the question (ARITHMETIC). "
              "Reply with exactly one word: LOOKUP or ARITHMETIC."),
    "query": ("Write the encyclopedia search queries needed to answer the question: one short keyword "
              "query per line, at most 2 lines, nothing else."),
    "expression": ("Using only the question and the notes, write ONE arithmetic expression (numbers, "
                   "+ - * / ** and parentheses) whose value answers the question. If no calculation is "
                   "needed, reply NONE. Reply with the expression or NONE only."),
    "answer": ("Answer the question using only the notes. Reply with one short sentence and then a last "
               "line 'Final answer: <answer>', where <answer> is only a number (no units, no thousands "
               "separators) or a short name."),
}

# ---------------------------------------------------------------------------
# 判分：答案规范化
# ---------------------------------------------------------------------------

_NUM = r"(?<![\w.])-?\d+(?:\.\d+)?"
_ARTICLES = {"a", "an", "the"}


def _clean(text):
    text = str(text).replace("−", "-").replace(" ", " ")
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)          # 1,953 → 1953；1,2 不动


def extract_final_answer(text):
    """取最后一个 "Final answer:"（大小写不敏感）之后的文字；没有这个标记就返回整段文字。"""
    if text is None:
        return ""
    text = str(text)
    hits = list(re.finditer(r"final\s+answer\s*[:：]", text, flags=re.I))
    return (text[hits[-1].end():] if hits else text).strip()


def _canon_number(tok):
    v = float(tok)
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else repr(v)


def normalize_answer(s):
    """把一个简短答案规范化成可比较的字符串：小写、去标点、去冠词（a/an/the）、合并空白、
    去掉数字里的千位分隔逗号，并把 "378.0" 这类数字写成 "378"。

    >>> normalize_answer("The  Eiffel Tower!")
    'eiffel tower'
    >>> normalize_answer("1,953.")
    '1953'
    """
    if s is None:
        return ""
    text = _clean(s).lower()
    out = []
    for tok in re.findall(_NUM + r"|[^\W_]+", text):
        if tok in _ARTICLES:
            continue
        if re.fullmatch(r"-?\d+(?:\.\d+)?", tok):
            tok = _canon_number(tok)
        out.append(tok)
    return " ".join(out)


def _numbers(text):
    return [float(x) for x in re.findall(_NUM, _clean(text))]


def _as_number(gold):
    if isinstance(gold, bool):
        return None
    if isinstance(gold, (int, float)):
        return float(gold)
    if isinstance(gold, str) and re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", _clean(gold)):
        return float(_clean(gold))
    return None


def is_correct(pred, gold, tol=None):
    """规范化之后判断预测答案 pred 是否等于标准答案 gold。

    1. 先取最后一个 "Final answer:" 之后的文字（没有标记就用全文）；
    2. gold 是数字时：去掉千位分隔逗号后找出所有数字——有标记时，标记后的数必须**只有一个值**
       （出现两个不同的数，例如 "1953, 1954"、"2010-2017"、"210 + 90 = 300"，就是"两头下注"或没有
       只写答案，判错），没有标记取全文**最后一个**数；与 gold 之差不超过 tol（默认 1e-6）算对。
       "15.3%"、"300 km"、"1,953." 都能对上；"twenty-one"、"about 21.3, so 21" 对不上；
    3. gold 是文字时：两边 normalize_answer 之后完全相等算对；
    4. gold 也可以是若干可接受答案的列表，任一对上即算对。
    """
    if isinstance(gold, (list, tuple)):
        return any(is_correct(pred, g, tol) for g in gold)
    if pred is None:
        return False
    text = str(pred)
    has_marker = re.search(r"final\s+answer\s*[:：]", text, flags=re.I) is not None
    answer = extract_final_answer(text)
    g = _as_number(gold)
    if g is not None:
        nums = _numbers(answer)
        if not nums or (has_marker and len(set(nums)) > 1):
            return False
        p = nums[0] if has_marker else nums[-1]
        t = 1e-6 if tol is None else float(tol)
        return abs(p - g) <= t + 1e-9 * abs(g)
    return normalize_answer(answer) == normalize_answer(gold) and normalize_answer(gold) != ""


def grade(item, answer):
    """按评测题 item 的 gold、aliases 与 tol 判分：``grade(item, result["answer"])`` → bool。"""
    golds = [item["gold"]] + list(item.get("aliases", []))
    return is_correct(answer, golds, item.get("tol"))


# ---------------------------------------------------------------------------
# 与 Claude SDK 同构的响应对象
# ---------------------------------------------------------------------------


class _Obj:
    """有属性、能转 dict 的小对象（字段名与 anthropic.types 中的同名类型一致）。"""
    _fields = ()

    def __init__(self, **kw):
        for f in self._fields:
            setattr(self, f, kw.get(f))

    def to_dict(self):
        out = {}
        for f in self._fields:
            v = getattr(self, f)
            if isinstance(v, _Obj):
                v = v.to_dict()
            elif isinstance(v, list):
                v = [x.to_dict() if isinstance(x, _Obj) else x for x in v]
            elif isinstance(v, dict):
                v = copy.deepcopy(v)
            out[f] = v
        return out

    model_dump = to_dict

    def __getitem__(self, key):              # 允许 block["type"] 这种字典式访问
        if key in self._fields:
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key, default=None):
        return getattr(self, key, default) if key in self._fields else default

    def __repr__(self):
        body = ", ".join(f"{f}={getattr(self, f)!r}" for f in self._fields)
        return f"{type(self).__name__}({body})"

    def __eq__(self, other):
        return isinstance(other, _Obj) and self.to_dict() == other.to_dict()


class TextBlock(_Obj):
    _fields = ("type", "text", "citations")


class ThinkingBlock(_Obj):
    _fields = ("type", "thinking", "signature")


class ToolUseBlock(_Obj):
    _fields = ("type", "id", "name", "input")


class Usage(_Obj):
    _fields = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


class StopDetails(_Obj):
    _fields = ("type", "category", "explanation")


class MockMessage(_Obj):
    """与 anthropic.types.Message 同名同义的字段：id、type、role、model、content、stop_reason、
    stop_sequence、stop_details、usage。"""
    _fields = ("id", "type", "role", "model", "content", "stop_reason", "stop_sequence",
               "stop_details", "usage")


_BadRequestBase = _anthropic.BadRequestError if _anthropic is not None else Exception


class MockBadRequestError(_BadRequestBase):
    """MockLLM 模拟真实 API 的 HTTP 400：请求违反了真实 API 也会拒绝的规则。

    装了 anthropic 时它是 ``anthropic.BadRequestError`` 的子类（带 ``status_code``、``message``、
    ``body``，``body`` 的结构与 API 的错误响应相同），所以 ``except anthropic.BadRequestError``
    对真实后端和 MockLLM 都适用。"""
    status_code = 400

    def __init__(self, message):
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
        if _anthropic is not None:
            request = _httpx.Request("POST", "https://api.anthropic.com/v1/messages")   # 只是个占位，从不发出
            super().__init__(message, response=_httpx.Response(400, request=request, json=body), body=body)
        else:                                       # pragma: no cover
            super().__init__(message)
            self.message, self.body = message, body


class MockLLMWarning(UserWarning):
    """MockLLM 的提醒：这种写法真实 API 也接受（不报错），但不推荐，并说明真实的后果。"""


class MockLoopGuardError(RuntimeError):
    """同一段对话里调用 MockLLM 超过上限：你的 Agent 循环多半缺少最大步数保护。"""


def _btype(block):
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _bget(block, key, default=None):
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


def _blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content or [])


def _text_of(content):
    parts = []
    for b in _blocks(content):
        t = _btype(b)
        if t == "text":
            parts.append(str(_bget(b, "text", "")))
        elif t == "tool_result":
            c = _bget(b, "content", "")
            parts.append(c if isinstance(c, str) else _text_of(c))
    return "\n".join(parts)


def _jsonable(x):
    if isinstance(x, _Obj):
        return x.to_dict()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "model_dump"):
        return x.model_dump()
    return x


def estimate_tokens(obj):
    """粗估 token 数：JSON 序列化后的字符数 / 4（英文文本的常见经验值）。MockLLM 用它模拟 usage。"""
    if obj is None:
        return 0
    s = obj if isinstance(obj, str) else json.dumps(_jsonable(obj), ensure_ascii=False, default=str)
    return math.ceil(len(s) / 4)


def _canon(x):
    """规范化成可比较的 JSON 结构：去掉值为 None 的字段与 cache_control 标记（它们不影响对话内容）。"""
    x = _jsonable(x)
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in x.items() if v is not None and k != "cache_control"}
    if isinstance(x, list):
        return [_canon(v) for v in x]
    return x


def print_transcript(messages, width=110):
    """把一段 Agent 对话（messages 列表）打印成易读的轨迹：谁说了什么、调了什么工具、工具返回了什么。

    thinking 块打印它的文字（请求设了 ``display: "summarized"`` 时是推理摘要，也是 ReAct 的 Thought
    的一部分）；文字为空时（默认的 ``display: "omitted"``）注明是空块。"""
    for m in messages:
        role = m["role"] if isinstance(m, dict) else getattr(m, "role", "?")
        content = m["content"] if isinstance(m, dict) else getattr(m, "content", "")
        for b in _blocks(content):
            t = _btype(b)
            if t == "text":
                line = f"[{role}] {_bget(b, 'text', '')}"
            elif t == "thinking":
                thought = str(_bget(b, "thinking", "") or "")
                line = (f"[{role}] (thinking 摘要) {thought}" if thought else
                        f"[{role}] (thinking 块：文字为空——默认 display=\"omitted\")")
            elif t == "tool_use":
                line = (f"[{role}] → tool_use {_bget(b, 'name')}"
                        f"({json.dumps(_bget(b, 'input'), ensure_ascii=False)})")
            elif t == "tool_result":
                c = _bget(b, "content", "")
                c = c if isinstance(c, str) else _text_of(c)
                flag = " [is_error]" if _bget(b, "is_error", False) else ""
                line = f"[{role}] ← tool_result{flag}: {c}"
            else:
                line = f"[{role}] ({t})"
            line = line.replace("\n", " ⏎ ")
            print(line if len(line) <= width else line[: width - 1] + "…")


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------

FAULT_TYPES = ("early_stop", "wrong_tool", "bad_args", "hallucination", "format_mismatch",
               "loop", "refusal", "truncated")
"""MockLLM 能注入的故障。前五种对应失败分析的五个类别；loop（一直重复同一个工具调用）、refusal（拒答）与
truncated（回复被 max_tokens 截断）用来测试循环怎样处理"非正常结束"。"""

FAULT_CATEGORY = {
    "wrong_tool": "工具选择错误",
    "bad_args": "参数错误",
    "early_stop": "提前结束",
    "hallucination": "幻觉/编造",
    "format_mismatch": "格式不匹配",
}
"""故障 → 失败分析的五个类别（中文名）。"""

STOP_CATEGORY = {
    "refusal": "拒答",
    "truncated": "截断",
    "loop": "超步数",
}
"""三种"非正常结束"的故障 → 失败初筛里单独计数的类别：拒答（stop_reason == "refusal"）、截断
（stop_reason == "max_tokens"）、超步数（用完 max_steps 仍没有结束，例如一直重复同一个调用——ReAct 原文
把"无法从重复步骤中恢复"归入推理错误）。它们不进入上面的五类。"""

DEMO_FAULTS = {
    "agent": {"a08": "wrong_tool", "a04": "bad_args", "f02": "early_stop",
              "f01": "hallucination", "a05": "format_mismatch"},
    "workflow": {"f06": "wrong_tool", "a06": "bad_args", "f03": "format_mismatch"},
}
"""make_backend() 在没有密钥时默认使用的"会犯错的模拟模型"（讲义演示用）：按真实模型常见的失败方式预先
注入的 8 处故障。作业用另一份故障分布（make_backend(mock_faults="homework")），课件里不展示。"""

_HOMEWORK_FAULTS = {
    "agent": {"a01": "wrong_tool", "f03": "bad_args", "a06": "early_stop",
              "f06": "hallucination", "a03": "format_mismatch", "a09": "loop"},
    "workflow": {"f05": "wrong_tool", "a10": "bad_args", "f08": "hallucination",
                 "f02": "format_mismatch", "a07": "refusal"},
}
# 作业 D 的失败分析用这一份（与讲义演示的 DEMO_FAULTS 不同）；标准标注只在开发仓库里。

_FAULT_PRESETS = {"demo": DEMO_FAULTS, "homework": _HOMEWORK_FAULTS}

_TOOL_OVERHEAD = 286          # 带 tools 的请求，API 会额外注入约 286 个 token 的工具系统提示（auto）
_MAX_CALLS_PER_CONVERSATION = 50

_UNKNOWN_REPLY = ("（MockLLM）我只认识课程评测集 agents.load_eval() 里的 20 道题，这个问题不在其中，"
                  "所以没有剧本可演。想问任意问题，请配置 ANTHROPIC_API_KEY 使用真实的 Claude。\n"
                  "(MockLLM only knows the 20 course evaluation questions.)")

_REFUSAL_PARTIAL = "Sure, let me start by"     # 模拟"输出到一半被拒答"：content 里只剩半句话

_SAMPLING = ("temperature", "top_p", "top_k")
_THINKING_TYPES = ("thinking", "redacted_thinking")

# anthropic SDK 2.3.0 的 Messages.create() 接受的全部关键字参数（装了别的版本时再并上它的签名）
_SDK_CREATE_KWARGS = {"model", "max_tokens", "messages", "cache_control", "container", "diagnostics",
                      "inference_geo", "metadata", "output_config", "service_tier", "speed", "stop_sequences",
                      "stream", "system", "thinking", "tool_choice", "tools", "user_profile_id", "workspace_id",
                      "extra_headers", "extra_query", "extra_body", "timeout"}


def _sdk_create_kwargs():
    names = set(_SDK_CREATE_KWARGS)
    if _anthropic is not None:
        try:
            import inspect                                      # noqa: PLC0415
            from anthropic.resources.messages import Messages   # noqa: PLC0415
            names |= {p for p in inspect.signature(Messages.create).parameters if p != "self"}
        except Exception:                                       # noqa: BLE001 —— SDK 内部结构变了也不影响 mock
            pass
    return names


_ALLOWED_KWARGS = _sdk_create_kwargs()

_BOUND_MSG = (
    "400 invalid_request_error（模拟）: messages.{i}.content.{j}: Invalid `signature` in `thinking` block. "
    "The block is bound to a different conversation. Remove the block, or set "
    "`thinking.block_binding.prefix_mismatch_behavior` to \"drop_block\". That setting requires the "
    "`thinking-binding-controls-2026-08-01` value in the `anthropic-beta` header. "
    "——（MockLLM 模拟 2026-08-31 及以后创建的账号的默认行为）这个 thinking 块之前的内容（system、tools "
    "或更早的某条消息）和它产生时不一样了：在 Opus 5.5 上，历史必须只追加。要清理旧的工具结果或压缩上下文，"
    "请用服务器端的 context editing（clear_tool_uses_20250919）或 compaction，不要在客户端改写已发出的消息。")


def _contains(evidence_text, needle):
    """证据检查：数字按数值比较（容差 1e-6），文字按大小写不敏感的子串比较。"""
    g = _as_number(needle)
    if g is not None:
        return any(abs(x - g) <= 1e-6 * max(1.0, abs(g)) for x in _numbers(evidence_text))
    return str(needle).lower() in str(evidence_text).lower()


def _requirements(item):
    req = list(item.get("evidence", []))
    if item.get("type") == "lookup" and item.get("needs_calculation"):
        req.append(str(item["gold"]))
    return req


def _display_on(kwargs):
    """请求是否要求返回 thinking 的文字（display 为 "summarized"；"updates" 是 beta，这里同样处理）。"""
    think = kwargs.get("thinking")
    return isinstance(think, dict) and think.get("display") in ("summarized", "updates")


def _thought_for(calls):
    """演示用的一句推理摘要：display="summarized" 时放进 thinking 块（对应真实 API 的摘要）。"""
    if not calls:
        return "I have what I need, so I will write the final answer now."
    parts = []
    for c in calls:
        arg = next(iter(c.get("input", {}).values()), "")
        parts.append(f"{c['name']} with {arg!r}")
    tail = "; they do not depend on each other, so both go in this turn" if len(calls) > 1 else ""
    return "Next I will call " + " and ".join(parts) + tail + "."


def _one_call_per_step(traj):
    """disable_parallel_tool_use=True：一轮至多一个工具调用，把并行的几个调用拆成依次进行的几步。"""
    out = []
    for step in traj:
        calls = step.get("tool_calls", [])
        if len(calls) <= 1:
            out.append(step)
            continue
        for n, c in enumerate(calls):
            out.append({"text": step.get("text", "") if n == 0 else "Now the next lookup.", "tool_calls": [c]})
    return out


class MockLLM:
    """确定性的脚本化"模型"，接口与 ``anthropic.Anthropic().messages`` 相同：``create(**kwargs)``。

    参数 faults：
      - None（默认）：每道题都按标准轨迹作答（全部正确）；
      - "demo"：使用 DEMO_FAULTS（讲义演示用的"偶尔犯错的模型"）；
      - "homework"：作业用的另一份故障分布（课件里不展示）；
      - 一个故障名（如 "early_stop"）：所有题都注入这一种故障；
      - {qid: 故障名}：只在这些题上注入（Agent 与 workflow 两种模式都受影响）；
      - {"agent": {...}, "workflow": {...}}：两种模式分别指定。

    只读属性 ``faults`` 是预设名（None、"demo"、"homework" 或那个故障名）；用 dict 指定时只显示
    "custom"——具体注入了什么不公开，失败类别要靠读轨迹判断。

    模式由请求自动判断：system 里含 WORKFLOW_PROMPTS 的某一条 → workflow 的那一步；
    带 tools → Agent 模式；都没有 → 单次调用（直接作答）。请求参数与 SDK 2.x 的 messages.create() 相同；
    ``tool_choice={"type": "none"}`` 时不调用工具，``disable_parallel_tool_use=True`` 时一轮至多一个调用，
    ``max_tokens=0``（缓存预热）返回空 content 与 stop_reason "max_tokens"。
    属性 ``n_calls``、``total_input_tokens``、``total_output_tokens``、``log`` 记录调用情况。
    """

    def __init__(self, faults=None):
        self._preset = faults if (faults is None or isinstance(faults, str)) else "custom"
        self._faults = self._parse_faults(faults)
        self._items = {it["id"]: it for it in _eval_raw()["items"]}
        self._bindings = {}      # thinking 块的 signature → (产生它时的对话前缀, 它前面的 thinking 块, 文字)
        self._warned = set()
        self._cur = {"keys": [None], "last_sig": None}
        self.reset_stats()

    @property
    def faults(self):
        """预设名：None、"demo"、"homework"、一个故障名，或 "custom"（用 dict 指定时不公开细节）。"""
        return self._preset

    # ----- 配置 -----
    @staticmethod
    def _parse_faults(faults):
        if faults is None:
            return {"agent": {}, "workflow": {}, "all": None}
        if isinstance(faults, str) and faults in _FAULT_PRESETS:
            faults = _FAULT_PRESETS[faults]
        if isinstance(faults, str):
            if faults not in FAULT_TYPES:
                raise ValueError(f"未知的故障 {faults!r}；可选 {FAULT_TYPES} 或预设 {tuple(_FAULT_PRESETS)}")
            return {"agent": {}, "workflow": {}, "all": faults}
        if not isinstance(faults, dict):
            raise ValueError("faults 应是 None、'demo'、'homework'、一个故障名，或 dict")
        if faults and set(faults) <= {"agent", "workflow"}:
            per = {"agent": dict(faults.get("agent", {})), "workflow": dict(faults.get("workflow", {})),
                   "all": None}
        else:
            per = {"agent": dict(faults), "workflow": dict(faults), "all": None}
        for mode in ("agent", "workflow"):
            for qid, f in per[mode].items():
                if f not in FAULT_TYPES:
                    raise ValueError(f"{qid}: 未知的故障 {f!r}；可选 {FAULT_TYPES}")
        return per

    def _fault(self, qid, mode):
        if self._faults["all"]:
            return self._faults["all"]
        return self._faults.get(mode, {}).get(qid)

    def reset_stats(self):
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.log = []
        self._ids = 0

    def _warn(self, kind, message):
        """同一个 MockLLM 对同一类写法只提醒一次。"""
        if kind not in self._warned:
            self._warned.add(kind)
            warnings.warn(message, MockLLMWarning, stacklevel=4)      # 指向调用 create() 的那一行

    # ----- 请求校验：SDK 本地的 TypeError、API 的 400，以及真实 API 接受但不推荐的写法 -----
    def _validate(self, kwargs):
        unknown = set(kwargs) - _ALLOWED_KWARGS
        if unknown:
            raise TypeError(f"Messages.create() got an unexpected keyword argument {sorted(unknown)[0]!r}")
        missing = [p for p in ("max_tokens", "messages", "model") if p not in kwargs]
        if missing:
            raise TypeError("Missing required arguments; Expected ('max_tokens', 'messages' and 'model') "
                            f"arguments to be given (missing: {', '.join(missing)})")
        extra = kwargs.get("extra_body") if isinstance(kwargs.get("extra_body"), dict) else {}
        for p in _SAMPLING:
            if p in kwargs or p in extra:          # 只可能经 extra_body（或旧版 SDK）发出去
                raise MockBadRequestError(
                    f"400 invalid_request_error（模拟）: `{p}` is not supported on API version 2026-10-21. "
                    f"——新版 API 不再接受采样参数，请删掉 {p}（SDK 2.x 的 create() 也没有这个参数，直接传会在"
                    "本地抛 TypeError）；评测的随机性要靠每题重复多次来处理。")
        if kwargs.get("stream"):
            raise NotImplementedError("MockLLM 不支持流式输出（stream=True）；课程的循环用非流式请求即可。")
        max_tokens = kwargs["max_tokens"]
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
            raise MockBadRequestError("400 invalid_request_error（模拟）: max_tokens 必须是非负整数。")
        output_config = kwargs.get("output_config")
        if max_tokens == 0 and isinstance(output_config, dict) and output_config.get("format"):
            raise MockBadRequestError("400 invalid_request_error（模拟）: max_tokens=0（缓存预热）"
                                      "不能与 output_config.format 一起用。")
        think = kwargs.get("thinking")
        if think is not None:
            ttype = think.get("type") if isinstance(think, dict) else None
            if ttype in ("disabled", "enabled"):
                raise MockBadRequestError(
                    f'400 invalid_request_error（模拟）: "thinking.type.{ttype}" is not supported for this model. '
                    'Use "thinking.type.adaptive" and "output_config.effort" to control thinking behavior. '
                    "——claude-opus-5-5 不能关闭思考，也不接受 budget_tokens；想省钱请用 "
                    "output_config={'effort': 'low'}。")
            if ttype != "adaptive":
                raise MockBadRequestError("400 invalid_request_error（模拟）: thinking 应是 "
                                          "{'type': 'adaptive'}（可加 'display': 'summarized'）。")
        tc = kwargs.get("tool_choice")
        if tc is not None:
            ttype = tc.get("type") if isinstance(tc, dict) else None
            if ttype in ("any", "tool"):
                raise MockBadRequestError(
                    '400 invalid_request_error（模拟）: tool_choice: type "tool" and "any" are not supported '
                    "for this model. ——请用默认的 auto，并在提示词里说明该用哪个工具。")
            if ttype not in ("auto", "none"):
                raise MockBadRequestError("400 invalid_request_error（模拟）: tool_choice 应是 "
                                          "{'type': 'auto'} 或 {'type': 'none'}。")
        msgs = kwargs["messages"]
        if not isinstance(msgs, list) or not msgs:
            raise MockBadRequestError("400 invalid_request_error（模拟）: messages 必须是非空列表。")
        turns, merged = [], set()
        for i, m in enumerate(msgs):
            if not isinstance(m, dict) or m.get("role") not in ("user", "assistant", "system") or "content" not in m:
                raise MockBadRequestError(
                    f"400 invalid_request_error（模拟）: messages[{i}] 应是 "
                    "{'role': 'user' 或 'assistant', 'content': ...} 形式的 dict。")
            role, blocks = m["role"], _blocks(m["content"])
            if role == "system":                     # 对话中途的 system 消息（Opus 5.5 支持）：剧本里忽略
                if i == 0:
                    raise MockBadRequestError("400 invalid_request_error（模拟）: messages[0] 不能是 system "
                                              "消息；系统提示请用 system= 参数。")
                continue
            if turns and turns[-1][0] == role:       # 与真实 API 一样：相邻的同角色消息合并成一轮
                turns[-1][1].extend(blocks)
                merged.add("tool_result" if any(_btype(b) == "tool_result" for b in blocks) else role)
            else:
                turns.append((role, list(blocks)))
        if turns[0][0] != "user":
            raise MockBadRequestError("400 invalid_request_error（模拟）: 第一条消息必须是 user。")
        pending, dropped_thinking = None, False
        for role, blocks in turns:
            result_ids = [_bget(b, "tool_use_id") for b in blocks if _btype(b) == "tool_result"]
            if role == "user":
                if pending:
                    missing = [t for t in pending if t not in result_ids]
                    if missing:
                        raise MockBadRequestError(
                            "400 invalid_request_error（模拟）: tool_use ids were found without tool_result "
                            f"blocks immediately after: {', '.join(missing)}. ——上一条 assistant 消息里的"
                            "每个 tool_use 都要在紧随其后的 user 消息里有对应的 tool_result"
                            "（tool_use_id 与 block.id 相同）。")
                    extra_ids = [t for t in result_ids if t not in pending]
                    if extra_ids:
                        raise MockBadRequestError(
                            "400 invalid_request_error（模拟）: unexpected `tool_use_id` found in "
                            f"`tool_result` blocks: {', '.join(map(str, extra_ids))}.")
                    if _btype(blocks[0]) != "tool_result":
                        raise MockBadRequestError(
                            "400 invalid_request_error（模拟）: tool_result 块必须排在这条 user 消息的最前面"
                            "（文字只能放在所有 tool_result 之后）。")
                elif result_ids:
                    raise MockBadRequestError(
                        "400 invalid_request_error（模拟）: 出现了 tool_result，但上一条 assistant 消息里"
                        "没有对应的 tool_use。")
                pending = None
            else:
                uses = [_bget(b, "id") for b in blocks if _btype(b) == "tool_use"]
                if uses and not any(_btype(b) in _THINKING_TYPES for b in blocks):
                    dropped_thinking = True
                pending = uses or None
        if turns[-1][0] != "user":
            raise MockBadRequestError(
                "400 invalid_request_error（模拟）: This model does not support assistant message prefill. "
                "The conversation must end with a user message.")
        keys = self._prefix_keys(kwargs)
        last_sig = self._check_bindings(msgs, keys)
        # 到这里请求已被接受；对真实 API 也接受、但不推荐的写法各提醒一次
        if "tool_result" in merged:
            self._warn("split_results",
                       "（MockLLM 提醒）同一轮的 tool_result 被拆成了几条相邻的 user 消息。真实 API 会把相邻的同角色"
                       "消息合并成一轮，不会报错；但拆开发送会让模型以后更少并行调用工具（官方 Parallel tool use 文档）。"
                       "请把同一轮的全部 tool_result 放进同一条 user 消息。")
        elif merged:
            self._warn("same_role",
                       "（MockLLM 提醒）出现了相邻的同角色消息。真实 API 会把它们合并成一轮，不会报错；"
                       "一般不需要这样写。")
        if dropped_thinking:
            self._warn("dropped_thinking",
                       "（MockLLM 提醒）有一条带 tool_use 的 assistant 消息里没有 thinking 块。真实 API 接受（不报错），"
                       "但模型会丢掉那一轮的推理。请把 response.content 原样追加为 assistant 消息。")
        return turns, keys, last_sig

    # ----- thinking 块与对话绑定（Opus 5.5 的 preserved thinking：历史必须只追加） -----
    @staticmethod
    def _prefix_keys(kwargs):
        """keys[i] = system、tools（按名字排序）与 messages[:i] 的摘要；thinking 块本身不计入前缀。"""
        system = kwargs.get("system")
        tools = sorted((_canon(t) for t in kwargs.get("tools") or []), key=lambda t: str(t.get("name")))
        head = {"system": _canon(_blocks(system) if system else []), "tools": tools}
        h = hashlib.sha256(json.dumps(head, sort_keys=True, ensure_ascii=False, default=str).encode())
        keys = [h.hexdigest()]
        for m in kwargs["messages"]:
            blocks = [_canon(b) for b in _blocks(m.get("content")) if _btype(b) not in _THINKING_TYPES]
            h.update(b"\x1e" + json.dumps({"role": m.get("role"), "content": blocks}, sort_keys=True,
                                          ensure_ascii=False, default=str).encode())
            keys.append(h.hexdigest())
        return keys

    def _check_bindings(self, messages, keys):
        """回传的每个（由本 MockLLM 产生的）thinking 块：它之前的对话必须与产生它时完全相同；thinking 块
        只能从最前面开始删，不能从中间删（每个块记着它前面的那一个）。返回请求里最后一个 thinking 块。"""
        prev = None
        for i, m in enumerate(messages):
            if m.get("role") != "assistant":
                continue
            for j, b in enumerate(_blocks(m.get("content"))):
                if _btype(b) not in _THINKING_TYPES:
                    continue
                sig = _bget(b, "signature")
                rec = self._bindings.get(sig)
                if rec is not None:
                    key, rec_prev, text = rec
                    if (_bget(b, "thinking", "") or "") != text:
                        raise MockBadRequestError(
                            f"400 invalid_request_error（模拟）: messages.{i}.content.{j}: Invalid `signature` in "
                            "`thinking` block. ——thinking 块的内容被改过；请原样回传。")
                    if key != keys[i] or (prev is not None and prev != rec_prev):
                        raise MockBadRequestError(_BOUND_MSG.format(i=i, j=j))
                prev = sig
        return prev

    # ----- 构造响应 -----
    def _new_id(self, prefix):
        self._ids += 1
        return f"{prefix}_mock_{self._ids:04d}"

    def _respond(self, kwargs, texts=(), calls=(), *, stop_reason=None, thought="", meta=None,
                 thinking=True, prewarm=False):
        content, sig = [], None
        if thinking and not prewarm:
            sig = self._new_id("sig")
            content.append(ThinkingBlock(type="thinking", thinking=thought if _display_on(kwargs) else "",
                                         signature=sig))
        if not prewarm:
            for t in texts:
                if t:
                    content.append(TextBlock(type="text", text=t, citations=None))
            for c in calls:
                content.append(ToolUseBlock(type="tool_use", id=self._new_id("toolu"), name=c["name"],
                                            input=copy.deepcopy(c["input"])))
        if stop_reason is None:
            stop_reason = "tool_use" if calls else "end_turn"
        details = (StopDetails(type="refusal", category=None,
                               explanation="(MockLLM) Simulated refusal for testing the refusal branch.")
                   if stop_reason == "refusal" else None)
        in_tok = estimate_tokens({"system": kwargs.get("system"), "messages": kwargs["messages"],
                                  "tools": kwargs.get("tools")})
        if kwargs.get("tools"):
            in_tok += _TOOL_OVERHEAD
        out_tok = 0 if prewarm else max(1, estimate_tokens(content))
        resp = MockMessage(id=self._new_id("msg"), type="message", role="assistant",
                           model=kwargs["model"], content=content, stop_reason=stop_reason,
                           stop_sequence=None, stop_details=details,
                           usage=Usage(input_tokens=in_tok, output_tokens=out_tok,
                                       cache_creation_input_tokens=0, cache_read_input_tokens=0))
        if sig is not None:      # 记下这个 thinking 块属于哪段对话（下次回传时核对）
            self._bindings[sig] = (self._cur["keys"][-1], self._cur["last_sig"], content[0].thinking)
        self.n_calls += 1
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        entry = {"stop_reason": stop_reason, "input_tokens": in_tok, "output_tokens": out_tok}
        entry.update(meta or {})
        self.log.append(entry)
        return resp

    def _refuse(self, kwargs, meta):
        """拒答：HTTP 200、stop_reason "refusal"；content 里可能只有中断前的半句话（不是答案）。"""
        return self._respond(kwargs, texts=[_REFUSAL_PARTIAL], stop_reason="refusal", thinking=False, meta=meta)

    def _truncate(self, kwargs, text, meta):
        """被 max_tokens 截断：stop_reason "max_tokens"，文字停在半句。"""
        partial = text[: max(8, len(text) // 2)].rstrip()
        return self._respond(kwargs, texts=[partial], stop_reason="max_tokens",
                             thought="Working out the first step.", meta=meta)

    def _find_item(self, messages):
        first = re.sub(r"\s+", " ", _text_of(messages[0]["content"]))
        for it in self._items.values():
            if re.sub(r"\s+", " ", it["question"]) in first:
                return it
        return None

    @staticmethod
    def _evidence_text(messages, item):
        """除题面以外，对话里 user 一侧出现过的全部文字（工具结果、workflow 的笔记）。"""
        q = re.sub(r"\s+", " ", item["question"])
        parts = []
        for m in messages:
            if m["role"] == "user":
                parts.append(re.sub(r"\s+", " ", _text_of(m["content"])).replace(q, " "))
        return "\n".join(parts)

    @staticmethod
    def _last_result(messages):
        for m in reversed(messages):
            if m["role"] == "user":
                for b in reversed(_blocks(m["content"])):
                    if _btype(b) == "tool_result":
                        c = _bget(b, "content", "")
                        return (c if isinstance(c, str) else _text_of(c)).strip()
        return ""

    def _final_text(self, item, mock, messages, text):
        """标准答案的最后一步：如果工具结果里找不到应有的证据，就像真实模型一样只能瞎猜。"""
        ev = self._evidence_text(messages, item)
        missing = [r for r in _requirements(item) if not _contains(ev, r)]
        if missing:
            return ("I could not confirm this from the tool results, so this is a guess.\n"
                    f"Final answer: {mock['guess']}"), missing
        return text, []

    # ----- 主入口 -----
    def create(self, **kwargs):
        """与 ``client.messages.create(...)`` 同签名（只接受关键字参数）；返回 MockMessage。"""
        turns, keys, last_sig = self._validate(kwargs)
        self._cur = {"keys": keys, "last_sig": last_sig}
        messages = kwargs["messages"]
        system = kwargs.get("system") or ""
        if isinstance(system, list):
            system = " ".join(str(_bget(b, "text", "")) for b in system)
        step_name = next((k for k, p in WORKFLOW_PROMPTS.items() if p in system), None)
        mode = "workflow" if step_name else ("agent" if kwargs.get("tools") else "direct")
        if kwargs["max_tokens"] == 0:                      # 缓存预热：只做 prefill，不生成
            return self._respond(kwargs, stop_reason="max_tokens", prewarm=True,
                                 meta={"qid": None, "mode": mode, "step": "prewarm"})
        item = self._find_item(messages)
        if item is None:
            return self._respond(kwargs, texts=[_UNKNOWN_REPLY], meta={"qid": None, "mode": mode})
        mock = item["mock"]
        meta = {"qid": item["id"], "mode": mode}
        if mode == "workflow":
            return self._workflow_step(kwargs, item, mock, step_name, meta)
        if mode == "direct":
            meta["step"] = "direct"
            return self._respond(kwargs, texts=[mock["direct"]], thought="Answering directly, without tools.",
                                 meta=meta)
        return self._agent_step(kwargs, item, mock, meta, sum(1 for role, _ in turns if role == "assistant"))

    def _agent_step(self, kwargs, item, mock, meta, k):
        messages = kwargs["messages"]
        meta["step"] = k
        if k >= _MAX_CALLS_PER_CONVERSATION:
            raise MockLoopGuardError(
                f"同一段对话已经调用了 {k} 次 MockLLM——你的 Agent 循环多半没有最大步数保护（max_steps）。")
        fault = self._fault(item["id"], "agent")
        gold = mock["agent"]
        tool_choice = kwargs.get("tool_choice") or {}
        if fault == "refusal":
            return self._refuse(kwargs, meta)
        if tool_choice.get("type") == "none":              # 不允许调用工具：只能不查资料直接作答
            return self._respond(kwargs, texts=[mock["direct"]],
                                 thought="Tools are turned off for this request, so I answer without them.",
                                 meta=meta)
        if fault == "truncated":
            return self._truncate(kwargs, gold[0].get("text") or "Let me work through this step by step.", meta)
        if fault == "loop":
            first = gold[0]["tool_calls"][0]
            return self._respond(kwargs, texts=["Let me check that again."], calls=[first],
                                 thought=_thought_for([first]), meta=meta)
        if fault == "early_stop":
            traj = [{"text": "I am fairly sure I know this without using any tools.\n"
                             f"Final answer: {mock['guess']}"}]
        elif fault == "wrong_tool":
            if item["type"] == "arithmetic":
                call = {"name": "wiki_search", "input": {"query": " ".join(item["question"].split()[:8])}}
            else:
                call = {"name": "calculator", "input": {"expression": gold[0]["tool_calls"][0]["input"]
                                                        .get("query", item["question"][:40])}}
            traj = [{"text": "Let me use a tool for this.", "tool_calls": [call]},
                    {"text": f"The tool did not settle it, so I will go with my best guess.\n"
                             f"Final answer: {mock['guess']}"}]
        elif fault == "bad_args":
            spec = mock["bad_args"]
            first = "Let me compute this." if spec["call"]["name"] == "calculator" else "Let me look this up."
            traj = [{"text": first, "tool_calls": [spec["call"]]},
                    {"text": spec["final"], "raw": True}]
        elif fault == "hallucination":
            traj = gold[:-1] + [{"text": f"Based on what I found, the answer is {mock['guess']}.\n"
                                         f"Final answer: {mock['guess']}", "raw": True}]
        elif fault == "format_mismatch":
            traj = gold[:-1] + [{"text": mock["format_variant"], "raw": True}]
        else:
            traj = gold
        if tool_choice.get("disable_parallel_tool_use"):
            traj = _one_call_per_step(traj)
        step = traj[min(k, len(traj) - 1)]
        calls = step.get("tool_calls", [])
        if calls:
            return self._respond(kwargs, texts=[step.get("text", "")], calls=calls, thought=_thought_for(calls),
                                 meta=meta)
        text = step["text"]
        if "{last}" in text:
            text = text.replace("{last}", self._last_result(messages))
        if not step.get("raw") and fault is None:
            text, missing = self._final_text(item, mock, messages, text)
            if missing:
                meta["missing_evidence"] = missing
        return self._respond(kwargs, texts=[text], thought=_thought_for(()), meta=meta)

    def _workflow_step(self, kwargs, item, mock, step_name, meta):
        messages = kwargs["messages"]
        wf = mock["workflow"]
        fault = self._fault(item["id"], "workflow")
        meta["step"] = step_name
        thought = f"Following the instructions for the '{step_name}' step."
        if fault == "refusal":
            return self._refuse(kwargs, meta)
        if fault == "truncated":
            return self._truncate(kwargs, "Let me think about what this step needs.", meta)
        lookup = item["type"] == "lookup"
        if step_name == "route":
            route = wf["route"]
            if fault == "wrong_tool":
                route = "ARITHMETIC" if route == "LOOKUP" else "LOOKUP"
            return self._respond(kwargs, texts=[route], thought=thought, meta=meta)
        if step_name == "query":
            if not lookup:
                queries = [" ".join(item["question"].split()[:8])]
            elif fault == "bad_args":
                queries = [mock["bad_args"]["call"]["input"]["query"]]
            else:
                queries = wf["queries"]
            return self._respond(kwargs, texts=["\n".join(queries)], thought=thought, meta=meta)
        if step_name == "expression":
            if fault == "bad_args":
                expr = mock["bad_args"]["call"]["input"]["expression"] if not lookup else "NONE"
            elif fault == "wrong_tool" and lookup:
                expr = "NONE"
            else:
                expr = wf["expression"]
            return self._respond(kwargs, texts=[expr], thought=thought, meta=meta)
        # answer
        if fault == "bad_args" and not lookup:
            expr = mock["bad_args"]["call"]["input"]["expression"]
            text = mock["bad_args"]["final"].replace("{last}", calculator(expr))
        elif fault == "hallucination":
            text = f"According to the notes the answer is {mock['guess']}.\nFinal answer: {mock['guess']}"
        elif fault == "format_mismatch":
            text = mock["format_variant"]
        else:
            text, missing = self._final_text(item, mock, messages, wf["answer"])
            if missing:
                meta["missing_evidence"] = missing
        return self._respond(kwargs, texts=[text], thought=thought, meta=meta)


# ---------------------------------------------------------------------------
# 后端选择
# ---------------------------------------------------------------------------


def _load_colab_secret():
    """在 Colab 上把 Secrets 里的 ANTHROPIC_API_KEY 读进环境变量（不打印、不返回密钥）。"""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        from google.colab import userdata      # noqa: PLC0415 （只在 Colab 上存在）
        key = userdata.get("ANTHROPIC_API_KEY")
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
    except Exception:                          # noqa: BLE001 —— 不在 Colab、没有该 Secret 或未授权
        pass


def make_backend(prefer="claude", mock_faults="demo", verbose=True):
    """返回一个有 ``.create(...)`` 方法的后端。

    - prefer="claude"、环境变量（或 Colab Secrets）里有 ANTHROPIC_API_KEY、且装了 anthropic →
      返回真实的 ``anthropic.Anthropic().messages``（每次调用都会计费）；
    - 否则（或 prefer="mock"、或环境变量 Z2R_LLM_BACKEND=mock）→ 返回 ``MockLLM(faults=mock_faults)``，
      默认 "demo"：一个按剧本作答、在 8 处预先注入了典型错误的模拟模型（讲义演示用）；
      作业用 ``mock_faults="homework"``（另一份故障分布）。CI 永远走这一支。

    密钥只从环境变量 / Colab Secrets 读取，本函数从不打印或返回它。
    """
    force_mock = prefer == "mock" or os.environ.get("Z2R_LLM_BACKEND", "").lower() == "mock"
    if not force_mock:
        _load_colab_secret()
        if os.environ.get("ANTHROPIC_API_KEY") and importlib.util.find_spec("anthropic") is not None:
            import anthropic                     # noqa: PLC0415
            if verbose:
                print(f"后端：真实 Claude API（模型 {MODEL}）——每次调用都会计费")
            return anthropic.Anthropic().messages
    if verbose:
        why = "已指定使用 mock" if force_mock else "未检测到 ANTHROPIC_API_KEY"
        extra = {"demo": "，模拟一个偶尔犯错的模型（讲义演示用的 agents.DEMO_FAULTS）",
                 "homework": "，模拟一个偶尔犯错的模型（作业用的故障分布，与讲义演示不同）"}
        extra = extra.get(mock_faults, "") if isinstance(mock_faults, str) else ""
        print(f"后端：MockLLM（{why}{extra}；它的准确率是剧本写的，不代表真实模型）")
    return MockLLM(faults=mock_faults)
