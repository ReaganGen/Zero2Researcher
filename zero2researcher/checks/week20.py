"""第20周自查函数。用法：from zero2researcher.checks import week20 as ck; ck.final_checklist(deliverables)

final_checklist 只检查交付物清单"填齐了没有、有没有明显不合格"（页数、种子数、是否一键跑通），
不评判研究质量——质量由汇报、同伴评审与教师评分决定。
"""
import numbers
import re

from . import report

FIELDS = {
    "title": "报告题目",
    "report_file": "4 页报告（PDF）的文件名或链接",
    "report_pages": "正文页数（不含参考文献与附录），1–4 的整数",
    "notebook_link": "可一键复现的 notebook 链接或仓库路径",
    "restart_and_run_all": "是否在全新运行时里'重启并运行全部'跑通过（True/False）",
    "runtime_minutes": "一键运行的总耗时（分钟）",
    "hardware": "运行环境，如 'Colab T4 GPU' 或 'Colab CPU'",
    "seeds": "主实验用的随机种子列表（至少 3 个）",
    "lab_notebook_link": "实验记录本链接",
    "lab_entries": "实验记录条数",
    "slides_file": "汇报幻灯片（PDF）的文件名或链接",
    "baseline": "基线是什么（一句话）",
    "ablations": "做了哪些消融（列表，每项一句话）",
    "ai_use_statement": "报告里是否有 AI 助手使用声明（True/False）",
    "limitations": "最重要的一条局限（一句话）",
}
MIN_LAB_ENTRIES = 5
_PLACEHOLDERS = ("在这里写", "todo", "tbd", "待填", "xxx")


def _blank(s):
    return not isinstance(s, str) or not s.strip() or s.strip().lower() in _PLACEHOLDERS


def _is_int(x):
    """整数（含 NumPy 整数，如 np.arange 生成的种子），但不含 True/False。"""
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def final_checklist(d):
    """检查终期交付物清单 dict（字段见 FIELDS）是否齐全、合格。"""
    name = "Capstone 终期交付物清单"
    if not isinstance(d, dict):
        return report(name, False, f"应传入一个 dict，实际是 {type(d).__name__}")
    missing = [k for k in FIELDS if k not in d]
    if missing:
        return report(name, False, f"缺少字段 {missing}", "；".join(f"{k}：{FIELDS[k]}" for k in missing))
    for k in ("title", "report_file", "notebook_link", "hardware", "lab_notebook_link", "slides_file",
              "baseline", "limitations"):
        if _blank(d[k]):
            return report(name, False, f"{k} 还没有填写", FIELDS[k])
    pages = d["report_pages"]
    if not _is_int(pages) or not 1 <= pages <= 4:
        return report(name, False, f"report_pages 应是 1–4 的整数，实际 {pages!r}",
                      "正文限 4 页（参考文献、附录与 AI 使用声明不计）；超出的内容移到附录")
    if d["restart_and_run_all"] is not True:
        return report(name, False, "restart_and_run_all 不是 True：notebook 还没有在全新运行时里一键跑通",
                      "Colab：'运行时 → 断开连接并删除运行时'，再'运行全部'；报告里的每个数字都应能在输出里找到")
    rt = d["runtime_minutes"]
    if isinstance(rt, bool) or not isinstance(rt, (int, float)) or not rt > 0:
        return report(name, False, f"runtime_minutes 应是正数，实际 {rt!r}", "记录一键运行的实际耗时，写进报告的 Reproducibility 一节")
    seeds = d["seeds"]
    if (not isinstance(seeds, (list, tuple)) or not all(_is_int(s) for s in seeds)
            or len({int(s) for s in seeds}) < 3):
        return report(name, False, f"seeds 应是至少 3 个不同整数的列表，实际 {seeds!r}",
                      "课程的最小实验单元：≥3 个种子，报告均值 ± 标准差")
    n = d["lab_entries"]
    if not _is_int(n) or n < MIN_LAB_ENTRIES:
        return report(name, False, f"lab_entries 应是 ≥ {MIN_LAB_ENTRIES} 的整数，实际 {n!r}",
                      "从第 8 周选题起，基线、主结果、每个消融都应各有一条实验记录")
    abl = d["ablations"]
    if not isinstance(abl, (list, tuple)) or not abl or any(_blank(a) for a in abl):
        return report(name, False, "ablations 应是非空列表，每项一句话描述一个消融",
                      "例如 'remove data augmentation (3 seeds)'；一次只拿掉一个组件")
    if d["ai_use_statement"] is not True:
        return report(name, False, "报告里还没有 AI 助手使用声明",
                      "写明用了哪些工具、用在哪些部分、你如何核查它们的输出（模板第 7 节）")
    return report(name, True)


def _count_words(text):
    return len(re.findall(r"[㐀-䶿一-鿿]", text)) + len(re.findall(r"[A-Za-z]+(?:'[a-z]+)?", text))


def reflection(text, min_words=300):
    """个人反思：非空、不是模板原样，长度至少 min_words（汉字按字计、英文按词计）。"""
    name = "个人反思"
    if not isinstance(text, str) or not text.strip():
        return report(name, False, "反思还是空的", "按模板的问题写，至少 300 字")
    if any(p in text for p in ("（在这里写）", "(在这里写)")):
        return report(name, False, "还留着模板里的 '（在这里写）'", "把占位符换成你自己的文字")
    n = _count_words(text)
    return report(name, n >= min_words, f"目前约 {n} 字（词），至少需要 {min_words}",
                  "具体一点：写一个你卡住过的地方、一个改变了你想法的实验结果、下一步打算学什么")
