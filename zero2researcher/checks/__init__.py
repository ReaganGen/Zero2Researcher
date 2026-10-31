"""Beginner-friendly self-checks.

Each check prints a short Chinese message (pass / fail + hint) instead of a long
traceback. Set the environment variable Z2R_STRICT=1 (as CI does) to make a
failed check raise an AssertionError, so solution notebooks fail loudly.
"""
import os
import math

STRICT = os.environ.get("Z2R_STRICT") == "1"


class CheckFailed(AssertionError):
    pass


def report(name, ok, fail_msg="", hint=None):
    if ok:
        print(f"\u2705 {name}\uff1a\u901a\u8fc7")
        return True
    msg = f"\u274c {name}\uff1a{fail_msg}"
    if hint:
        msg += f"\n   \u63d0\u793a\uff1a{hint}"
    print(msg)
    if STRICT:
        raise CheckFailed(msg)
    return False


def close(a, b, tol=1e-6):
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def safe_call(name, fn, *args):
    """Call a student function; turn any exception into a friendly failure."""
    try:
        return True, fn(*args)
    except NotImplementedError:
        report(name, False, "\u51fd\u6570\u8fd8\u6ca1\u6709\u5b9e\u73b0\uff08\u4ecd\u662f NotImplementedError\uff09")
        return False, None
    except Exception as e:  # noqa: BLE001
        report(name, False, f"\u8fd0\u884c\u65f6\u51fa\u9519 {type(e).__name__}: {e}",
               "\u5148\u8bfb\u62a5\u9519\u4fe1\u606f\u7684\u6700\u540e\u4e00\u884c\uff0c\u518d\u7528 print() \u68c0\u67e5\u4e2d\u95f4\u53d8\u91cf")
        return False, None
