"""Data loaders. Students call these; they never need to read this file.

Week 1 loaders return plain Python lists/dicts (students have not met NumPy yet).
"""
from importlib import resources
import csv

SUBJECTS = ["math", "english", "physics"]


def path(name: str) -> str:
    """Return the file path of a bundled dataset, e.g. path("tinyshakespeare.txt")."""
    return str(resources.files("zero2researcher") / "datasets" / name)


def _gradebook_rows():
    with open(path("gradebook.csv"), newline="") as f:
        return list(csv.DictReader(f))


def load_names():
    """List of the 30 student names, in gradebook order."""
    return [r["name"] for r in _gradebook_rows()]


def load_scores():
    """Nested list: one row per student, each row [math, english, physics] (ints).

    scores[0] belongs to load_names()[0]. Some entries are data-entry errors.
    """
    return [[int(r[s]) for s in SUBJECTS] for r in _gradebook_rows()]


def load_gradebook():
    """Dict: student name -> {"math": int, "english": int, "physics": int}."""
    return {r["name"]: {s: int(r[s]) for s in SUBJECTS} for r in _gradebook_rows()}


def load_temperature():
    """List of 365 daily mean temperatures (deg C), day 1 = 1 January."""
    with open(path("temperature.csv"), newline="") as f:
        return [float(r["temp_c"]) for r in csv.DictReader(f)]
