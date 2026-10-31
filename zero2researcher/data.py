"""Data loaders. Students call these; they never need to read this file.

Week 1 loaders return plain Python lists/dicts (students have not met NumPy yet).
"""
from importlib import resources
import csv

SUBJECTS = ["math", "english", "physics"]


def path(name: str) -> str:
    """Return the file path of a bundled dataset, e.g. path("tinyshakespeare.txt")."""
    return str(resources.files("zero2researcher") / "datasets" / name)


def download_root() -> str:
    """Shared folder for datasets downloaded at run time (torchvision, PyG ...).

    Every notebook passes this as `root=`, so MNIST, CIFAR-10 etc. are downloaded
    once and reused by all weeks. Override with the environment variable Z2R_DATA.
    """
    import os
    root = os.path.expanduser(os.environ.get("Z2R_DATA", "~/.cache/zero2researcher"))
    os.makedirs(root, exist_ok=True)
    return root


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


# ---------- Week 3+ loaders (return pandas DataFrames) ----------

def load_titanic():
    """Titanic passengers (891 rows x 15 cols), the classic seaborn copy.

    Column of interest: `survived` (0/1). Known missing values: age (177),
    deck (688), embarked (2), embark_town (2). Source: the copy in the seaborn-data repository
    (which carries no licence file); the data originate from the Kaggle
    "Titanic" competition and widely redistributed historical passenger records.
    """
    import pandas as pd
    return pd.read_csv(path("titanic.csv"))


def load_penguins():
    """Palmer Penguins (344 rows x 7 cols), the classic seaborn copy.

    Source: Gorman et al. 2014 / palmerpenguins (CC0); a few missing values.
    """
    import pandas as pd
    return pd.read_csv(path("penguins.csv"))


def load_imdb(split="train"):
    """Fixed course subsample of the IMDB movie-review dataset (Maas et al. 2011).

    split="train" -> 5,000 reviews, split="test" -> 2,000 reviews; columns
    `text` and `label` (0 = negative, 1 = positive), balanced classes,
    deduplicated and shuffled with seed 2026 (see scripts/generate_data_week06.py).
    """
    import pandas as pd
    assert split in ("train", "test"), 'split must be "train" or "test"'
    return pd.read_csv(path(f"imdb_{split}.csv.gz"))


def load_shakespeare():
    """The full Tiny Shakespeare text as one string (1,115,394 characters, 65 distinct)."""
    with open(path("tinyshakespeare.txt")) as f:
        return f.read()


def load_baby_names():
    """List of 32,033 lowercase English first names (one per line in names.txt).

    Source: karpathy/makemore (MIT License), built from the US Social Security
    Administration's 2018 baby-name list. Not to be confused with load_names(),
    which returns the 30 students of the week-1 gradebook.
    """
    with open(path("names.txt")) as f:
        return [w for w in f.read().split("\n") if w]
