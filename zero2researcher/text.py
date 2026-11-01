"""Word-level text preprocessing for the IMDB weeks (10, 11, 13, 14).

课程不教但绕不开的部分：把影评切成词、建词表、转成等长的整数矩阵。
第12周的字符级分词（stoi / itos）由学习者自己写，不在这里。

Typical use:
    from zero2researcher import text
    Xtr, ytr, Xte, yte, vocab = text.imdb_arrays(max_len=256, max_vocab=20000)
"""
import re
from collections import Counter

import numpy as np

PAD, UNK = "<pad>", "<unk>"
_TOKEN = re.compile(r"[a-z0-9']+|[!?.,]")


def tokenize(s: str) -> list:
    """Lowercase, drop the '<br />' HTML line breaks IMDB reviews contain, split into words.

    Punctuation ! ? . , is kept as separate tokens; other symbols are dropped.
    """
    return _TOKEN.findall(s.lower().replace("<br />", " "))


def build_vocab(texts, max_size=20000, min_freq=2) -> dict:
    """Map the most frequent words to integer ids. id 0 = <pad>, id 1 = <unk>.

    Build it on the TRAINING texts only: a vocabulary fitted on the test set is
    the same leak as fitting TfidfVectorizer on all data (week 6).
    """
    counts = Counter(tok for t in texts for tok in tokenize(t))
    words = [w for w, c in counts.most_common() if c >= min_freq][: max_size - 2]
    return {PAD: 0, UNK: 1, **{w: i + 2 for i, w in enumerate(words)}}


def encode(texts, vocab, max_len=256) -> np.ndarray:
    """Turn texts into an int64 array of shape (len(texts), max_len).

    Unknown words become <unk>; long reviews are truncated to the first max_len
    tokens; short ones are right-padded with <pad> (= 0).
    """
    unk = vocab[UNK]
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(tok, unk) for tok in tokenize(t)][:max_len]
        out[i, : len(ids)] = ids
    return out


def imdb_arrays(max_len=256, max_vocab=20000, min_freq=2):
    """Course IMDB subsample as arrays: (X_train, y_train, X_test, y_test, vocab).

    X_* are int64 (N, max_len) token-id matrices, y_* int64 labels (0 = neg, 1 = pos);
    the vocabulary is built from the 5,000 training reviews only.
    """
    from . import data

    tr, te = data.load_imdb("train"), data.load_imdb("test")
    vocab = build_vocab(tr["text"], max_size=max_vocab, min_freq=min_freq)
    return (encode(tr["text"], vocab, max_len), tr["label"].to_numpy(np.int64),
            encode(te["text"], vocab, max_len), te["label"].to_numpy(np.int64), vocab)
