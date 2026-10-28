"""Scoring against the registry."""
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from protex.schema import FIELDS, NOT_STATED


def norm_value(field, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    v = str(v).strip().lower()
    if field == "minimum_age":
        if v in ("none", NOT_STATED, ""):
            return "none"
        try:
            return f"{float(v):g}"
        except ValueError:
            return v
    if field == "number_of_arms":
        try:
            return str(int(float(v)))
        except ValueError:
            return v
    return v


def correct(field, pred, gold):
    g, p = norm_value(field, gold), norm_value(field, pred)
    if g is None:
        return None
    return p == g


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def score(pred, key):
    """pred in long form; key: one column per field."""
    k = key.melt(id_vars="nct_id", value_vars=FIELDS, var_name="field", value_name="gold")
    d = pred.merge(k, on=["nct_id", "field"], how="left")
    d["correct"] = [correct(f, p, g) for f, p, g in zip(d.field, d.value, d.gold)]
    return d[d.correct.notna()].assign(correct=lambda x: x.correct.astype(bool))


def accuracy(scored):
    g = scored.groupby(["method", "field"]).correct.agg(["sum", "count"]).reset_index()
    ci = [wilson(s, n) for s, n in zip(g["sum"], g["count"])]
    return g.assign(accuracy=g["sum"] / g["count"], lo=[c[0] for c in ci], hi=[c[1] for c in ci])


def per_doc(scored):
    return scored.groupby(["method", "nct_id"]).correct.mean().unstack("method")


def paired(doc, a, b, reps=2000, seed=2027):
    d = (doc[a] - doc[b]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (reps, len(d)), replace=True).mean(axis=1)
    return {"contrast": f"{a} - {b}", "n": len(d), "diff": d.mean(),
            "lo": np.quantile(boots, 0.025), "hi": np.quantile(boots, 0.975)}


def mcnemar(scored, a, b, field):
    w = scored[scored.field == field].pivot_table(index="nct_id", columns="method", values="correct").dropna()
    n10 = int(((w[a] == 1) & (w[b] == 0)).sum())
    n01 = int(((w[a] == 0) & (w[b] == 1)).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {"field": field, "contrast": f"{a} - {b}", "a_only": n10, "b_only": n01, "p": p}


def squash(s, strict=True):
    if strict:
        return re.sub(r"\s+", " ", s.lower()).strip()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def grounded(quote, text, strict=True):
    q = squash(quote, strict)
    return bool(q) and q in squash(text, strict)
