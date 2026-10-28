"""Predictions, scores, verdicts and figures from cached results."""
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from protex import evaluate, llm, pdftext, retrieve  # noqa: E402
from protex.schema import FIELDS, NOT_STATED  # noqa: E402

log = logging.getLogger("protex.report")
LLM_METHODS = ["full-opus", "rag-opus", "full-haiku", "rag-haiku"]


def llm_predictions(role, data, results):
    use = pd.read_csv(results / f"parse_{role}.csv")
    use = use[use.has_text.fillna(False).astype(bool)].nct_id
    rows, usage = [], []
    for m in LLM_METHODS:
        kind, model = m.split("-")
        for nct in use:
            f = data / "responses" / f"{m}-{nct}.json"
            if not f.exists():
                continue
            ext, u, status = llm.read(json.loads(f.read_text(encoding="utf-8")))
            if u:
                tin = u["input_tokens"] + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
                usage.append({"method": m, "nct_id": nct, "status": status, "input_tokens": tin,
                              "output_tokens": u["output_tokens"], "usd": llm.cost(model, tin, u["output_tokens"])})
            else:
                usage.append({"method": m, "nct_id": nct, "status": status})
            for fld in FIELDS:
                item = getattr(ext, fld) if ext else None
                rows.append({"nct_id": nct, "method": m, "field": fld,
                             "value": item.value if item else None, "quote": item.quote if item else ""})
    return pd.DataFrame(rows), pd.DataFrame(usage)


def add_grounding(pred, data):
    texts = {}
    g_strict, g_loose = [], []
    for r in pred.itertuples():
        if r.method == "rules" or r.value in (None, NOT_STATED) or not r.quote:
            g_strict.append(None)
            g_loose.append(None)
            continue
        if r.nct_id not in texts:
            pg = json.loads((data / "text" / f"{r.nct_id}.json").read_text(encoding="utf-8"))
            texts[r.nct_id] = " ".join(pg)
        g_strict.append(evaluate.grounded(r.quote, texts[r.nct_id], strict=True))
        g_loose.append(evaluate.grounded(r.quote, texts[r.nct_id], strict=False))
    return pred.assign(grounded=g_strict, grounded_loose=g_loose)


def run(role, data, results):
    data, results = Path(data), Path(results)
    parse = pd.read_csv(results / f"parse_{role}.csv")
    key = pd.read_csv(results / "answer_key.csv", dtype=str)
    rules_pred = pd.read_csv(results / f"rules_{role}.csv", dtype=str)
    lp, usage = llm_predictions(role, data, results)
    pred = add_grounding(pd.concat([rules_pred, lp], ignore_index=True), data)
    pred.to_csv(results / f"predictions_{role}.csv", index=False)
    usage.to_csv(results / f"usage_{role}.csv", index=False)

    sc = evaluate.score(pred, key)
    acc = evaluate.accuracy(sc)
    acc.to_csv(results / f"accuracy_{role}.csv", index=False)
    doc = evaluate.per_doc(sc)
    methods = [m for m in LLM_METHODS if m in doc]
    contrasts = [(m, "rules") for m in methods] + [
        (a, b) for a, b in [("rag-opus", "full-opus"), ("rag-haiku", "rag-opus"), ("full-haiku", "full-opus"),
                            ("rag-haiku", "full-haiku")] if a in doc and b in doc]
    pair = pd.DataFrame([evaluate.paired(doc, a, b) for a, b in contrasts])
    pair.to_csv(results / f"paired_{role}.csv", index=False)
    mc = pd.DataFrame([evaluate.mcnemar(sc, a, b, f) for a, b in [("rag-opus", "full-opus"), ("rag-haiku", "rag-opus")]
                       if a in doc and b in doc for f in FIELDS])
    mc.to_csv(results / f"mcnemar_{role}.csv", index=False)

    overall = doc.mean().rename("mean_accuracy").to_frame()
    if len(usage) and "usd" in usage:
        u = usage.groupby("method").agg(input_tokens=("input_tokens", "median"), usd_per_protocol=("usd", "mean"),
                                        failures=("status", lambda s: int((s != "ok").sum())))
        overall = overall.join(u)
    gr = pred[pred.grounded.notna()].groupby("method")[["grounded", "grounded_loose"]].mean()
    overall.join(gr).to_csv(results / f"summary_{role}.csv")

    w = pred[pred.method.isin(methods)].copy()
    w["v"] = [evaluate.norm_value(f, v) for f, v in zip(w.field, w.value)]
    agree = w.pivot_table(index=["nct_id", "field"], columns="method", values="v", aggfunc="first")
    same = agree[agree.nunique(axis=1) == 1].iloc[:, 0].rename("llm_value").reset_index() if len(agree) else pd.DataFrame()
    if len(same):
        k = key.melt(id_vars="nct_id", value_vars=FIELDS, var_name="field", value_name="registry")
        dis = same.merge(k, on=["nct_id", "field"])
        dis = dis[[evaluate.correct(f, p, g) is False for f, p, g in zip(dis.field, dis.llm_value, dis.registry)]]
        dis.to_csv(results / f"discrepancies_{role}.csv", index=False)

    verdicts(role, parse, doc, pair, usage, pred, results)
    figures(acc, overall, results, role)


def verdicts(role, parse, doc, pair, usage, pred, results):
    p = pair.set_index("contrast")
    v = []
    share = parse.has_text.fillna(False).astype(bool).mean()
    v.append(("P1", "protocols with extractable text (>= 90%)", f"{share:.1%}", share >= 0.9))
    llm_vs_rules = p[p.index.str.endswith("- rules")]
    v.append(("P2", "every LLM method beats rules (lower bound > 0)",
              "; ".join(f"{i.split(' - ')[0]} {r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" for i, r in llm_vs_rules.iterrows()),
              bool(len(llm_vs_rules)) and bool((llm_vs_rules.lo > 0).all())))
    if "rag-opus - full-opus" in p.index:
        r = p.loc["rag-opus - full-opus"]
        v.append(("P3", "RAG vs full text, Opus 5.5 (lower bound > -0.03)",
                  f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", r.lo > -0.03))
    if len(usage) and {"full-opus", "rag-opus"} <= set(usage.method):
        t = usage.pivot_table(index="nct_id", columns="method", values="input_tokens")
        ratio = (t["rag-opus"] / t["full-opus"]).median()
        v.append(("P4", "RAG input tokens / full text, Opus 5.5 (median <= 25%)", f"{ratio:.1%}", ratio <= 0.25))
    g = pred[(pred.method == "rag-opus") & pred.grounded.notna()].grounded
    if len(g):
        v.append(("P5", "quotes found in the text, RAG Opus 5.5 (>= 95%)", f"{g.astype(bool).mean():.1%}",
                  g.astype(bool).mean() >= 0.95))
    if "rag-haiku - rag-opus" in p.index:
        r = p.loc["rag-haiku - rag-opus"]
        v.append(("P6", "Haiku 5.5 vs Opus 5.5, RAG (lower bound > -0.05)",
                  f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", r.lo > -0.05))
    out = pd.DataFrame(v, columns=["id", "check", "value", "met"])
    out.to_csv(results / f"verdicts_{role}.csv", index=False)
    log.info("\n%s", out.to_string(index=False))


def figures(acc, overall, results, role):
    fig_dir = results.parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    order = ["rules"] + LLM_METHODS
    colors = {"rules": "#888888", "full-opus": "#1f4e79", "rag-opus": "#5b9bd5", "full-haiku": "#7f3f00",
              "rag-haiku": "#f4a261"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, m in enumerate([m for m in order if m in set(acc.method)]):
        a = acc[acc.method == m].set_index("field").reindex(FIELDS)
        y = [j + (i - 2) * 0.15 for j in range(len(FIELDS))]
        ax.errorbar(a.accuracy, y, xerr=[a.accuracy - a.lo, a.hi - a.accuracy], fmt="o", ms=4,
                    color=colors[m], label=m, capsize=2)
    ax.set_yticks(range(len(FIELDS)), [f.replace("_", " ") for f in FIELDS])
    ax.invert_yaxis()
    ax.set_xlabel("agreement with the registry (95% Wilson interval)")
    ax.set_xlim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"accuracy_{role}.png", dpi=150)
    plt.close(fig)
    if "usd_per_protocol" in overall:
        o = overall.dropna(subset=["usd_per_protocol"])
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for m, r in o.iterrows():
            ax.scatter(r.usd_per_protocol, r.mean_accuracy, color=colors.get(m, "k"), s=40)
            ax.annotate(m, (r.usd_per_protocol, r.mean_accuracy), xytext=(5, 3), textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("cost per protocol, USD (batch prices, log scale)")
        ax.set_ylabel("mean accuracy over 7 fields")
        fig.tight_layout()
        fig.savefig(fig_dir / f"cost_accuracy_{role}.png", dpi=150)
        plt.close(fig)
