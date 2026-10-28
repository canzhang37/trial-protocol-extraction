"""Part 2 scores and checks, next to the part 1 methods."""
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from protex import evaluate, llm, report  # noqa: E402
from protex.schema import FIELDS, NOT_STATED  # noqa: E402

log = logging.getLogger("protex.agent_report")
AGENT_METHODS = ["agent-haiku", "agent-opus"]


def load(data, ids):
    rows, usage = [], []
    for m in AGENT_METHODS:
        model = m.split("-")[1]
        for nct in ids:
            f = data / "agent" / f"{m}-{nct}.json"
            if not f.exists():
                continue
            rec = json.loads(f.read_text(encoding="utf-8"))
            ext, u, status = llm.read(rec["extraction"])
            tin = sum(x.get("input_tokens", 0) for x in rec["agent_usage"])
            if u:
                tin += u["input_tokens"]
            usage.append({"method": m, "nct_id": nct, "status": status, "agent_status": rec["agent_status"],
                          "tool_calls": len(rec["calls"]), "hit_limit": rec["hit_limit"],
                          "searches": sum(c["tool"] == "search" for c in rec["calls"]),
                          "pages_read": sum(c["tool"] == "read_page" for c in rec["calls"]),
                          "context_chars": rec["context_chars"], "input_tokens": tin,
                          "usd": rec["agent_usd"] + rec["extract_usd"],
                          "refusals": sum(s == "refusal" for s in rec["agent_stop"])})
            for fld in FIELDS:
                item = getattr(ext, fld) if ext else None
                rows.append({"nct_id": nct, "method": m, "field": fld,
                             "value": item.value if item else None, "quote": item.quote if item else ""})
    return pd.DataFrame(rows), pd.DataFrame(usage)


def run(data, results):
    data, results = Path(data), Path(results)
    parse = pd.read_csv(results / "parse_evaluation.csv")
    ids = parse[parse.has_text.fillna(False).astype(bool)].nct_id.tolist()
    key = pd.read_csv(results / "answer_key.csv", dtype=str)
    part1 = pd.read_csv(results / "predictions_evaluation.csv", dtype={"value": str})
    ap, usage = load(data, ids)
    ap = report.add_grounding(ap, data)
    ap.to_csv(results / "agent_predictions.csv", index=False)
    usage.to_csv(results / "agent_usage.csv", index=False)
    pred = pd.concat([part1, ap], ignore_index=True)

    sc = evaluate.score(pred, key)
    acc = evaluate.accuracy(sc)
    acc.to_csv(results / "agent_accuracy.csv", index=False)
    doc = evaluate.per_doc(sc)
    pairs = [("agent-haiku", "rag-haiku"), ("agent-haiku", "full-haiku"), ("agent-opus", "rag-opus"),
             ("agent-opus", "full-opus"), ("agent-opus", "agent-haiku")]
    pair = pd.DataFrame([evaluate.paired(doc, a, b) for a, b in pairs if a in doc and b in doc])
    pair.to_csv(results / "agent_paired.csv", index=False)
    mc = pd.DataFrame([evaluate.mcnemar(sc, a, b, f) for a, b in pairs[:3:2] if a in doc and b in doc
                       for f in FIELDS])
    mc.to_csv(results / "agent_mcnemar.csv", index=False)

    llm_pred = pred[pred.method != "rules"]
    ns = llm_pred.assign(ns=llm_pred.value.eq(NOT_STATED)).groupby(["method", "field"]).ns.mean().unstack()
    ns.round(3).to_csv(results / "agent_not_stated.csv")

    sub = doc[doc["agent-opus"].notna()].index if "agent-opus" in doc else []
    summary = pd.DataFrame({"mean_all": doc.mean(), "mean_opus_subset": doc.loc[sub].mean() if len(sub) else None})
    if len(usage):
        u = usage.groupby("method").agg(protocols=("nct_id", "size"), median_input_tokens=("input_tokens", "median"),
                                        usd_per_protocol=("usd", "mean"), usd_total=("usd", "sum"),
                                        median_tool_calls=("tool_calls", "median"),
                                        hit_limit=("hit_limit", "mean"),
                                        failures=("status", lambda s: int((s != "ok").sum())))
        summary = summary.join(u)
    gr = pred[pred.grounded.notna()].groupby("method")[["grounded", "grounded_loose"]].mean()
    summary.join(gr).to_csv(results / "agent_summary.csv")

    p = pair.set_index("contrast") if len(pair) else pd.DataFrame()
    v = []

    def lb(contrast, limit, label, cid):
        if contrast in p.index:
            r = p.loc[contrast]
            v.append((cid, label, f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", r.lo > limit))

    lb("agent-haiku - rag-haiku", 0, "agent vs fixed RAG, Haiku (lower bound > 0)", "K1")
    lb("agent-haiku - full-haiku", -0.03, "agent vs full text, Haiku (lower bound > -0.03)", "K2")
    full = pd.read_csv(results / "usage_evaluation.csv")
    full = full[full.method == "full-haiku"].set_index("nct_id").input_tokens
    ah = usage[usage.method == "agent-haiku"].set_index("nct_id").input_tokens if len(usage) else pd.Series(dtype=float)
    if len(ah):
        ratio = (ah / full.reindex(ah.index)).median()
        v.append(("K3", "input tokens, agent-haiku / full-haiku (median <= 50%)", f"{ratio:.1%}", ratio <= 0.5))
    g = ap[(ap.method == "agent-haiku") & ap.grounded.notna()].grounded
    if len(g):
        v.append(("K4", "quotes found in the text, agent-haiku (>= 95%)", f"{g.astype(bool).mean():.1%}",
                  g.astype(bool).mean() >= 0.95))
    lb("agent-opus - rag-opus", 0, "agent vs fixed RAG, Opus, all 148 (lower bound > 0)", "K5")
    if {"agent-opus", "rag-opus"} <= set(doc):
        r = evaluate.paired(doc.loc[[n for n in ids[:50] if n in doc.index]], "agent-opus", "rag-opus")
        v.append(("K5 (first 50)", "the same on the first 50, as first planned",
                  f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]", r["lo"] > 0))
    out = pd.DataFrame(v, columns=["id", "check", "value", "met"])
    out.to_csv(results / "agent_verdicts.csv", index=False)
    log.info("\n%s", out.to_string(index=False))
    figure(acc, results)


def figure(acc, results):
    order = ["rules", "full-haiku", "rag-haiku", "agent-haiku", "full-opus", "rag-opus", "agent-opus"]
    colors = {"rules": "#888888", "full-haiku": "#7f3f00", "rag-haiku": "#f4a261", "agent-haiku": "#e76f51",
              "full-opus": "#1f4e79", "rag-opus": "#5b9bd5", "agent-opus": "#2a9d8f"}
    ms = [m for m in order if m in set(acc.method)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, m in enumerate(ms):
        a = acc[acc.method == m].set_index("field").reindex(FIELDS)
        y = [j + (i - (len(ms) - 1) / 2) * 0.11 for j in range(len(FIELDS))]
        ax.errorbar(a.accuracy, y, xerr=[a.accuracy - a.lo, a.hi - a.accuracy], fmt="o", ms=3.5,
                    color=colors[m], label=m, capsize=2)
    ax.set_yticks(range(len(FIELDS)), [f.replace("_", " ") for f in FIELDS])
    ax.invert_yaxis()
    ax.set_xlabel("agreement with the registry (95% Wilson interval)")
    ax.set_xlim(0, 1.02)
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(results.parent / "figures" / "agent_accuracy.png", dpi=150)
    plt.close(fig)
