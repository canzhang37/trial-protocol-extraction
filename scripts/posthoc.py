"""Post hoc checks, made after the results."""
import json
from pathlib import Path

import pandas as pd

from protex.evaluate import paired, per_doc, score
from protex.schema import NOT_STATED

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# registry defaults for implicit fields
DEFAULTS = {"phase": "not applicable", "sex": "all", "blinding": "open label"}


def readable_share(nct):
    pages = json.loads((ROOT / "data" / "text" / f"{nct}.json").read_text(encoding="utf-8"))
    s = " ".join(pages)
    return sum(c.isalpha() or c.isspace() for c in s) / max(len(s), 1)


def main():
    pred = pd.read_csv(RES / "predictions_evaluation.csv")
    key = pd.read_csv(RES / "answer_key.csv")
    llm = pred.method != "rules"

    ns = pred[llm].assign(ns=pred.value.eq(NOT_STATED)).groupby(["method", "field"]).ns.mean().unstack()
    ns.round(3).to_csv(RES / "posthoc_not_stated.csv")

    conv = pred.copy()
    for f, v in DEFAULTS.items():
        m = llm & conv.field.eq(f) & conv.value.eq(NOT_STATED)
        conv.loc[m, "value"] = v

    text = pd.DataFrame({"nct_id": key.nct_id})
    text["readable"] = [readable_share(n) for n in text.nct_id]
    garbled = set(text.nct_id[text.readable < 0.6])

    rows = []
    for label, p in [("as planned", pred), ("defaults for not stated", conv),
                     ("defaults, garbled text dropped", conv[~conv.nct_id.isin(garbled)])]:
        doc = per_doc(score(p, key))
        for a, b in [("rag-opus", "full-opus"), ("rag-haiku", "rag-opus"), ("full-haiku", "full-opus")]:
            r = paired(doc, a, b)
            rows.append({"scoring": label, **r})
        for m in doc.columns:
            rows.append({"scoring": label, "contrast": f"mean {m}", "n": doc[m].notna().sum(),
                         "diff": doc[m].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "posthoc_scoring.csv", index=False)
    text.to_csv(RES / "posthoc_text_quality.csv", index=False)
    print(ns.round(3).to_string())
    print("garbled:", sorted(garbled))
    print(out.round(3).to_string(index=False))
    if (RES / "agent_predictions.csv").exists():
        agent_defaults(pred, key)


def agent_defaults(part1, key):
    # part 2 arms under the same registry defaults
    ap = pd.read_csv(RES / "agent_predictions.csv", dtype={"value": str})
    both = pd.concat([part1, ap], ignore_index=True)
    conv = both.copy()
    llm = conv.method != "rules"
    for f, v in DEFAULTS.items():
        conv.loc[llm & conv.field.eq(f) & conv.value.eq(NOT_STATED), "value"] = v
    rows = []
    for label, p in [("as planned", both), ("defaults for not stated", conv)]:
        doc = per_doc(score(p, key))
        for a, b in [("agent-haiku", "rag-haiku"), ("agent-haiku", "full-haiku"), ("agent-opus", "rag-opus"),
                     ("agent-opus", "full-opus")]:
            rows.append({"scoring": label, **paired(doc, a, b)})
        for m in ["agent-haiku", "agent-opus"]:
            rows.append({"scoring": label, "contrast": f"mean {m}", "n": doc[m].notna().sum(), "diff": doc[m].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "posthoc_agent.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
