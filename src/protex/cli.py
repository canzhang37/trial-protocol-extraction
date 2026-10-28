"""Command line: python -m protex <command>."""
import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd

from protex import llm, pdftext, registry, report, retrieve, rules

log = logging.getLogger("protex")
DATA = Path("data")
RESULTS = Path("results")
METHODS = ["full-opus", "rag-opus", "full-haiku", "rag-haiku"]


def cmd_frame(a):
    studies, total = registry.fetch_frame()
    registry.save({"downloaded": dt.date.today().isoformat(), "total": total, "studies": studies},
                  DATA / "frame.json")
    pilot, sample = registry.draw(studies, seed=a.seed, pilot=a.pilot, n=a.n)
    by_id = {s["protocolSection"]["identificationModule"]["nctId"]: s for s in studies}
    rows = []
    for role, ids in (("pilot", pilot), ("evaluation", sample)):
        for nct in ids:
            doc = registry.protocol_doc(by_id[nct])
            rows.append({"nct_id": nct, "role": role, "filename": doc["filename"], "size": doc.get("size")})
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "sample.csv", index=False)
    key = [{"nct_id": nct, **registry.answer_key(by_id[nct])} for nct in pilot + sample]
    pd.DataFrame(key).to_csv(RESULTS / "answer_key.csv", index=False)
    log.info("frame %d studies (API total %s); pilot %d, evaluation %d", len(studies), total, len(pilot), len(sample))


def cmd_download(a):
    s = pd.read_csv(RESULTS / "sample.csv")
    if a.role:
        s = s[s.role == a.role]
    out = DATA / "pdf"
    out.mkdir(parents=True, exist_ok=True)
    for r in s.itertuples():
        try:
            registry.download(r.nct_id, r.filename, out)
        except Exception as e:  # noqa: BLE001
            log.warning("%s download failed: %s", r.nct_id, e)


def cmd_parse(a):
    s = pd.read_csv(RESULTS / "sample.csv")
    if a.role:
        s = s[s.role == a.role]
    out = DATA / "text"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in s.itertuples():
        pdf = DATA / "pdf" / f"{r.nct_id}.pdf"
        row = {"nct_id": r.nct_id, "role": r.role, "downloaded": pdf.exists()}
        if pdf.exists():
            try:
                pg = pdftext.pages(pdf)
                (out / f"{r.nct_id}.json").write_text(json.dumps(pg), encoding="utf-8")
                row.update(pages=len(pg), chars=sum(map(len, pg)), has_text=pdftext.has_text(pg))
            except Exception as e:  # noqa: BLE001
                row.update(pages=0, chars=0, has_text=False, error=str(e)[:200])
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / f"parse_{a.role or 'all'}.csv", index=False)
    log.info("parsed %d", len(rows))


def usable(role):
    p = pd.read_csv(RESULTS / f"parse_{role}.csv")
    return p[p.has_text.fillna(False).astype(bool)].nct_id.tolist()


def load_pages(nct):
    return json.loads((DATA / "text" / f"{nct}.json").read_text(encoding="utf-8"))


def contexts(nct, kind):
    pg = load_pages(nct)
    if kind == "full":
        return pdftext.full_text(pg)
    return retrieve.context(retrieve.retrieve(retrieve.chunk(pg)))


def cmd_rules(a):
    rows = []
    for nct in usable(a.role):
        for f, v in rules.extract(load_pages(nct)).items():
            rows.append({"nct_id": nct, "method": "rules", "field": f, "value": v, "quote": ""})
    pd.DataFrame(rows).to_csv(RESULTS / f"rules_{a.role}.csv", index=False)


def plan(role, methods):
    reqs = []
    for m in methods:
        kind, model = m.split("-")
        for nct in usable(role):
            reqs.append((f"{kind}-{model}-{nct}", model, llm.params(model, contexts(nct, kind), kind)))
    return reqs


def cmd_estimate(a):
    reqs = plan(a.role, a.methods)
    c = llm.client() if a.count else None
    rows = []
    for cid, model, p in reqs:
        chars = len(p["system"]) + len(p["messages"][0]["content"])
        tok = llm.count(c, p) if c else int(chars / 3.2)
        rows.append({"custom_id": cid, "model": model, "chars": chars, "input_tokens": tok,
                     "usd_batch": llm.cost(model, tok, a.out_tokens)})
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / f"estimate_{a.role}.csv", index=False)
    d["method"] = d.custom_id.str.rsplit("-", n=1).str[0]
    print(d.groupby("method").agg(requests=("custom_id", "size"), median_tokens=("input_tokens", "median"),
                                   total_tokens=("input_tokens", "sum"), usd=("usd_batch", "sum")).round(2))
    print(f"total estimated cost (batch prices, {a.out_tokens} output tokens each): ${d.usd_batch.sum():.2f}")


def cmd_submit(a):
    est = pd.read_csv(RESULTS / f"estimate_{a.role}.csv")
    if est.usd_batch.sum() > a.budget:
        raise SystemExit(f"estimate ${est.usd_batch.sum():.2f} is above the budget ${a.budget:.2f}")
    path = DATA / f"batches_{a.role}.json"
    batches = json.loads(path.read_text()) if path.exists() else {}
    c = llm.client()
    for m in a.methods:
        if m in batches:
            log.info("%s already submitted as %s", m, batches[m])
            continue
        reqs = [(cid, p) for cid, _, p in plan(a.role, [m])]
        batches[m] = llm.submit(c, reqs)
        path.write_text(json.dumps(batches, indent=1))


def cmd_collect(a):
    batches = json.loads((DATA / f"batches_{a.role}.json").read_text())
    c = llm.client()
    for m, bid in batches.items():
        b = llm.wait(c, bid)
        n = llm.collect(c, bid, DATA / "responses")
        log.info("%s: %d results (%s)", m, n, b.request_counts)


def cmd_evaluate(a):
    report.run(a.role, DATA, RESULTS)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="protex")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frame", help="fetch the registry frame and draw the sample")
    f.add_argument("--seed", type=int, default=2027)
    f.add_argument("--pilot", type=int, default=5)
    f.add_argument("--n", type=int, default=150)
    for name in ("download", "parse"):
        p = sub.add_parser(name)
        p.add_argument("--role", choices=["pilot", "evaluation"])
    for name in ("rules", "estimate", "submit", "collect", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--role", choices=["pilot", "evaluation"], default="evaluation")
        if name in ("estimate", "submit"):
            p.add_argument("--methods", nargs="+", default=METHODS)
        if name == "estimate":
            p.add_argument("--count", action="store_true", help="count tokens with the API")
            p.add_argument("--out-tokens", type=int, default=3000)
        if name == "submit":
            p.add_argument("--budget", type=float, required=True, help="USD")
    a = ap.parse_args(argv)
    {"frame": cmd_frame, "download": cmd_download, "parse": cmd_parse, "rules": cmd_rules,
     "estimate": cmd_estimate, "submit": cmd_submit, "collect": cmd_collect, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
