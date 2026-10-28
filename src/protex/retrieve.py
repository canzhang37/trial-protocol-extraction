"""Chunking and BM25 retrieval."""
import re

from rank_bm25 import BM25Okapi

SIZE, OVERLAP, TOP_K = 300, 50, 4
QUERIES = {
    "phase": "phase study protocol title phase I II III trial",
    "randomized": "randomized randomization randomly assigned allocation ratio stratified",
    "intervention_model": "study design parallel crossover single arm cohorts sequential dose escalation factorial",
    "blinding": "blinding masking double blind open label placebo unblinded",
    "number_of_arms": "arms treatment arm cohort group assigned regimen",
    "minimum_age": "inclusion criteria age years older eligible",
    "sex": "inclusion criteria male female women men gender",
}
WORD = re.compile(r"[a-z0-9]+")


def tokens(s):
    return WORD.findall(s.lower())


def chunk(pages, size=SIZE, overlap=OVERLAP):
    words = [(i + 1, w) for i, p in enumerate(pages) for w in p.split()]
    out, step = [], size - overlap
    for start in range(0, max(len(words) - overlap, 1), step):
        part = words[start:start + size]
        if not part:
            break
        out.append({"start": start, "page": part[0][0], "text": " ".join(w for _, w in part)})
    return out


def retrieve(chunks, top_k=TOP_K, queries=QUERIES):
    bm = BM25Okapi([tokens(c["text"]) for c in chunks])
    picked = {}
    for field, q in queries.items():
        scores = bm.get_scores(tokens(q))
        for j in sorted(range(len(chunks)), key=lambda k: -scores[k])[:top_k]:
            picked.setdefault(j, []).append(field)
    return [dict(chunks[j], fields=picked[j]) for j in sorted(picked)]


def context(selected):
    return "\n\n".join(f"[page {c['page']}] {c['text']}" for c in selected)
