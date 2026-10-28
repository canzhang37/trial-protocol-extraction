"""Page text from protocol PDFs."""
import re

import pymupdf

MIN_CHARS = 2000


def normalize(s):
    s = s.replace(" ", " ").replace("‐", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", s).strip()


def pages(path):
    with pymupdf.open(path) as doc:
        return [normalize(p.get_text()) for p in doc]


def full_text(pg):
    return "\n\n".join(f"[page {i + 1}] {t}" for i, t in enumerate(pg) if t)


def has_text(pg):
    return sum(len(t) for t in pg) >= MIN_CHARS
