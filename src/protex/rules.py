"""Keyword and regex baseline."""
import re

ROMAN = {"0": "early phase 1", "i": "1", "1": "1", "ii": "2", "2": "2", "iii": "3", "3": "3", "iv": "4", "4": "4"}
PHASE = re.compile(r"\bphase\s*(0|iv|iii|ii|i|1|2|3|4)\b(?:\s*(?:/|-|and)\s*(?:phase\s*)?(iv|iii|ii|i|1|2|3|4)\b)?", re.I)
AGE = re.compile(r"(?:aged?|age of|years of age|age)\s*(?:≥|>=|>|at least|of at least|of|is)?\s*(\d{1,2})\s*(?:years|yrs)", re.I)
AGE2 = re.compile(r"(?:≥|>=)\s*(\d{1,2})\s*(?:years|yrs)", re.I)
ARM = re.compile(r"\b(?:arm|cohort|group)\s+([a-h]|[1-9])\b", re.I)


def head(pages, n=3):
    return " ".join(pages[:n])


def phase(pages):
    m = PHASE.search(head(pages)) or PHASE.search(" ".join(pages))
    if not m:
        return "not applicable"
    a = ROMAN[m.group(1).lower()]
    if m.group(2):
        b = ROMAN[m.group(2).lower()]
        pair = f"{a}/{b}"
        return pair if pair in ("1/2", "2/3") else a
    return a


def randomized(text):
    t = re.sub(r"non[- ]?randomi[sz]ed", " ", text.lower())
    return "yes" if re.search(r"\brandomi[sz](ed|ation)\b", t) else "no"


def blinding(text):
    t = text.lower()
    if re.search(r"double[- ]blind|single[- ]blind|triple[- ]blind|placebo[- ]controlled|\bmasked\b", t):
        return "blinded"
    return "open label"


def intervention_model(text, rand):
    t = text.lower()
    if re.search(r"cross[- ]?over", t) and rand == "yes":
        return "crossover"
    if "factorial" in t:
        return "factorial"
    return "parallel" if rand == "yes" else "single group"


def number_of_arms(text, rand):
    labels = {m.group(1).lower() for m in ARM.finditer(text)}
    if labels:
        return str(len(labels))
    return "2" if rand == "yes" else "1"


def minimum_age(text):
    m = AGE.search(text) or AGE2.search(text)
    return m.group(1) if m else "18"


def sex(text):
    t = text.lower()
    if re.search(r"prostate|\bmale patients\b|\bmen with\b", t) and not re.search(r"\bwomen\b|female", t):
        return "male"
    if re.search(r"ovarian|cervical|endometrial|uterine|\bwomen with\b|female patients", t) and not re.search(r"\bmen\b|\bmale\b", t):
        return "female"
    return "all"


def extract(pages):
    text = " ".join(pages)
    r = randomized(text)
    return {"phase": phase(pages), "randomized": r, "intervention_model": intervention_model(text, r),
            "blinding": blinding(text), "number_of_arms": number_of_arms(text, r),
            "minimum_age": minimum_age(text), "sex": sex(text)}
