"""ClinicalTrials.gov frame, answer key and PDF download."""
import json
import random
import time
from pathlib import Path

import requests

API = "https://clinicaltrials.gov/api/v2/studies"
CDN = "https://cdn.clinicaltrials.gov/large-docs/{tail}/{nct}/{fn}"
FIELDS = ["NCTId", "BriefTitle", "Phase", "DesignAllocation", "DesignInterventionModel", "DesignMasking",
          "ArmGroupLabel", "MinimumAge", "Sex", "LargeDocFilename", "LargeDocHasProtocol", "LargeDocSize"]
QUERY = {"query.cond": "cancer",
         "filter.advanced": "AREA[LargeDocHasProtocol]true AND AREA[StudyType]INTERVENTIONAL"}

PHASES = {("EARLY_PHASE1",): "early phase 1", ("PHASE1",): "1", ("PHASE1", "PHASE2"): "1/2",
          ("PHASE2",): "2", ("PHASE2", "PHASE3"): "2/3", ("PHASE3",): "3", ("PHASE4",): "4",
          ("NA",): "not applicable"}
MODELS = {"SINGLE_GROUP": "single group", "PARALLEL": "parallel", "CROSSOVER": "crossover",
          "SEQUENTIAL": "sequential", "FACTORIAL": "factorial"}


def fetch_frame(session=None):
    s = session or requests.Session()
    params = dict(QUERY, fields=",".join(FIELDS), pageSize=1000, countTotal="true")
    studies, total = [], None
    while True:
        r = s.get(API, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()
        total = d.get("totalCount", total)
        studies += d["studies"]
        if not d.get("nextPageToken"):
            break
        params["pageToken"] = d["nextPageToken"]
        time.sleep(0.5)
    return studies, total


def protocol_doc(study):
    docs = study.get("documentSection", {}).get("largeDocumentModule", {}).get("largeDocs", [])
    docs = sorted((x for x in docs if x.get("hasProtocol")), key=lambda x: x["filename"])
    return docs[0] if docs else None


def age_years(txt):
    if not txt:
        return "none"
    n, unit = txt.split()[:2]
    per = {"Year": 1, "Month": 1 / 12, "Week": 1 / 52, "Day": 1 / 365, "Hour": 1 / 8760, "Minute": 1 / 525600}
    f = next(v for k, v in per.items() if unit.startswith(k))
    return round(float(n) * f, 2)


def answer_key(study):
    p = study["protocolSection"]
    dm = p.get("designModule", {})
    di = dm.get("designInfo", {})
    arms = p.get("armsInterventionsModule", {}).get("armGroups")
    el = p.get("eligibilityModule", {})
    mask = di.get("maskingInfo", {}).get("masking")
    return {
        "phase": PHASES.get(tuple(dm.get("phases", []))),
        "randomized": None if "allocation" not in di and not arms else ("yes" if di.get("allocation") == "RANDOMIZED" else "no"),
        "intervention_model": MODELS.get(di.get("interventionModel")),
        "blinding": None if mask is None else ("open label" if mask == "NONE" else "blinded"),
        "number_of_arms": len(arms) if arms else None,
        "minimum_age": age_years(el.get("minimumAge")),
        "sex": {"ALL": "all", "FEMALE": "female", "MALE": "male"}.get(el.get("sex")),
    }


def draw(studies, seed=2027, pilot=5, n=150):
    ids = sorted(s["protocolSection"]["identificationModule"]["nctId"] for s in studies)
    random.Random(seed).shuffle(ids)
    return ids[:pilot], ids[pilot:pilot + n]


def pdf_url(nct, filename):
    return CDN.format(tail=nct[-2:], nct=nct, fn=filename)


def download(nct, filename, out_dir, session=None):
    out = Path(out_dir) / f"{nct}.pdf"
    if out.exists() and out.stat().st_size > 0:
        return out
    s = session or requests.Session()
    r = s.get(pdf_url(nct, filename), timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1), encoding="utf-8")
