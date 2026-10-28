import json

import pandas as pd
import pytest

from protex import evaluate, llm, registry, retrieve, rules, schema


def study(phases=("PHASE2",), allocation="RANDOMIZED", model="PARALLEL", masking="DOUBLE",
          arms=("A", "B"), age="18 Years", sex="ALL"):
    di = {"allocation": allocation, "interventionModel": model, "maskingInfo": {"masking": masking}}
    return {"protocolSection": {
        "identificationModule": {"nctId": "NCT00000001"},
        "designModule": {"phases": list(phases), "designInfo": di},
        "armsInterventionsModule": {"armGroups": [{"label": a} for a in arms]},
        "eligibilityModule": {"minimumAge": age, "sex": sex}}}


def test_answer_key_maps_registry_values():
    k = registry.answer_key(study())
    assert k == {"phase": "2", "randomized": "yes", "intervention_model": "parallel", "blinding": "blinded",
                 "number_of_arms": 2, "minimum_age": 18.0, "sex": "all"}
    k = registry.answer_key(study(phases=("PHASE1", "PHASE2"), allocation="NA", model="SINGLE_GROUP",
                                  masking="NONE", arms=("A",), age=None, sex="FEMALE"))
    assert (k["phase"], k["randomized"], k["blinding"], k["minimum_age"], k["sex"]) == ("1/2", "no", "open label", "none", "female")
    assert registry.age_years("6 Months") == 0.5


def test_pdf_url():
    assert registry.pdf_url("NCT01027416", "Prot_SAP_000.pdf") == \
        "https://cdn.clinicaltrials.gov/large-docs/16/NCT01027416/Prot_SAP_000.pdf"


def test_draw_is_reproducible_and_disjoint():
    s = [study() for _ in range(30)]
    for i, x in enumerate(s):
        x["protocolSection"]["identificationModule"]["nctId"] = f"NCT{i:08d}"
    p1, e1 = registry.draw(s, pilot=5, n=20)
    p2, e2 = registry.draw(list(reversed(s)), pilot=5, n=20)
    assert (p1, e1) == (p2, e2) and not set(p1) & set(e1)


def test_chunks_overlap_and_retrieve_finds_the_age_criterion():
    filler = [" ".join(f"p{k}w{i}" for i in range(400)) for k in range(5)]
    pages = filler + ["Inclusion criteria: patients aged 18 years or older are eligible."]
    ch = retrieve.chunk(pages)
    assert ch[0]["text"].split()[-50:] == ch[1]["text"].split()[:50]
    sel = retrieve.retrieve(ch, top_k=1, queries={"minimum_age": retrieve.QUERIES["minimum_age"]})
    assert "aged 18 years" in sel[0]["text"] and sel[0]["page"] >= 5


def test_rules_on_a_toy_protocol():
    pages = ["A Phase III, Randomized, Double-Blind, Placebo-Controlled Study",
             "Arm A: drug. Arm B: placebo. Patients must be aged ≥ 18 years. Men and women may enroll."]
    r = rules.extract(pages)
    assert r == {"phase": "3", "randomized": "yes", "intervention_model": "parallel", "blinding": "blinded",
                 "number_of_arms": "2", "minimum_age": "18", "sex": "all"}
    assert rules.phase(["Phase I/II dose escalation"]) == "1/2"
    assert rules.randomized("a non-randomized study") == "no"


def test_schema_is_strict_and_parses_a_response():
    s = schema.json_schema()
    assert s["additionalProperties"] is False and set(s["required"]) == set(schema.FIELDS)
    out = {f: {"value": "not stated", "quote": ""} for f in schema.FIELDS}
    out["phase"] = {"value": "2", "quote": "a phase 2 study"}
    assert schema.parse(json.dumps(out)).phase.value == "2"
    out["extra"] = {}
    with pytest.raises(ValueError):
        schema.parse(json.dumps(out))


def test_request_parameters():
    p = llm.params("opus", "text", "rag")
    assert p["model"] == "claude-opus-5-5" and "temperature" not in p
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert "Excerpts" in p["messages"][0]["content"]


def test_cost_uses_batch_and_long_prompt_rates():
    assert llm.cost("opus", 1_000_000, 0) == pytest.approx(2.0)
    assert llm.cost("haiku", 50_000, 0) == pytest.approx(0.0025)
    assert llm.cost("haiku", 200_000, 0) == pytest.approx(0.05)


def test_read_handles_success_refusal_and_errors():
    body = json.dumps({f: {"value": "not stated", "quote": ""} for f in schema.FIELDS})
    ok = {"custom_id": "x", "type": "succeeded", "message": {"stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1},
                                                             "content": [{"type": "text", "text": body}]}}
    assert llm.read(ok)[2] == "ok"
    ref = {"custom_id": "x", "type": "succeeded", "message": {"stop_reason": "refusal", "usage": {}, "content": []}}
    assert llm.read(ref)[2] == "refusal"
    assert llm.read({"custom_id": "x", "type": "errored"})[2] == "errored"


def test_scoring_and_intervals():
    key = pd.DataFrame([{"nct_id": "a", "phase": "2", "randomized": "yes", "intervention_model": "parallel",
                         "blinding": "blinded", "number_of_arms": "2", "minimum_age": "18.0", "sex": "all"}])
    pred = pd.DataFrame([{"nct_id": "a", "method": "m", "field": "minimum_age", "value": "18"},
                         {"nct_id": "a", "method": "m", "field": "number_of_arms", "value": "3"}])
    sc = evaluate.score(pred, key)
    assert sc.set_index("field").correct.to_dict() == {"minimum_age": True, "number_of_arms": False}
    lo, hi = evaluate.wilson(8, 10)
    assert round(lo, 3) == 0.490 and round(hi, 3) == 0.943
    assert evaluate.correct("minimum_age", "not stated", "none") is True


def test_grounding_normalizes_whitespace_and_case():
    text = "Patients must be\n  18 YEARS or older."
    assert evaluate.grounded("patients must be 18 years", text)
    assert not evaluate.grounded("patients must be 21 years", text)
    assert evaluate.grounded("18 years, or older", text, strict=False)
