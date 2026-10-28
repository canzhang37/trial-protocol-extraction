"""Fields, prompt and output schema."""
import json

from pydantic import BaseModel, ConfigDict

NOT_STATED = "not stated"
CATEGORIES = {
    "phase": ["early phase 1", "1", "1/2", "2", "2/3", "3", "4", "not applicable"],
    "randomized": ["yes", "no"],
    "intervention_model": ["single group", "parallel", "crossover", "sequential", "factorial"],
    "blinding": ["open label", "blinded"],
    "sex": ["all", "female", "male"],
}
NUMERIC = ["number_of_arms", "minimum_age"]
FIELDS = ["phase", "randomized", "intervention_model", "blinding", "number_of_arms", "minimum_age", "sex"]

DEFINITIONS = """\
- phase: the trial phase. Use "early phase 1" for exploratory first-in-human studies \
(formerly phase 0), "1", "1/2", "2", "2/3", "3" or "4" for drug and biologic trials, and \
"not applicable" when the study has no phase, for example a device, procedure or \
behavioral study.
- randomized: "yes" if participants are assigned to arms at random, "no" if assignment is \
not random or there is a single arm.
- intervention_model: "single group" if every participant gets the same intervention, \
"parallel" if groups get different interventions at the same time, "crossover" if each \
participant gets two or more interventions in turn, "sequential" if groups are enrolled one \
after another based on earlier results (as in dose escalation cohorts), "factorial" if two \
or more interventions are given alone and in combination.
- blinding: "open label" if nobody is masked to the assigned intervention, "blinded" if \
participants, care providers, investigators or outcome assessors are masked.
- number_of_arms: the number of arms or groups the protocol defines, as a whole number \
written in digits. Cohorts or dose levels that get different interventions count as \
separate arms.
- minimum_age: the youngest age allowed by the eligibility criteria, in years, written in \
digits (for example "18"). Write "none" if the criteria set no lower age limit.
- sex: "all", "female" or "male", according to who may enroll."""

SYSTEM = f"""You read clinical trial protocols and record seven design facts about the \
trial, using these definitions:

{DEFINITIONS}

For each field, give the value and a short quote, copied word for word from the text \
(under 40 words), that supports it. If the text you were given does not state a field, \
write "{NOT_STATED}" as the value and an empty quote. Use only the text you were given."""


def json_schema():
    props = {}
    for f in FIELDS:
        value = {"type": "string", "enum": CATEGORIES[f] + [NOT_STATED]} if f in CATEGORIES else {"type": "string"}
        props[f] = {"type": "object", "properties": {"value": value, "quote": {"type": "string"}},
                    "required": ["value", "quote"], "additionalProperties": False}
    return {"type": "object", "properties": props, "required": FIELDS, "additionalProperties": False}


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str
    quote: str


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phase: Item
    randomized: Item
    intervention_model: Item
    blinding: Item
    number_of_arms: Item
    minimum_age: Item
    sex: Item


def parse(text):
    return Extraction.model_validate(json.loads(text))


def user_message(context, kind):
    head = ("The full text of a trial protocol follows." if kind == "full" else
            "Excerpts from a trial protocol follow, in document order. The rest of the protocol is not shown.")
    return f"{head}\n\n<protocol>\n{context}\n</protocol>\n\nRecord the seven fields."
