"""Requests, cost estimate and Message Batches."""
import json
import logging
import os
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from protex import schema

log = logging.getLogger("protex.llm")
MODELS = {"opus": "claude-opus-5-5", "haiku": "claude-haiku-5-5"}
# USD per M tokens; batches half price
PRICE = {"opus": (4.00, 20.00, 4.00, 20.00), "haiku": (0.10, 0.50, 0.50, 2.50)}
LONG = 100_000  # Haiku prompts above this bill at the higher rate
MAX_TOKENS = 16000
EFFORT = "medium"


def params(model, context, kind):
    return {
        "model": MODELS[model],
        "max_tokens": MAX_TOKENS,
        "system": schema.SYSTEM,
        "messages": [{"role": "user", "content": schema.user_message(context, kind)}],
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": schema.json_schema()}},
    }


def cost(model, input_tokens, output_tokens, batch=True):
    pin, pout, pin_long, pout_long = PRICE[model]
    if input_tokens > LONG:
        pin, pout = pin_long, pout_long
    usd = (input_tokens * pin + output_tokens * pout) / 1e6
    return usd / 2 if batch else usd


def count(client, p):
    # the output schema adds a few hundred tokens on top
    r = client.messages.count_tokens(model=p["model"], system=p["system"], messages=p["messages"])
    return r.input_tokens


def submit(client, reqs):
    batch = client.messages.batches.create(
        requests=[Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**p)) for cid, p in reqs])
    log.info("batch %s submitted with %d requests", batch.id, len(reqs))
    return batch.id


def wait(client, batch_id, every=60):
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            return b
        log.info("batch %s: %s, %d processing", batch_id, b.processing_status, b.request_counts.processing)
        time.sleep(every)


def collect(client, batch_id, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in client.messages.batches.results(batch_id):
        rec = {"custom_id": r.custom_id, "type": r.result.type}
        if r.result.type == "succeeded":
            rec["message"] = r.result.message.to_dict()
        elif r.result.type == "errored":
            rec["error"] = r.result.error.to_dict()
        (out / f"{r.custom_id}.json").write_text(json.dumps(rec), encoding="utf-8")
        n += 1
    return n


def read(rec):
    """Parsed extraction and usage from a cached batch result."""
    if rec["type"] != "succeeded":
        return None, None, rec["type"]
    m = rec["message"]
    usage = m["usage"]
    if m.get("stop_reason") != "end_turn":
        return None, usage, m.get("stop_reason")
    text = next(b["text"] for b in m["content"] if b["type"] == "text")
    try:
        return schema.parse(text), usage, "ok"
    except (ValueError, KeyError) as e:
        log.warning("%s: unparsable output (%s)", rec["custom_id"], e)
        return None, usage, "invalid"


def load_env(path=".env"):
    p = Path(path)
    if "ANTHROPIC_API_KEY" in os.environ or not p.exists():
        return
    # utf-8-sig: Notepad may add a byte order mark
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "ANTHROPIC_API_KEY" and v.strip():
            os.environ["ANTHROPIC_API_KEY"] = v.strip().strip('"')


def client():
    load_env()
    return anthropic.Anthropic(max_retries=4)
