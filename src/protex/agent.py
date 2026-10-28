"""Part 2: an agent gathers evidence, then the part 1 request extracts."""
import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from rank_bm25 import BM25Okapi

from protex import llm, retrieve, schema

log = logging.getLogger("protex.agent")
MAX_CALLS = 12
MAX_K = 5
PAGE_WORDS = 1500
AGENT_TOKENS = 8000
# cache reads as a share of base input price
CACHE_READ = {"opus": 0.05, "haiku": 0.10}

SYSTEM = f"""You help extract seven design facts from a clinical trial protocol. You \
cannot see the protocol. You can search it and read its pages with the tools. The facts:

{schema.DEFINITIONS}

Find the passages that state each fact. If a search misses, search again with other \
words, or read the page that likely holds the protocol summary, schema or study design. \
Stop when you have evidence for every fact, or have searched enough to judge that a fact \
is not stated. You may make at most {MAX_CALLS} tool calls. When you stop, reply with one \
short line per fact saying where you found it, or "not found"."""


class Evidence:
    """Tools over one protocol, and the text they returned."""

    def __init__(self, pages):
        self.pages = pages
        self.chunks = retrieve.chunk(pages)
        self.bm = BM25Okapi([retrieve.tokens(c["text"]) for c in self.chunks])
        self.hits = {}  # (page, start) -> text
        self.calls = []

    def search(self, query, k=3):
        k = max(1, min(int(k), MAX_K))
        scores = self.bm.get_scores(retrieve.tokens(query))
        top = sorted(range(len(self.chunks)), key=lambda j: -scores[j])[:k]
        out = []
        for j in top:
            c = self.chunks[j]
            self.hits[(c["page"], c["start"])] = c["text"]
            out.append(f"[page {c['page']}] {c['text']}")
        self.calls.append({"tool": "search", "query": query, "k": k, "pages": [self.chunks[j]["page"] for j in top]})
        return "\n\n".join(out) if out else "no passages"

    def read_page(self, page):
        page = int(page)
        if not 1 <= page <= len(self.pages):
            self.calls.append({"tool": "read_page", "page": page, "ok": False})
            return f"no page {page}; the protocol has {len(self.pages)} pages"
        text = " ".join(self.pages[page - 1].split()[:PAGE_WORDS])
        self.hits[(page, -1)] = text
        self.calls.append({"tool": "read_page", "page": page, "ok": True})
        return f"[page {page}] {text}"

    def context(self):
        # document order; full page before its passages
        return "\n\n".join(f"[page {p}] {t}" for (p, _), t in sorted(self.hits.items()))

    def tools(self):
        @tool
        def search(query: str, k: int = 3) -> str:
            """Search the protocol for passages that match the query (BM25 keyword search).
            Returns up to k passages (k from 1 to 5), each with its page number."""
            return self.search(query, k)

        @tool
        def read_page(page: int) -> str:
            """Return the text of one page of the protocol, cut at 1,500 words."""
            return self.read_page(page)

        return [search, read_page]


def route(state):
    last = state["messages"][-1]
    done = sum(isinstance(m, ToolMessage) for m in state["messages"])
    if getattr(last, "tool_calls", None) and done + len(last.tool_calls) <= MAX_CALLS:
        return "tools"
    return END


def graph(chat, tools):
    def agent(state):
        return {"messages": [chat.invoke(state["messages"], cache_control={"type": "ephemeral"})]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def usd(model, u):
    pin, pout, _, _ = llm.PRICE[model]
    d = u.get("input_token_details") or {}
    read = d.get("cache_read") or 0
    # writes may come only as the per-TTL counts
    w5 = d.get("ephemeral_5m_input_tokens") or 0
    w1 = d.get("ephemeral_1h_input_tokens") or 0
    if not (w5 or w1):
        w5 = d.get("cache_creation") or 0
    base = u["input_tokens"] - read - w5 - w1
    return (base * pin + (1.25 * w5 + 2 * w1) * pin + read * CACHE_READ[model] * pin
            + u["output_tokens"] * pout) / 1e6


def run(nct, pages, model, client):
    ev = Evidence(pages)
    tools = ev.tools()
    chat = ChatAnthropic(model=llm.MODELS[model], max_tokens=AGENT_TOKENS, effort=llm.EFFORT,
                         max_retries=4).bind_tools(tools)
    start = [SystemMessage(SYSTEM),
             HumanMessage(f"The protocol has {len(pages)} pages. Find the evidence for the seven facts.")]
    rec = {"nct_id": nct, "model": model}
    try:
        state = graph(chat, tools).invoke({"messages": start}, {"recursion_limit": 2 * MAX_CALLS + 5})
        msgs = state["messages"]
        rec["agent_status"] = "ok"
    except Exception as e:  # noqa: BLE001
        log.warning("%s %s agent step failed: %s", model, nct, e)
        msgs, rec["agent_status"] = [], f"error: {str(e)[:200]}"
    ai = [m for m in msgs if m.type == "ai"]
    rec["turns"] = len(ai)
    rec["calls"] = ev.calls
    rec["hit_limit"] = bool(ai and getattr(ai[-1], "tool_calls", None))
    rec["agent_stop"] = [m.response_metadata.get("stop_reason") for m in ai]
    rec["agent_usage"] = [dict(m.usage_metadata or {}) for m in ai]
    rec["agent_usd"] = sum(usd(model, u) for u in rec["agent_usage"] if u)
    rec["agent_note"] = ai[-1].text if ai else ""
    ctx = ev.context()
    rec["context_chars"] = len(ctx)
    try:
        m = client.messages.create(**llm.params(model, ctx or "(no passages were found)", "rag"))
        rec["extraction"] = {"custom_id": f"agent-{model}-{nct}", "type": "succeeded", "message": m.to_dict()}
        u = m.usage
        rec["extract_usd"] = llm.cost(model, u.input_tokens, u.output_tokens, batch=False)
    except Exception as e:  # noqa: BLE001
        log.warning("%s %s extraction failed: %s", model, nct, e)
        rec["extraction"] = {"custom_id": f"agent-{model}-{nct}", "type": "errored", "error": str(e)[:300]}
        rec["extract_usd"] = 0.0
    return rec


def save(rec, out):
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"agent-{rec['model']}-{rec['nct_id']}.json"
    p.write_text(json.dumps(rec, default=str), encoding="utf-8")
    return p
