from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph.graph import END

from protex import agent

PAGES = [
    "Protocol title. A phase 2 randomized study of drug X.",
    "Eligibility: adults 18 years or older, men and women.",
    "Masking: this is an open label study. " + "filler " * 2000,
]


def test_search_caps_k_and_keeps_hits():
    ev = agent.Evidence(PAGES)
    out = ev.search("open label masking", k=50)
    assert "open label" in out.split("\n\n")[0]
    assert len(ev.calls) == 1 and ev.calls[0]["k"] == agent.MAX_K
    assert len(ev.calls[0]["pages"]) == agent.MAX_K == len(ev.hits)


def test_read_page_bounds_and_cut():
    ev = agent.Evidence(PAGES)
    assert "no page 9" in ev.read_page(9)
    text = ev.read_page(3)
    assert len(text.split()) <= agent.PAGE_WORDS + 2
    assert ev.calls[-1] == {"tool": "read_page", "page": 3, "ok": True}


def test_context_in_page_order_without_repeats():
    ev = agent.Evidence(PAGES)
    ev.read_page(2)
    ev.search("phase randomized", k=1)
    ev.search("phase randomized", k=1)
    ctx = ev.context()
    assert ctx.index("[page 1]") < ctx.index("[page 2]")
    assert ctx.count("[page 1]") == 1


def test_usd_counts_cache_writes_and_reads():
    u = {"input_tokens": 1000, "output_tokens": 100,
         "input_token_details": {"cache_read": 600, "ephemeral_5m_input_tokens": 300}}
    pin, pout, _, _ = agent.llm.PRICE["haiku"]
    want = (100 * pin + 300 * 1.25 * pin + 600 * 0.10 * pin + 100 * pout) / 1e6
    assert abs(agent.usd("haiku", u) - want) < 1e-12


def test_route_stops_at_call_limit():
    calls = [{"name": "search", "args": {"query": "q"}, "id": f"c{i}"} for i in range(3)]
    done = [ToolMessage("x", tool_call_id=f"t{i}") for i in range(agent.MAX_CALLS - 2)]
    state = {"messages": [HumanMessage("go"), *done, AIMessage("", tool_calls=calls)]}
    assert agent.route(state) == END
    state["messages"][-1] = AIMessage("", tool_calls=calls[:2])
    assert agent.route(state) == "tools"
    state["messages"][-1] = AIMessage("done")
    assert agent.route(state) == END
