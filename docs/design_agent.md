# Design, part 2: an agent that decides what to read

Written after the part 1 results ([readout.md](readout.md)) and before any agent request was sent. Changes after this point go under Amendments, with the reason.

## Question

In part 1, retrieval gave the model four fixed BM25 passages per field. It cost Opus 1.6 points of agreement against the full text, mostly on blinding, where the deciding sentence is often far from a "masking" keyword. Would a model that chooses its own searches, reads the pages it wants and stops when it has enough do better than fixed retrieval, and at what cost in tokens?

## Design

The agent only gathers evidence; the extraction step is the part 1 request unchanged. That way the agent arms differ from the RAG arms in one thing: which passages the extraction model sees.

- **Evidence step.** A LangGraph loop of a Claude model with two tools over the protocol text:
  - `search(query, k)`: the top k passages (k at most 5) by BM25 over the part 1 chunks (300 words, 50 overlap), each with its page number;
  - `read_page(page)`: the text of one page, cut at 1,500 words.
  The model sees the field definitions, the number of pages and no protocol text at the start. It is told to find the passages that state each of the seven fields, to search again with other words when a search misses, and to stop when it has evidence for every field or has searched enough to judge a field not stated. It may make at most 12 tool calls; at the limit the loop stops.
- **Context.** Every passage and page the tools returned, without repeats, in document order, each marked with its page as in part 1.
- **Extraction step.** The part 1 request (`llm.params`, same system prompt, JSON schema and effort) with that context in place of the BM25 passages, as a synchronous request.
- **Models.** Claude Haiku 5.5 on all 148 evaluation protocols (`agent-haiku`), and Claude Opus 5.5 on the first 50 evaluation protocols with text, in `results/sample.csv` order (`agent-opus`). The evidence step uses the same model as the extraction step, effort medium, and automatic prompt caching.
- **Development.** The tools and the evidence prompt are written and tried on the 5 pilot protocols only. After the pilot, changes are allowed only to fix a failure of the loop or of the format, not to raise agreement. The evaluation protocols are not looked at before the runs.

## Scoring

As in part 1 (Amendment 1): registry values as the answer key, a failed request (error, refusal, unparsable output, or no extraction) wrong on every field, "not stated" for minimum age scored as no minimum. The comparisons are paired on the same protocols, with 95% intervals from a bootstrap over protocols (2,000 resamples, seed 2027), and exact McNemar tests per field.

## Checks and verdicts

| id | check | rule |
|---|---|---|
| K1 | agent vs fixed RAG, Haiku, mean agreement | lower 95% bound of the paired difference above 0 |
| K2 | agent vs full text, Haiku | lower 95% bound of the paired difference above -0.03 |
| K3 | input tokens per protocol (evidence and extraction steps together), agent-haiku over full-haiku | median ratio at most 50% |
| K4 | quotes found in the text, agent-haiku | at least 95% of non-null values (strict, as P5) |
| K5 | agent vs fixed RAG, Opus, on the 50 protocols | lower 95% bound of the paired difference above 0 |
| K6 | tests in continuous integration and in the Docker image | pass |

Also reported: tool calls per protocol, how often the 12-call limit was reached, cost per protocol, agreement by field, and the share of "not stated" answers.

Expectations, written in advance:

- E1: the agent gains most on blinding.
- E2: the agent does not remove Haiku's "not stated" answers on phase and sex, because they come from the extraction prompt, which is unchanged.
- E3: the agent costs more per protocol than fixed RAG and less than the full text.

## Budget

$25 for part 2, the pilot included. After the pilot, the cost per protocol is measured for each model. If the projected total for 148 Haiku protocols and 50 Opus protocols is above what is left of the budget, the Opus subset is cut to the largest multiple of 10 that fits, and the cut is reported.

## Amendments

**Amendment 1, after the pilot and before any evaluation request.** The pilot ran all 10 agent requests (5 protocols, both models) without a loop or format failure: no refusal, no protocol reached the 12-call limit, and every extraction parsed, so nothing was changed. It cost $0.64: about $0.006 per protocol with Haiku and $0.12 with Opus. At that rate Opus on all 148 evaluation protocols fits the budget (about $20 projected with Haiku and the pilot), so `agent-opus` runs on all 148 instead of the first 50, which makes K5 a test on the full sample. K5 is judged on all 148; the first 50, as first planned, are also reported. The run stops if spending reaches $24.
