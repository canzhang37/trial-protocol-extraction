# Can an LLM read a trial protocol? Extracting design fields from ClinicalTrials.gov PDFs: full text, retrieval and an agent

Clinical development teams read protocols to pull out the same design facts again and again. ClinicalTrials.gov records many of those facts as structured fields and, for thousands of studies, also hosts the protocol PDF, so the registry is a ready-made answer key. This project asks two questions on 148 cancer protocols drawn at random from the 10,589 interventional cancer studies with an uploaded protocol: how often do Claude Opus 5.5 and Claude Haiku 5.5 return the registry's values for seven design fields (phase, randomization, intervention model, blinding, number of arms, minimum age, sex), and does giving the model only the passages a BM25 retriever picks (retrieval-augmented generation, RAG) do as well as the whole protocol at a fraction of the tokens? A rule-based extractor is the baseline. Every answer comes with a verbatim quote, checked against the text. The plan, the checks and their thresholds were written down before any protocol was scored: [docs/design.md](docs/design.md). The results are in [docs/readout.md](docs/readout.md). A second part, also planned in advance ([docs/design_agent.md](docs/design_agent.md)), lets the model run its own searches as a LangGraph agent before the same extraction request ([docs/readout_agent.md](docs/readout_agent.md)).

| method | agreement with the registry | median input tokens | cost per protocol |
|---|---|---|---|
| Opus 5.5, full text | 86.1% | 59,860 | $0.141 |
| Opus 5.5, RAG | 84.5% | 16,663 | $0.036 |
| Haiku 5.5, full text | 81.8% | 59,860 | $0.009 |
| Haiku 5.5, RAG | 80.5% | 16,663 | $0.001 |
| rules | 58.6% | | |

Every LLM method beat the rules by 22 to 28 points, and 98% of Opus's RAG answers came with a quote found in the protocol. Three of the six scored checks failed. Retrieval cost Opus 1.6 points, and the interval reached 3.2 points against a limit of 3. Retrieved text was 28% of the full text, against a target of 25%. Haiku trailed Opus by 4 points with retrieval. A post hoc look found where most of the gap comes from: Haiku answers "not stated" where the registry has a convention (no drug phase is "not applicable", both sexes eligible is "all"), and every method does the same for blinding in open-label protocols. Scored with those conventions, every method agrees on 87-89% of fields and the gaps shrink to 0.4 and 1.3 points. That is not the planned scoring and the failed verdicts stand; it points to a prompt fix that would need testing on new protocols. The evaluation cost $27.70 through the Message Batches API.

![agreement by field](figures/accuracy_evaluation.png)

## Part 2: an agent that decides what to read

The agent gets two tools over the protocol (a BM25 search and a page reader), up to 12 calls, and the field definitions; whatever it reads becomes the context for the part 1 extraction request, so the agent arms differ from the RAG arms only in which passages the model sees.

| method | agreement | cost per protocol, standard prices |
|---|---|---|
| Opus 5.5, full text | 86.1% | $0.281 |
| Opus 5.5, agent | 85.1% | $0.126 |
| Opus 5.5, RAG | 84.5% | $0.073 |
| Haiku 5.5, agent | 81.9% | $0.005 |
| Haiku 5.5, full text | 81.8% | $0.018 |
| Haiku 5.5, RAG | 80.5% | $0.002 |

The agent did not beat fixed retrieval with either model (+1.4 points with Haiku, interval -0.2 to +3.1; +0.7 with Opus), so four of the six checks failed (the comparison with the full text and the tests passed). With Haiku it matched the full text at under a third of its cost, gaining where a fixed query misses the page with the answer (the title page for the phase, the schema for the arms), and it did nothing for blinding, the field expected to gain most. Its token count was as high as the full text's, because a loop resends its history every turn; prompt caching is what kept the cost low. Part 2 cost $20.08.

## Run

Python 3.10 or later:

```
pip install -e ".[dev]"
python -m protex frame
python -m protex download
python -m protex parse
python -m protex rules
python -m protex estimate --count
python -m protex submit --budget 45
python -m protex collect
python -m protex evaluate
python scripts/posthoc.py
python -m protex agent --role pilot --model haiku --budget 25
python -m protex agent --model haiku --budget 24
python -m protex agent --model opus --budget 24
python -m protex agent-evaluate
pytest -q
```

`submit`, `collect` and `estimate --count` call the Anthropic API and read `ANTHROPIC_API_KEY` from the environment or from a `.env` file in the project folder (not committed). `submit` counts the input tokens of every request first and stops if the estimate is above `--budget`. Responses are cached under `data/responses/`, so `evaluate` recomputes every number without calling the API again. The Docker image runs the same commands (`docker build -t protex .`); `data/` and `results/` are mounted at run time.

The pipeline lives in `src/protex/`: the registry frame and answer key (`registry.py`), PDF text (`pdftext.py`), chunking and BM25 (`retrieve.py`), the rule baseline (`rules.py`), the JSON schema and prompts (`schema.py`), batch requests and caching (`llm.py`), scoring with Wilson intervals, bootstrap and McNemar tests (`evaluate.py`, `report.py`), and the part 2 agent and its scoring (`agent.py`, `agent_report.py`). `agent` calls the API synchronously, caches each protocol's run under `data/agent/`, and stops when part 2 spending passes `--budget`.

## Data

Protocol PDFs and registry records from ClinicalTrials.gov (API v2), a service of the US National Library of Medicine; the PDFs are downloaded by the scripts and not committed. `results/` holds the sample, the answer key, every prediction with its quote, and the summary tables. The code is under the MIT license.

Can Zhang, zhan8639@umn.edu
