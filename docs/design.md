# Design: extracting trial design fields from protocol PDFs with an LLM, full text vs retrieval

Written before any protocol was downloaded for evaluation. Changes after that point go under Amendments, with the reason.

## Question

Clinical development teams read trial protocols to pull out the same design facts again and again. ClinicalTrials.gov records many of those facts as structured fields, and for thousands of studies it also hosts the protocol itself as a PDF. That makes the registry a ready-made answer key: can an LLM read a protocol and return the registry's values, and does giving it only the passages a retriever picks (retrieval-augmented generation, RAG) do as well as giving it the whole document, at a fraction of the tokens?

## Sample

Frame: studies returned by the ClinicalTrials.gov API v2 for the condition query "cancer", study type interventional, with an uploaded document flagged as containing the protocol. The frame is saved as a list of NCT IDs with the download date. It is shuffled with seed 2027. The first 5 studies are a pilot, used only to test the PDF parser and to estimate token counts and cost, and are never scored. The next 150 are the evaluation sample. When a study has several protocol documents, the first one by filename is used.

A protocol is excluded, and counted, if the PDF cannot be downloaded or opened, or if it yields fewer than 2,000 characters of text (a scanned document without a text layer; OCR is out of scope).

## Fields and answer key

Each value comes from the registry record of the same study.

| Field | Registry source | Scored as |
|---|---|---|
| phase | designModule.phases | exact category: early phase 1, 1, 1/2, 2, 2/3, 3, 4, not applicable |
| randomized | designInfo.allocation | yes if RANDOMIZED, no otherwise |
| intervention model | designInfo.interventionModel | exact: single group, parallel, crossover, sequential, factorial |
| blinding | designInfo.maskingInfo.masking | open label if NONE, blinded otherwise |
| number of arms | armGroups | exact count |
| minimum age | eligibilityModule.minimumAge | exact, in years; no minimum counts as its own value |
| sex | eligibilityModule.sex | exact: all, female, male |

The registry is entered by sponsors and can disagree with the protocol, for instance after an amendment. A disagreement is therefore not always an extraction error. Disagreements where every LLM method agrees with each other and not with the registry are counted and reported as probable registry-protocol discrepancies; they are not adjudicated by hand.

## Methods

The PDF text is extracted page by page with PyMuPDF and whitespace is normalized.

- **Rules**: regular expressions and keyword rules over the full text, written from the field definitions before any evaluation protocol is read.
- **Full text**: the whole protocol text in one request.
- **RAG**: the text is split into chunks of about 300 words with 50 words of overlap. For each field a fixed query retrieves the top 4 chunks by BM25. The union of retrieved chunks, in document order, goes into one request.

Each LLM method asks for all seven fields at once through a JSON schema (structured outputs), with, for every field, the value or null and a verbatim quote from the text that supports it. The same system prompt, with the field definitions taken from the ClinicalTrials.gov data element definitions, is used for every method.

Models: Claude Opus 5.5 and Claude Haiku 5.5, each with full text and with RAG, so the comparison is a 2 x 2 of context (full, RAG) by model. Requests go through the Message Batches API. Responses are cached on disk, so every number can be recomputed without calling the API again.

## Analysis

- Accuracy per field and method against the registry, with Wilson 95% intervals.
- Mean accuracy over the seven fields per protocol, and paired differences between methods with 95% intervals from a bootstrap over protocols (2,000 resamples, seed 2027); exact McNemar tests per field for the main contrasts.
- Grounding: the share of non-null LLM values whose quote is found in the protocol text after whitespace and case normalization.
- Input tokens and cost per protocol from the usage the API reports, at batch prices.

## Checks and verdicts

| id | check | rule |
|---|---|---|
| P1 | protocols with extractable text | at least 90% of the evaluation sample |
| P2 | every LLM method vs rules, mean accuracy | lower 95% bound of the paired difference above 0 |
| P3 | RAG vs full text with Opus 5.5 | lower 95% bound of the paired difference above -0.03 |
| P4 | RAG input tokens relative to full text, Opus 5.5 | median ratio at most 25% |
| P5 | grounding of Opus 5.5 with RAG | at least 95% of non-null values |
| P6 | Haiku 5.5 vs Opus 5.5, both with RAG | lower 95% bound of the paired difference above -0.05 |
| P7 | tests in continuous integration and in the Docker image | pass |

A check that fails is reported as failed. Prompts and retrieval settings are not tuned on the evaluation sample.

## Budget

Before the batches are sent, input tokens are counted for every request and the total cost is estimated. The run stops if the estimate is above the budget agreed in advance.

## Amendments

**Amendment 1, before any API call and before any scoring.**

- The 5 pilot protocols are also run through all four LLM methods once, to test the requests, the batch flow and the output schema end to end. Pilot outputs are not scored, and the prompt may change after the pilot only to fix a format or parsing failure, not to raise agreement.
- Scoring details the design left open: a failed request (an error, a refusal or output that does not parse) counts as wrong on every field; for minimum age, "not stated" is scored as no minimum; fields missing from the registry record are left out for that protocol.
- Grounding is reported twice: as planned (whitespace and case normalized), and also with punctuation removed, because PDF text extraction often changes hyphens, bullets and symbols such as ≥. P5 uses the planned version.
- Of the 150 evaluation protocols, 148 have a text layer. One PDF has 2 pages with 180 characters and one is a 109-page scan with no text.

**Amendment 2, after the pilot and before the evaluation batches.** The two Haiku pilot batches finished within about an hour, and all 10 outputs parsed. The two Opus pilot batches were still queued three hours after submission, so one synchronous Opus request per context type on the first pilot protocol was used to test the Opus output instead; both parsed. No prompt was changed. The evaluation batches were then submitted without waiting for the Opus pilot batches, whose outputs are checked when they arrive and, as planned, not scored.
