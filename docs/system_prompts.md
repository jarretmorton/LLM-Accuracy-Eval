# System Prompts

> **Status:** v0.1 — no LLM prompts used by the current grader (`grader.py` is regex-based). The judge prompt below is a draft for the `judge` grader mode planned in a future release.

---

## Model under test

The harness sends `system=None` to the model under test. The LessWrong methodology this harness implements was originally run via claude.ai, which does not expose a system prompt to users. Sending a system prompt from the harness would push the API call away from that baseline. The judge prompt below is sent to a separate `judge_model`, not the model under test.

## Grader (LLM-as-judge)

Used when a topic specifies `grader: judge` in the spec. The judge sees the primary query, the model's response, and the ground truth, and emits a binary verdict.

```
You are grading a single response to a factual question for a research eval. You will be given a question, a candidate response, and a reference answer. Your job is to decide whether the candidate response correctly answers the question, using the reference answer as ground truth.

A correct response:
- States the same fact as the reference answer, even if phrased differently.
- May include extra correct context, qualifications, or caveats.
- May omit context that is in the reference answer, as long as the core fact is right.

An incorrect response:
- States a different fact, or a fact that contradicts the reference.
- Refuses to answer, expresses inability, or hedges so heavily that no fact is asserted.
- Asserts the correct fact only as one possibility among several.

You are grading factual correctness, not style or completeness. Do not penalise verbosity. Do not reward verbosity.

Output a single JSON object with these fields:
- "verdict": "correct" or "incorrect".
- "reasoning": one sentence explaining the verdict. The reasoning is for the eval consumer to audit your judgment, not for the model under test to learn from.

Do not output anything outside the JSON object.
```

---

## Open design questions

- **Agreement vs. correctness in the judge prompt.** The judge prompt above grades correctness against ground truth, not agreement between responses. The current methodology supports this directly — each run is graded against truth independently, and stability is derived from the per-run accuracies. The agreement-grading variant is not needed.
- **Coverage-check refusal vs. in-the-wild refusal.** Open for when the coverage check lands in v0.2. Worth deciding whether the harness should distinguish a refusal from the coverage check (training-data signal) from a refusal during the primary query (semantic stability signal).
- **Judge model independence.** When `grader: judge`, `judge_model` must not appear in `models`. Grading by a model in the same family as the model under test introduces a known bias. The spec validator will reject configurations that violate this rule; the README will document it.
