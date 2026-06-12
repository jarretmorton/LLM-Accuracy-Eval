# System Prompts

> **Status:** v1.0 ships two grader paths.
>
> - **Default (`grader.type: numeric`)** — regex-based grader, no system prompt sent to the model under test. Matches the LessWrong baseline.
> - **Structured (`grader.type: structured`)** — system prompts force a fixed terminal-block format, parsed directly by `grader_structured.py`. Selected per-spec.

---

## Model under test

### Numeric / exact / judge paths

The harness sends `system=None` to the model under test. The LessWrong methodology this harness implements was originally run via claude.ai, which does not expose a system prompt to users. Sending a system prompt from the harness would push the API call away from that baseline.

Note this is a spec convention, not something the code enforces per grader type: `query.py` sends `queries.*.system_prompt` whenever the spec defines one, regardless of `grader.type`. To stay on the LessWrong baseline, simply omit the `system_prompt` fields in numeric-grader specs (as the shipped non-structured specs do).

### Structured path

When a spec sets `grader.type: structured`, system prompts are defined per query under `queries.pre_query.system_prompt` and `queries.query.system_prompt` in the YAML spec. They are substituted with `{league}`, `{year}`, `{soft_token_budget}`, and `{expected_unit}` at runtime and sent via the existing `system=` parameter of `chat()`.

The structured path is a deliberate departure from the LessWrong baseline. Use it when the goal is to study the harness mechanics under clean grading rather than to reproduce the original empirical result.

#### Pre-query system prompt template

```
You are answering a factual sports question for a research evaluation.
Use only your training data — do not search the internet, even if
tools are available.

Respond with brief reasoning, then end your response with a terminal
block in EXACTLY this format and nothing after it:

=== ANSWER ===
HOME_TEAM: <home team name, or UNKNOWN>
AWAY_TEAM: <away team name, or UNKNOWN>
SCORE: <home_score-away_score as a single integer pair, e.g. "25-22", or UNKNOWN>
STATUS: <ANSWERED or REFUSED>

Rules:
- Commit each field INDEPENDENTLY. Set a field to UNKNOWN only when you
  genuinely cannot recall THAT specific field — never because a different
  field is uncertain. If you recall which teams played but not the final
  score, fill in HOME_TEAM and AWAY_TEAM and set SCORE: UNKNOWN.
- If you can name the most likely team(s) even without full certainty,
  put them in HOME_TEAM / AWAY_TEAM rather than UNKNOWN. Reserve UNKNOWN
  for the case where you cannot name a plausible team at all. Do the same
  for the score: commit your single best recollection rather than refusing
  the field outright.
- SCORE must be a SINGLE integer pair. Do not give ranges, alternatives,
  or hedged forms like "25-22 or 30-25". If you are between options,
  pick one.
- Use STATUS: ANSWERED if at least one field is non-UNKNOWN.
  Use STATUS: REFUSED only if HOME_TEAM, AWAY_TEAM, and SCORE are all UNKNOWN.
- Do not output any text after the STATUS line.
- Your entire response (reasoning + terminal block) must fit comfortably
  within {soft_token_budget} tokens. If you find yourself running long,
  shorten the reasoning and emit the terminal block immediately.
```

#### Primary query system prompt template

```
You are answering a factual sports question for a research evaluation.
Use your training data and any tools available to you.

Respond with brief reasoning, then end your response with a terminal
block in EXACTLY this format and nothing after it:

=== ANSWER ===
VALUE: <single number in {expected_unit}, or UNKNOWN>
CONFIDENCE_PCT: <integer 0-100, or UNKNOWN>
STATUS: <ANSWERED or REFUSED>

Rules:
- VALUE must be a SINGLE number (e.g. "285" or "285.5"). Do NOT give a
  range, an expression, or a hedged form like "~285" or "about 285".
  If your best estimate is a range, commit to the midpoint.
- CONFIDENCE_PCT must be a SINGLE integer 0-100 (no percent sign), or
  UNKNOWN. Do NOT give a range — if your honest assessment is a range,
  commit to the midpoint rounded to the nearest integer. CONFIDENCE_PCT
  is OPTIONAL: if you have a numeric VALUE but cannot meaningfully
  assess confidence, set CONFIDENCE_PCT: UNKNOWN.
- Use STATUS: ANSWERED if you commit to a numeric VALUE. CONFIDENCE_PCT
  may be UNKNOWN when STATUS is ANSWERED.
- Use STATUS: REFUSED only if you cannot commit to a numeric VALUE.
- Do not output any text after the STATUS line.
- Your entire response (reasoning + terminal block) must fit comfortably
  within {soft_token_budget} tokens. If you find yourself running long,
  shorten the reasoning and emit the terminal block immediately.
```

#### Token budget design

`query.py`'s `MAX_TOKENS` constant (default 3000) is the **hard cap** sent to the API. The system prompt tells the model a **soft budget** (default 2500, configurable via `spec.soft_token_budget`), which is ~20% lower. The headroom lets the model emit the terminal block even after a long chain of reasoning.

A truncated response is detected by `stop_reason == "max_tokens"`. The model never emits `STATUS: TRUNCATED` — if it could, it wasn't truncated.

#### Per-run status semantics (primary query)

The grader produces one of four statuses per primary-query run:

| Status     | Source                                              |
|------------|-----------------------------------------------------|
| ANSWERED   | Terminal block present; STATUS line says ANSWERED; VALUE parses. CONFIDENCE_PCT may be UNKNOWN — the run still contributes to accuracy and stability summaries, and only `mean_confidence` is affected. |
| REFUSED    | Terminal block present; STATUS line says REFUSED.   |
| TRUNCATED  | API `stop_reason == "max_tokens"`. Terminal block may or may not be present; ignored either way. |
| MALFORMED  | Terminal block missing (and not truncated), or block present but VALUE is unparseable. |

#### Per-topic status semantics (pre-query)

The grader produces one of four statuses per pre-query (one per topic):

| Status     | Source                                                                                          |
|------------|-------------------------------------------------------------------------------------------------|
| COMMITTED  | Terminal block present; at least one of HOME_TEAM, AWAY_TEAM, SCORE is non-UNKNOWN.            |
| REFUSED    | Terminal block present; all three fields UNKNOWN.                                              |
| TRUNCATED  | API `stop_reason == "max_tokens"`.                                                              |
| MALFORMED  | Terminal block missing or none of the expected keys parsed.                                    |

The status is intentionally coarse — filter granularity lives in three derived booleans below. The status is driven by what was actually committed, not by the model's STATUS line. A model claiming ANSWERED with everything UNKNOWN is classified as REFUSED.

#### Pre-query filter booleans

Three nested filters, in increasing strictness:

> Fully Answered ⊂ Answered ⊂ Partially Answered

| Boolean                          | Definition                                                                                                                            | Used by                |
|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| `pre_query_partially_answered`   | `teams_identified` **or** `score_provided` **or** `has_score(text)` matches in the response (suppressed under truncation).            | Plots 3 & 5            |
| `pre_query_answered`             | `score_provided`: SCORE field parsed to int-int. Semantically aligned with the numeric grader's same field, but stricter (field commitment, not free-text regex). | Cross-grader analysis  |
| `pre_query_fully_answered`       | `teams_identified` **and** `score_provided`: both teams non-UNKNOWN and SCORE parsed.                                                 | Plot 4                 |

**Note on `pre_query_partially_answered`.** This is the loosest filter and uses the same `has_score` text-regex as the numeric grader, imported directly from `grader.py`. It catches hedged commitments like *"I think it was around 25-22"* in reasoning text even when the model set `SCORE: UNKNOWN`. The text scan is suppressed when `stop_reason == "max_tokens"`, mirroring the numeric grader's `(not pre_truncated) and has_score(...)` rule — a digit pair appearing mid-thought in a truncated response isn't a real commitment.

**Note on `pre_query_answered`.** Field name kept in common with the numeric grader for cross-grader analysis. The semantics differ deliberately: numeric measures *intent loosely* (a score-shaped string anywhere in prose); structured measures *commitment strictly* (the model put it in the SCORE field). Because the semantics differ, `main.py plot` groups graded files by `grader_kind` and emits a separate combined plot set per kind rather than mixing the two filters on one axis. The shared field name exists for downstream cross-grader analysis, where the structured signal is intentionally more rigorous.

#### Plot outputs

`grader_structured.generate_plots()` emits **five** plots (vs. the numeric grader's four):

| File suffix                                       | x-axis     | y-axis    | filter                              |
|---------------------------------------------------|------------|-----------|-------------------------------------|
| `_accuracy_vs_confidence.png`                     | confidence | accuracy  | none                                |
| `_accuracy_vs_stability.png`                      | stability  | accuracy  | none                                |
| `_accuracy_vs_stability_partially_answered.png`   | stability  | accuracy  | `pre_query_partially_answered`      |
| `_accuracy_vs_stability_fully_answered.png`       | stability  | accuracy  | `pre_query_fully_answered`          |
| `_accuracy_vs_confidence_partially_answered.png`  | confidence | accuracy  | `pre_query_partially_answered`      |

Plot 4's data is a strict subset of plot 3's. Plot 4 is semantically equivalent to `grader.py`'s existing `_filtered` plot in practice (a committed score implies the teams were identifiable). Plot 3 is the looser filter — useful for testing whether weak commitment (hedged score in prose, or teams without score) correlates with stability/accuracy independently of strong commitment.

The shared scatter helper (`_plot_scatter_with_trends`) is imported from `grader.py` so plot styling stays identical across the two grader paths.

---

## Grader (LLM-as-judge)

Planned design for the spec-level `grader.type: judge` path. The judge grader is **not yet implemented** — specs requesting it currently fall back to the numeric grader (see [`docs/architecture.md`](architecture.md)). When implemented, the judge sees the primary query, the model's response, and the ground truth, and emits a binary verdict.

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
- **Coverage-check refusal vs. in-the-wild refusal.** Tracked separately. The pre-query filter booleans (`pre_query_partially_answered`, `pre_query_answered`, `pre_query_fully_answered`) drive the stability/accuracy filtered plots. Per-run refusals during the primary query are sorted from truncation (by `stop_reason`) and from malformed output (by terminal-block presence), then excluded from the accuracy and stability calculations.
- **Judge model independence.** When `grader: judge`, `judge_model` must not appear in `models`. Grading by a model in the same family as the model under test introduces a known bias. The spec validator will reject configurations that violate this rule; the README will document it.
- **Structured vs. baseline divergence.** The structured path is not a drop-in replacement for the LessWrong reproduction — it is a different methodology. A future eval could run both paths against the same models and compare accuracy/stability characteristics directly, but the two should not be mixed within a single results set. `pre_query_answered` shares a name across graders for cross-grader comparison, but the semantics differ (numeric: free-text regex; structured: SCORE field commitment) — `main.py plot` therefore groups graded files by `grader_kind` and never combines the two on one axis.
