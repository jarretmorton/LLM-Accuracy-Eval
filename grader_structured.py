"""
grader_structured.py — Alternate grader for structured (system-prompted) outputs.

Pairs with system prompts that force the model to end its response with a
terminal block in EXACTLY this format:

  Pre-query:
    === ANSWER ===
    HOME_TEAM: <name or UNKNOWN>
    AWAY_TEAM: <name or UNKNOWN>
    SCORE: <int-int or UNKNOWN>
    STATUS: ANSWERED | REFUSED

  Primary query:
    === ANSWER ===
    VALUE: <single number or UNKNOWN>
    CONFIDENCE_PCT: <integer 0-100 or UNKNOWN>      (optional; may be UNKNOWN)
    STATUS: ANSWERED | REFUSED

This is the "alternate grader" path, selected by `grader.type: structured`
in the YAML spec. It is dramatically simpler than the regex-based numeric
grader because the model has been instructed to commit to a fixed output
shape — no scoring tables, no label heuristics, no range midpoint logic
at the grading layer (the model is told to commit a midpoint, not a range).

Per-run statuses produced (primary query):
  ANSWERED   — terminal block present, VALUE parses (CONFIDENCE_PCT optional)
  REFUSED    — terminal block present, STATUS line says REFUSED
  TRUNCATED  — API stop_reason == "max_tokens"
  MALFORMED  — terminal block missing OR present but VALUE unparseable

Per-topic statuses produced (pre-query):
  COMMITTED  — terminal block present, at least one field non-UNKNOWN
  REFUSED    — terminal block present, all three fields UNKNOWN
  TRUNCATED  — API stop_reason == "max_tokens"
  MALFORMED  — terminal block missing or no expected keys parsed

The pre-query status is intentionally coarse — the *filter* granularity
lives in three derived booleans documented below.

Filter booleans (per topic). Nested: fully ⊆ answered ⊆ partially.

  pre_query_answered           — SCORE field parsed to int-int (strict;
                                 semantically aligned with grader.py's same
                                 field, which uses has_score on free text).
                                 Field-commitment is stricter than numeric's
                                 text-regex; this gap is deliberate.

  pre_query_fully_answered     — both teams identified AND score parsed.
                                 Used by plot 4. Closest structured analog
                                 of numeric grader's pre_query_answered
                                 in practice.

  pre_query_partially_answered — teams identified OR score parsed OR a
                                 score-shaped hedge appears in free text
                                 (has_score match on the response, skipped
                                 if truncated). Used by plot 3. The loosest
                                 filter — catches "I think it was around
                                 25-22" even when the model committed
                                 SCORE: UNKNOWN.

Public API mirrors grader.py:
    grade_results(results_path, spec) -> Path
    generate_plots(graded_path, spec) -> list[Path]

generate_plots emits five plots; shared scatter helper imported from grader.py.
"""

# --- Imports --------------------------

import json
import re
import statistics
from datetime import date
from pathlib import Path

# has_score is the free-text score detector used by the numeric grader.
# Importing it here keeps the structured grader's "hedged score in prose"
# detection bit-for-bit identical to the numeric grader's primary signal.
from grader import has_score


# --- Constants --------------------------

# Find ALL sentinel occurrences; the terminal block is the last one, since the
# system prompt instructs the model to emit it only at the end. Loose anchoring
# (no ^/$ requirement) tolerates the " ".join used in query.chat() to merge
# multiple text blocks from web-search responses.
SENTINEL_RE = re.compile(r"={3,}\s*ANSWER\s*={3,}")

# Status values the model is allowed to emit. TRUNCATED, MALFORMED, and
# COMMITTED are grader-only; the model never emits them.
ALLOWED_MODEL_STATUSES = {"ANSWERED", "REFUSED"}

# Single number: no commas (stripped before match), no ranges, no expressions.
# Negative allowed only for forward-compatibility — current evals are positive.
SINGLE_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

# Score format: <int>-<int>, hyphen or en-dash, optional whitespace. ≤3 digits
# per side so year ranges ("2022-23") can't pass.
SCORE_RE = re.compile(r"^\s*\d{1,3}\s*[-–]\s*\d{1,3}\s*$")

# Keys we expect to see in a pre-query / primary-query terminal block. If none
# of these appear in the parsed dict, the block is malformed (model emitted
# the sentinel but didn't follow the schema).
PRE_QUERY_EXPECTED_KEYS = {"HOME_TEAM", "AWAY_TEAM", "SCORE", "STATUS"}
QUERY_EXPECTED_KEYS = {"VALUE", "CONFIDENCE_PCT", "STATUS"}


# --- Helpers --------------------------

def parse_terminal_block(text):
    """
    Return a dict of KEY -> raw value strings from the LAST '=== ANSWER ==='
    block in `text`, or None if no sentinel is present.

    Parsing is line-based: each non-empty line of the body must be 'KEY: value'.
    Keys are uppercased so downstream lookups are case-insensitive. Lines
    without a colon are silently ignored (tolerates accidental commentary
    inside the block, though the system prompt forbids it).
    """
    matches = list(SENTINEL_RE.finditer(text))
    if not matches:
        return None
    body = text[matches[-1].end():]
    fields = {}
    for line in body.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip().upper()] = value.strip()
    return fields if fields else None


def _is_unknown(value):
    """True if a parsed field value is missing or the literal 'UNKNOWN'."""
    return value is None or value.strip().upper() == "UNKNOWN"


def grade_pre_query(text, stop_reason):
    """
    Parse a pre-query response into a structured dict.

    Returns:
      status:           COMMITTED | REFUSED | TRUNCATED | MALFORMED
      home_team:        team string or None
      away_team:        team string or None
      score:            score string ("25-22") or None
      teams_identified: both home_team and away_team are non-None  (diagnostic)
      score_provided:   score is non-None and matches int-int format (diagnostic)
      score_in_text:    has_score match anywhere in the response text,
                        guarded by truncation. Caught hedges like
                        "I think it was around 25-22" even when SCORE: UNKNOWN.

    Status is COMMITTED if any structured field was non-UNKNOWN, REFUSED
    if all were UNKNOWN. The model's STATUS line is advisory — a model
    claiming ANSWERED with everything UNKNOWN is reclassified as REFUSED.
    Filter-level granularity (fully/answered/partially) lives in the
    summary booleans built from these flags, not in this status string.
    """
    # API-reported truncation is unambiguous. Score-in-text is also
    # suppressed under truncation (mirrors the numeric grader's
    # `(not pre_truncated) and has_score(...)` rule).
    if stop_reason == "max_tokens":
        return {"status": "TRUNCATED", "home_team": None, "away_team": None,
                "score": None, "teams_identified": False,
                "score_provided": False, "score_in_text": False}

    parsed = parse_terminal_block(text)
    if parsed is None:
        # No terminal block. Still scan for a hedged score — a model that
        # ignored the format but mentioned "25-22" in prose has shown
        # partial knowledge.
        return {"status": "MALFORMED", "home_team": None, "away_team": None,
                "score": None, "teams_identified": False,
                "score_provided": False, "score_in_text": has_score(text)}

    # Safety net: block present but none of the expected keys parsed.
    if not (PRE_QUERY_EXPECTED_KEYS & parsed.keys()):
        return {"status": "MALFORMED", "home_team": None, "away_team": None,
                "score": None, "teams_identified": False,
                "score_provided": False, "score_in_text": has_score(text)}

    home = parsed.get("HOME_TEAM")
    away = parsed.get("AWAY_TEAM")
    score = parsed.get("SCORE")

    home_clean = None if _is_unknown(home) else home
    away_clean = None if _is_unknown(away) else away

    score_clean = None
    if not _is_unknown(score) and SCORE_RE.match(score):
        score_clean = score.strip()

    teams_identified = home_clean is not None and away_clean is not None
    score_provided = score_clean is not None

    # Coarse status: anything committed → COMMITTED; nothing committed → REFUSED.
    # Filter granularity is downstream of these structured-field flags.
    if teams_identified or score_provided:
        status = "COMMITTED"
    else:
        status = "REFUSED"

    # Free-text hedge scan. Runs over the full text (committed score in the
    # SCORE field would also trip this, but that's harmless — the boolean is
    # OR'd into partially_answered, so a redundant True doesn't double-count).
    score_in_text = has_score(text)

    return {
        "status": status,
        "home_team": home_clean,
        "away_team": away_clean,
        "score": score_clean,
        "teams_identified": teams_identified,
        "score_provided": score_provided,
        "score_in_text": score_in_text,
    }


def grade_run(text, known_answer, stop_reason):
    """
    Parse a single primary-query run and compute accuracy vs the truth value.

    Returns a per-run dict suitable for the graded JSON's `runs` list:
      status, truncated, malformed, run_refused, extracted, known, exact_match,
      accuracy, confidence.

    Each run is exactly one of TRUNCATED, MALFORMED, REFUSED, or ANSWERED.
    ANSWERED requires only that VALUE parses to a number; CONFIDENCE_PCT may
    be UNKNOWN. The run still contributes to accuracy and stability summaries
    in that case — only the confidence summary is affected.
    """
    out = {
        "status": "MALFORMED",
        "truncated": False,
        "malformed": False,
        "run_refused": False,
        "extracted": None,
        "known": known_answer,
        "exact_match": False,
        "accuracy": None,
        "confidence": None,
    }

    if stop_reason == "max_tokens":
        out["status"] = "TRUNCATED"
        out["truncated"] = True
        return out

    parsed = parse_terminal_block(text)
    if parsed is None:
        out["status"] = "MALFORMED"
        out["malformed"] = True
        return out

    # Safety net: block present but none of the expected keys parsed.
    if not (QUERY_EXPECTED_KEYS & parsed.keys()):
        out["status"] = "MALFORMED"
        out["malformed"] = True
        return out

    value_raw = parsed.get("VALUE")
    conf_raw = parsed.get("CONFIDENCE_PCT")
    raw_status = parsed.get("STATUS", "").upper()

    # Parse VALUE. Strict single-number form; strip commas first so '6,589.33'
    # parses, but anything else (ranges, expressions, '~285', 'about 285')
    # falls through to None.
    extracted = None
    if not _is_unknown(value_raw):
        stripped = value_raw.replace(",", "").strip()
        if SINGLE_NUMBER_RE.match(stripped):
            extracted = float(stripped)

    # Parse CONFIDENCE_PCT. Strict integer 0-100 (trailing '%' tolerated even
    # though the system prompt forbids it — small kindness for a common slip).
    # CONFIDENCE_PCT is optional; missing/UNKNOWN leaves confidence at None
    # and does NOT block an ANSWERED classification.
    confidence = None
    if not _is_unknown(conf_raw):
        stripped = conf_raw.rstrip("%").strip()
        if re.match(r"^\d+$", stripped):
            v = int(stripped)
            if 0 <= v <= 100:
                confidence = v

    if raw_status == "REFUSED":
        out["status"] = "REFUSED"
        out["run_refused"] = True
    elif raw_status == "ANSWERED" and extracted is not None:
        out["status"] = "ANSWERED"
        out["extracted"] = extracted
        out["confidence"] = confidence  # may be None — still ANSWERED
        out["accuracy"] = round(1 - abs(extracted - known_answer) / known_answer, 4)
        out["exact_match"] = extracted == known_answer
    else:
        # Model said ANSWERED but VALUE didn't parse, or status itself was
        # unrecognised. Both count as MALFORMED — can't trust the output.
        out["status"] = "MALFORMED"
        out["malformed"] = True

    return out


def grade_entry(entry, truth_lookup):
    """
    Grade one (model, topic) block from the raw results file.

    Summary field semantics for pre-query filters (nested: fully ⊆ answered ⊆ partially):

      pre_query_answered           — SCORE field parsed (compat with numeric grader,
                                     which uses free-text has_score). The structured
                                     version is stricter by design.
      pre_query_fully_answered     — both teams identified AND score parsed.
      pre_query_partially_answered — teams identified OR score parsed OR has_score
                                     match in free text (suppressed under truncation).

    Additional fields:
      pre_query_status     — COMMITTED | REFUSED | TRUNCATED | MALFORMED
      pre_query_truncated, pre_query_malformed — bools
      teams_identified, score_provided, score_in_text — diagnostic flags that
                                     feed the booleans above
      home_team, away_team, pre_query_score — diagnostic, for human review
      runs_malformed       — count of primary-query runs with no/bad block
    """
    league = entry["league"]
    year = entry["year"]

    known_answer = truth_lookup.get((league, year))
    if known_answer is None:
        raise ValueError(
            f"No truth value in spec for ({league!r}, {year}). "
            f"Results file may have been generated from a different spec."
        )

    pre = grade_pre_query(entry["pre_answer"], entry.get("pre_stop_reason"))

    graded_runs = []
    for run in entry["runs"]:
        gr = grade_run(run["answer"], known_answer, run.get("stop_reason"))
        graded_runs.append({"run": run["run"], **gr})

    # --- Summary metrics --------------------------

    valid_accuracies = [r["accuracy"] for r in graded_runs if r["accuracy"] is not None]
    mean_accuracy = (round(sum(valid_accuracies) / len(valid_accuracies), 4)
                     if valid_accuracies else None)

    valid_extracted = [r["extracted"] for r in graded_runs if r["extracted"] is not None]
    stdev_extracted = (round(statistics.stdev(valid_extracted), 4)
                       if len(valid_extracted) > 1 else None)
    mean_extracted = (round(statistics.mean(valid_extracted), 4)
                      if len(valid_extracted) > 1 else None)
    stability = (round(1 - (stdev_extracted / mean_extracted), 4)
                 if stdev_extracted is not None and mean_extracted else None)

    valid_conf = [r["confidence"] for r in graded_runs if r["confidence"] is not None]
    mean_confidence = round(sum(valid_conf) / len(valid_conf), 1) if valid_conf else None

    runs_truncated = sum(1 for r in graded_runs if r["status"] == "TRUNCATED")
    runs_refused = sum(1 for r in graded_runs if r["status"] == "REFUSED")
    runs_malformed = sum(1 for r in graded_runs if r["status"] == "MALFORMED")
    runs_extracted = len(valid_accuracies)

    # Pre-query filter booleans. Three nested filters as documented above.
    teams = pre["teams_identified"]
    score = pre["score_provided"]
    score_text = pre["score_in_text"]

    summary = {
        # Pre-query filters used by generate_plots:
        "pre_query_answered": score,
        "pre_query_fully_answered": teams and score,
        "pre_query_partially_answered": teams or score or score_text,
        "pre_query_truncated": pre["status"] == "TRUNCATED",
        "pre_query_malformed": pre["status"] == "MALFORMED",
        "pre_query_status": pre["status"],
        # Diagnostic fields (feed the booleans; useful for analysis):
        "teams_identified": teams,
        "score_provided": score,
        "score_in_text": score_text,
        "home_team": pre["home_team"],
        "away_team": pre["away_team"],
        "pre_query_score": pre["score"],
        # Per-run aggregates:
        "runs_graded": len(graded_runs),
        "runs_with_extraction": runs_extracted,
        "runs_with_refusals": runs_refused,
        "runs_truncated": runs_truncated,
        "runs_malformed": runs_malformed,
        "all_runs_accounted_for": (
            runs_extracted + runs_refused + runs_truncated + runs_malformed
        ) == len(graded_runs),
        # Score metrics:
        "mean_confidence": mean_confidence,
        "mean_accuracy_of_extracted": mean_accuracy,
        "stability_of_extracted": stability,
    }

    out = {
        "model": entry["model"],
        "league": league,
        "year": year,
        "pre_query": entry["pre_query"],
        "pre_answer": entry["pre_answer"],
        "query": entry["query"],
        "n": entry["n"],
        "summary": summary,
        "runs": graded_runs,
    }
    if "pre_stop_reason" in entry:
        out["pre_stop_reason"] = entry["pre_stop_reason"]
    return out


# --- Public API (grading) --------------------------

def grade_results(results_path, spec) -> Path:
    """
    Grade a raw results file produced by an eval that used structured system
    prompts. Same signature, return type, and on-disk shape as
    grader.grade_results — drop-in alternative selected by spec.grader.type.

    No refusal_patterns are consulted; refusals are identified by the STATUS
    line in the terminal block. spec.grader.refusal_patterns may still exist
    in the YAML (it's still required by spec.py) but is ignored by this grader.
    """
    results_path = Path(results_path)
    with open(results_path) as f:
        raw = json.load(f)

    if raw.get("spec_name") != spec.name:
        print(f"  Note: results file spec_name ({raw.get('spec_name')!r}) "
              f"does not match passed spec ({spec.name!r}). Continuing.")

    truth_lookup = {(t["league"], t["year"]): t["truth"] for t in spec.topics}

    graded_entries = []
    for entry in raw["results"]:
        print(f"  Grading [{entry['model']}] {entry['league']} {entry['year']}...")
        ge = grade_entry(entry, truth_lookup)
        s = ge["summary"]
        print(
            f"    pre_status={s['pre_query_status']}  "
            f"answered={s['pre_query_answered']}  "
            f"full={s['pre_query_fully_answered']}  "
            f"partial={s['pre_query_partially_answered']}  "
            f"extracted={s['runs_with_extraction']}/{s['runs_graded']}  "
            f"refused={s['runs_with_refusals']}  "
            f"truncated={s['runs_truncated']}  "
            f"malformed={s['runs_malformed']}  "
            f"accuracy={s['mean_accuracy_of_extracted']}  "
            f"stability={s['stability_of_extracted']}"
        )
        if not s["all_runs_accounted_for"]:
            print("    WARNING: run states do not sum to runs_graded")
        graded_entries.append(ge)

    output = {
        "spec_name": raw.get("spec_name"),
        "spec_version": raw.get("spec_version"),
        "run_date": raw.get("run_date"),
        "graded_date": str(date.today()),
        "grader_kind": "structured",
        "results": graded_entries,
    }

    graded_path = results_path.with_name(
        results_path.stem + "_graded" + results_path.suffix
    )
    with open(graded_path, "w") as f:
        json.dump(output, f, indent=2)
    return graded_path


# --- Public API (plotting) --------------------------

def generate_plots(graded_path, spec, output_dir=None):
    """
    Generate five accuracy plots from a structured-grader graded results file.

    Plots:
      1. accuracy_vs_confidence                       — no filter
      2. accuracy_vs_stability                        — no filter
      3. accuracy_vs_stability_partially_answered     — filter: pre_query_partially_answered
                                                        (teams OR score field OR has_score
                                                        in free text)
      4. accuracy_vs_stability_fully_answered         — filter: pre_query_fully_answered
                                                        (teams AND score field)
      5. accuracy_vs_confidence_partially_answered    — filter: pre_query_partially_answered
                                                        (confidence-axis counterpart of plot 3)

    Plot 4's data is a strict subset of plot 3's. Plot 4 is semantically what
    grader.py's existing _filtered plot computes (a committed score implies
    teams were identified in practice). Plot 3 is the looser filter and
    additionally catches hedges like "I think it was around 25-22" in
    reasoning text — useful for testing whether weak commitment correlates
    with stability/accuracy independently of strong commitment.

    Each plot has scatter colored by model with per-model and combined linear
    trends. The shared scatter helper is imported from grader.py so plot
    styling stays identical across the two grader paths.

    Parameters mirror grader.generate_plots; return value is the list of
    written PNG paths in plot order.
    """
    # Lazy import — module loads without matplotlib until plots are requested.
    from grader import _ensure_plotting_deps, _plot_scatter_with_trends

    plt, np = _ensure_plotting_deps()

    graded_path = Path(graded_path)
    with open(graded_path) as f:
        graded = json.load(f)

    if output_dir is None:
        output_dir = graded_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Five parallel dicts: model -> [(x, y), ...]. Each plot projects the
    # same source entries onto different (x, y) pairs and/or filters.
    confidence_data = {}
    confidence_partial = {}  # pre_query_partially_answered (loosest filter)
    stability_data = {}
    stability_partial = {}   # pre_query_partially_answered (loosest filter)
    stability_full = {}      # pre_query_fully_answered (strictest filter)

    for entry in graded["results"]:
        model = entry["model"]
        s = entry["summary"]

        confidence_data.setdefault(model, []).append(
            (s["mean_confidence"], s["mean_accuracy_of_extracted"])
        )
        stability_data.setdefault(model, []).append(
            (s["stability_of_extracted"], s["mean_accuracy_of_extracted"])
        )

        if s["pre_query_partially_answered"]:
            confidence_partial.setdefault(model, []).append(
                (s["mean_confidence"], s["mean_accuracy_of_extracted"])
            )
            stability_partial.setdefault(model, []).append(
                (s["stability_of_extracted"], s["mean_accuracy_of_extracted"])
            )
        if s["pre_query_fully_answered"]:
            stability_full.setdefault(model, []).append(
                (s["stability_of_extracted"], s["mean_accuracy_of_extracted"])
            )

    eval_name = spec.name
    paths = []

    # Derive a filename prefix from the graded JSON stem so each plot is
    # clearly tied to the spec that produced it.
    stem = graded_path.stem
    file_prefix = stem[: -len("_graded")] if stem.endswith("_graded") else stem

    p1 = output_dir / f"{file_prefix}_accuracy_vs_confidence.png"
    _plot_scatter_with_trends(
        confidence_data,
        x_label="Mean stated confidence (%)",
        y_label="Mean accuracy of extracted answers",
        title=f"{eval_name}: Accuracy vs Confidence",
        output_path=p1, plt=plt, np=np,
    )
    paths.append(p1)

    p2 = output_dir / f"{file_prefix}_accuracy_vs_stability.png"
    _plot_scatter_with_trends(
        stability_data,
        x_label="Stability of extracted answers (1 - stdev/mean)",
        y_label="Mean accuracy of extracted answers",
        title=f"{eval_name}: Accuracy vs Stability (all topics)",
        output_path=p2, plt=plt, np=np,
    )
    paths.append(p2)

    p3 = output_dir / f"{file_prefix}_accuracy_vs_stability_partially_answered.png"
    _plot_scatter_with_trends(
        stability_partial,
        x_label="Stability of extracted answers (1 - stdev/mean)",
        y_label="Mean accuracy of extracted answers",
        title=f"{eval_name}: Accuracy vs Stability (pre-query partially answered)",
        output_path=p3, plt=plt, np=np,
    )
    paths.append(p3)

    p4 = output_dir / f"{file_prefix}_accuracy_vs_stability_fully_answered.png"
    _plot_scatter_with_trends(
        stability_full,
        x_label="Stability of extracted answers (1 - stdev/mean)",
        y_label="Mean accuracy of extracted answers",
        title=f"{eval_name}: Accuracy vs Stability (pre-query fully answered)",
        output_path=p4, plt=plt, np=np,
    )
    paths.append(p4)

    p5 = output_dir / f"{file_prefix}_accuracy_vs_confidence_partially_answered.png"
    _plot_scatter_with_trends(
        confidence_partial,
        x_label="Mean stated confidence (%)",
        y_label="Mean accuracy of extracted answers",
        title=f"{eval_name}: Accuracy vs Confidence (pre-query partially answered)",
        output_path=p5, plt=plt, np=np,
    )
    paths.append(p5)

    return paths
