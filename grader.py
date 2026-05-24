"""
grader.py — Grade raw eval results against the truth values in the spec.

For each (model, topic) block in the raw results file, compute:
  - Extracted numeric answer per run (with unit awareness)
  - Per-run accuracy vs the spec's truth value
  - Mean accuracy across runs (excluding non-extractions)
  - Standard deviation and mean of extracted values
  - Stability (1 - stdev/mean of extracted values)
  - Per-run refusal flag (from spec.grader.refusal_patterns)
  - Pre-query refusal flag
  - Mean confidence across runs that produced both an answer and a confidence

Writes a single consolidated graded JSON file alongside the source, with
the same envelope as the raw file but each entry's `runs` replaced by
graded versions and a `summary` block added.

Public API:
    grade_results(results_path: Path, spec) -> Path
"""

# --- Imports --------------------------

import json
import re
import statistics
from datetime import date
from pathlib import Path


# --- Helpers --------------------------

def is_refusal(text, patterns):
    """
    Return True if any refusal pattern matches anywhere in `text`.

    Patterns come from the spec, not a module-level list — this lets each
    eval define its own refusal vocabulary without touching the grader.
    Case-insensitive matching; patterns are passed to re.search so regex
    syntax is supported (escape literals with re.escape if needed).
    """
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def extract_confidence(text):
    """
    Extract a confidence percentage from the response text.

    Looks for a range first (e.g., "25-30%") and returns the midpoint,
    then falls back to a single value (e.g., "Confidence Level: 15%").
    Returns None if no percentage is found.
    """
    # Try range first (e.g., "25-30%") — return the midpoint.
    range_match = re.search(r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%', text)
    if range_match:
        low, high = float(range_match.group(1)), float(range_match.group(2))
        return round((low + high) / 2, 1)
    # Fall back to single value (e.g., "Confidence Level: 15%").
    single_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if single_match:
        return float(single_match.group(1))
    return None


def extract_last_number(text, unit=None):
    """
    Extract the last numeric value from `text`, optionally requiring it to
    be paired with `unit` (e.g., "1,400 hours").

    Strategy:
      - Strip percentage values first so confidence numbers aren't matched.
      - If `unit` is given, prefer a range paired with the unit (return
        the midpoint), then fall back to the last single number paired
        with the unit.
      - If `unit` is None, return the last number found anywhere.

    Returns None if no matching number is found.
    """
    # Strip percentage values (e.g., "15%", "25-30%") so confidence
    # numbers are never matched as the answer.
    text = re.sub(r'\d+(?:\.\d+)?\s*(?:[-–]\s*\d+(?:\.\d+)?)?\s*%', '', text)

    if unit:
        # Check for a range paired with the unit (e.g., "778-788 hours") —
        # return the average of the bounds.
        range_matches = re.findall(
            rf'\b(\d+(?:,\d+)*(?:\.\d+)?)\s*[-–]\s*(\d+(?:,\d+)*(?:\.\d+)?)\s+{unit}s?\b',
            text, re.IGNORECASE
        )
        if range_matches:
            low, high = range_matches[-1]
            return round((float(low.replace(",", "")) + float(high.replace(",", ""))) / 2, 4)
        # Numbers immediately followed by the unit word (singular or plural).
        matches = re.findall(
            rf'\b(\d+(?:,\d+)*(?:\.\d+)?)\s+{unit}s?\b',
            text, re.IGNORECASE
        )
    else:
        matches = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text)

    if not matches:
        return None
    # Remove commas from numbers like "1,400" before converting to float.
    return float(matches[-1].replace(",", ""))


def grade_run(answer_text, known_answer, unit=None):
    """
    Grade a single run: extract the answer and compute accuracy vs truth.

    Returns a dict with:
      - extracted:   the parsed numeric value, or None if extraction failed
      - known:       the ground truth value (kept in the output for reference)
      - exact_match: True iff extracted == known
      - accuracy:    1 - abs(extracted - known) / known
                     (1.0 = perfect, 0.0 = 100% off, negative = worse than 100%)
                     None if extraction failed
    """
    extracted = extract_last_number(answer_text, unit)

    if extracted is None:
        return {
            "extracted": None,
            "known": known_answer,
            "exact_match": False,
            "accuracy": None,
        }

    accuracy = round(1 - abs(extracted - known_answer) / known_answer, 4)

    return {
        "extracted": extracted,
        "known": known_answer,
        "exact_match": extracted == known_answer,
        "accuracy": accuracy,
    }


def grade_entry(entry, truth_lookup, unit, refusal_patterns):
    """
    Grade one (model, topic) entry from the results file.

    `entry` is one element from the raw results['results'] list — has
    keys: model, league, year, pre_query, pre_answer, query, n, runs.

    Returns a new entry dict with `summary` added and `runs` replaced by
    their graded versions. Preserves the rest of the entry's metadata so
    the graded file is self-identifying.
    """
    league = entry["league"]
    year = entry["year"]

    # Look up the truth value for this (league, year) from the spec's
    # topics. If missing, the results file references a topic the spec
    # doesn't have — almost certainly a spec/results mismatch.
    known_answer = truth_lookup.get((league, year))
    if known_answer is None:
        raise ValueError(
            f"No truth value in spec for ({league!r}, {year}). "
            f"Results file may have been generated from a different spec."
        )

    # Pre-query: True if the model attempted an answer, False if it matched
    # any refusal pattern. This is the coverage_check signal — used by the
    # downstream analysis to filter out uncovered topics.
    pre_query_answered = not is_refusal(entry["pre_answer"], refusal_patterns)

    # Grade each individual run.
    graded_runs = []
    for run in entry["runs"]:
        graded = grade_run(run["answer"], known_answer, unit)
        # run_refused is True whenever a refusal phrase is found, regardless
        # of whether a number was extracted. By design — a refused-but-
        # extracted run gets flagged by the all_runs_accounted_for check
        # in the summary below.
        run_refused = is_refusal(run["answer"], refusal_patterns)
        # Confidence is null when no number was extracted — a stated
        # confidence without an answer isn't meaningful.
        confidence = extract_confidence(run["answer"]) if graded["extracted"] is not None else None
        graded_runs.append({
            "run": run["run"],
            "run_refused": run_refused,
            **{k: v for k, v in graded.items() if k != "accuracy"},
            "confidence": confidence,
            "accuracy": graded["accuracy"],
        })

    # --- Summary metrics --------------------------

    # Accuracies from runs where extraction succeeded.
    valid_accuracies = [r["accuracy"] for r in graded_runs if r["accuracy"] is not None]
    mean_accuracy_of_extracted = (
        round(sum(valid_accuracies) / len(valid_accuracies), 4)
        if valid_accuracies else None
    )

    # Raw extracted numbers from runs where extraction succeeded.
    valid_extracted = [r["extracted"] for r in graded_runs if r["extracted"] is not None]
    # statistics.stdev requires at least 2 samples.
    stdev_of_extracted = (
        round(statistics.stdev(valid_extracted), 4)
        if len(valid_extracted) > 1 else None
    )
    mean_of_extracted = (
        round(statistics.mean(valid_extracted), 4)
        if len(valid_extracted) > 1 else None
    )
    # Stability per the LessWrong post: 1 - (stdev/mean of extracted values).
    # 1.0 = perfectly consistent, lower = more variance relative to mean.
    stability_of_extracted = (
        round(1 - (stdev_of_extracted / mean_of_extracted), 4)
        if stdev_of_extracted is not None and mean_of_extracted else None
    )

    runs_with_refusals = sum(1 for r in graded_runs if r["run_refused"])

    valid_confidences = [r["confidence"] for r in graded_runs if r["confidence"] is not None]
    mean_confidence = (
        round(sum(valid_confidences) / len(valid_confidences), 1)
        if valid_confidences else None
    )

    summary = {
        "pre_query_answered": pre_query_answered,
        "runs_graded": len(graded_runs),
        "runs_with_extraction": len(valid_accuracies),
        "runs_with_refusals": runs_with_refusals,
        "all_runs_accounted_for": (len(valid_accuracies) + runs_with_refusals) == len(graded_runs),
        "mean_confidence": mean_confidence,
        "mean_accuracy_of_extracted": mean_accuracy_of_extracted,
        "stability_of_extracted": stability_of_extracted,
    }

    # Return a new entry mirroring the input shape with summary + graded runs.
    return {
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


# --- Public API --------------------------

def grade_results(results_path, spec) -> Path:
    """
    Grade a raw results file against the spec.

    Reads the consolidated raw results produced by query.py's run_harness,
    grades every (model, topic) entry inside, and writes a single graded
    JSON file alongside the source with `_graded` appended to the stem.

    Parameters
    ----------
    results_path : str | Path
        Path to the raw results JSON file.
    spec : Spec
        Loaded spec from spec.py — provides truth values via spec.topics
        and grading config via spec.grader.

    Returns
    -------
    Path
        Path to the written graded JSON file.

    Raises
    ------
    FileNotFoundError: results_path doesn't exist
    ValueError:        results file references a (league, year) absent from spec.topics
    """
    results_path = Path(results_path)

    with open(results_path) as f:
        raw_results = json.load(f)

    # Soft check: warn if the results were produced from a different spec.
    # Don't error — a user may legitimately regrade old results with an
    # updated spec — but flag it so unexpected mismatches surface visibly.
    if raw_results.get("spec_name") != spec.name:
        print(
            f"  Note: results file spec_name "
            f"({raw_results.get('spec_name')!r}) "
            f"does not match passed spec ({spec.name!r}). Continuing."
        )

    # Build a (league, year) → truth lookup from the spec's topics. O(1)
    # per entry instead of O(n) scan; matters as topic counts grow.
    truth_lookup = {
        (topic["league"], topic["year"]): topic["truth"]
        for topic in spec.topics
    }

    # The unit used for numeric extraction. Strip trailing "s" so the regex
    # pattern \b{unit}s?\b matches both singular and plural ("hours" → "hour",
    # then regex matches "hour" or "hours"). This is a v1.0 simplification —
    # works fine for "hours", "minutes", "dollars", etc. Edge case: a unit
    # that genuinely ends in "s" in singular form would be over-trimmed.
    unit = spec.grader.expected_unit.rstrip("s")

    # Grade each (model, topic) entry. Each gets a per-entry one-line
    # summary printed so the user can see progress without scrolling.
    graded_entries = []
    for entry in raw_results["results"]:
        print(f"  Grading [{entry['model']}] {entry['league']} {entry['year']}...")
        graded_entry = grade_entry(
            entry, truth_lookup, unit, spec.grader.refusal_patterns
        )
        s = graded_entry["summary"]
        print(
            f"    pre_answered={s['pre_query_answered']}  "
            f"extracted={s['runs_with_extraction']}/{s['runs_graded']}  "
            f"refused={s['runs_with_refusals']}  "
            f"accuracy={s['mean_accuracy_of_extracted']}  "
            f"stability={s['stability_of_extracted']}"
        )
        if not s["all_runs_accounted_for"]:
            print(
                f"    WARNING: runs_with_extraction + runs_with_refusals "
                f"!= runs_graded"
            )
        graded_entries.append(graded_entry)

    # Compose the output envelope, preserving metadata from the raw file
    # so the graded file is self-identifying without cross-referencing.
    output = {
        "spec_name": raw_results.get("spec_name"),
        "spec_version": raw_results.get("spec_version"),
        "run_date": raw_results.get("run_date"),
        "graded_date": str(date.today()),
        "results": graded_entries,
    }

    # Derive the graded filename: foo/example.json → foo/example_graded.json.
    # with_name works across all Python versions; with_stem (3.9+) would
    # be slightly cleaner but adds a version floor for no real benefit.
    graded_path = results_path.with_name(
        results_path.stem + "_graded" + results_path.suffix
    )

    with open(graded_path, "w") as f:
        json.dump(output, f, indent=2)

    return graded_path