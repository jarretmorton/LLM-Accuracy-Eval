"""
grader.py — Grade raw eval results against the truth values in the spec.

For each (model, topic) block in the raw results file, compute:
  - Extracted numeric answer per run (with unit awareness)
  - Per-run accuracy vs the spec's truth value
  - Mean accuracy across runs (excluding non-extractions)
  - Standard deviation and mean of extracted values
  - Stability (1 - stdev/mean of extracted values)
  - Per-run refusal flag and the matched refusal pattern (from spec.grader.refusal_patterns)
  - Pre-query answered flag (True only if the pre-answer contains a recognisable game score)
  - Mean confidence across runs that produced both an answer and a confidence

Writes a single consolidated graded JSON file alongside the source, with
the same envelope as the raw file but each entry's `runs` replaced by
graded versions and a `summary` block added.

Public API:
    grade_results(results_path: Path, spec) -> Path
    generate_plots(graded_path: Path, spec, output_dir: Path | None) -> list[Path]
"""

# --- Imports --------------------------

import json
import re
import statistics
from datetime import date
from pathlib import Path

# Plotting dependencies are lazy-loaded inside generate_plots() rather than
# at module-level, so importing grader.py is cheap and `grade` works on
# machines without matplotlib. The actual import happens at first plot call.
import importlib.util
import subprocess
import sys

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


def find_refusal_pattern(text, patterns):
    """
    Return the first refusal pattern that matches `text`, or None.

    Same matching logic as is_refusal — case-insensitive re.search — but
    returns the pattern string itself so callers can record which pattern
    triggered the refusal flag rather than just that one did.
    """
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return p
    return None


def has_score(text):
    """True if `text` contains a final-game-score commitment.

    Recognizes two forms:
      A) N-N / N–N (hyphen or en-dash) — both numbers ≤ 3 digits, so
         year ranges like "2022-23" cannot trigger a match.
      B) N, <team-words> N — comma-separated form with the second team
         name (1–5 words) between the comma and the closing number.
    """
    if re.search(r'\b\d{1,3}\s*[-–]\s*\d{1,3}\b', text):
        return True
    if re.search(r'\b\d{1,3},\s+\w+(?:\s+\w+){0,4}\s+\d{1,3}\b', text):
        return True
    return False


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


def is_truncated(answer_text, stop_reason=None):
    """
    Return True if `answer_text` looks like a cut-off response.

    Two-tier detection:
      1. Definitive: if `stop_reason` is provided and equals "max_tokens",
         the API itself reports that generation was cut off. This is the
         gold-standard signal and overrides any text inspection.
      2. Heuristic fallback (used when stop_reason is None — e.g. legacy
         data from before query.py was updated to capture it): strip trailing
         whitespace and markdown emphasis markers, then check whether the
         resulting last character is a completion signal. Completion signals
         are sentence terminators, closing brackets, closing quotes, and '%'.

    The heuristic catches mid-sentence, mid-table, mid-equation, mid-list,
    and mid-word truncations (~98% recall on the 120-run validation set)
    but misses cases where the model happens to finish a sentence before
    hitting the cap without producing its final answer. Prefer stop_reason
    whenever it's available.
    """
    # Tier 1: API-reported truncation is unambiguous.
    if stop_reason == "max_tokens":
        return True
    # If stop_reason is provided and is anything else ("end_turn",
    # "stop_sequence", "tool_use"), the API says the generation completed
    # normally. Trust it and skip the heuristic — the heuristic exists for
    # legacy data, not as a second-guess on confirmed completions.
    if stop_reason is not None:
        return False

    # Tier 2 (legacy data): heuristic over the text itself.
    stripped = answer_text.rstrip()
    if not stripped:
        return True
    # A trailing fully-closed markdown bold span (e.g. "...**285 hours**")
    # signals a finished commitment even when no sentence terminator follows.
    # We require the pair to be intact: a dangling opening "**" without a
    # close is itself a sign of truncation.
    if re.search(r"\*\*[^*\n]+\*\*$", stripped):
        return False
    # Otherwise: the last visible character must be a sentence terminator,
    # closing bracket/quote, or a percent sign (for "Confidence: 80%" style
    # endings). Anything else — letters, digits, operators, table pipes,
    # commas, continuation markers — suggests cut-off mid-generation.
    return stripped[-1] not in ".!?\")]'%"


def extract_answer(text, unit=None):
    """
    Extract the model's committed numeric answer using a scoring system over
    all `<num> <unit>` matches in the text, where score reflects the strength
    of "this is the model's committed answer" markers.

    Scoring:
    +4 if inside markdown bold AND the bold span contains a label keyword
        ('total' / 'grand total' / 'answer' / 'final answer')
    +3 if on the same line as a label keyword AND inside bold
    +2 if on the same line as a label keyword (no bold required)
    +2 if inside markdown bold (but no label nearby)
    +1 if immediately preceded by '=' or '≈'
    0  otherwise (plain mention with no commitment signal)

    Selection: take the highest-scoring tier; within that tier, take the LAST
    candidate. Rationale: when the model gives an intermediate "Total: X" early
    and refines to a final "Estimated Total: Y" later, Y is the committed
    answer. Hedged alternatives rarely tie at the top tier because they lack
    the strong markers (bold-with-label or "Grand total:" prefix) of a primary
    commitment.

    Cross-tier filter: any '<num> <unit>' immediately followed by 'per <word>'
    is a rate and is skipped — handles 'X hours per match' etc.

    Range handling: if the winning match is the high end of a '<low>-<high>
    <unit>' or '<low> to <high> <unit>' range, the midpoint is returned.
    """
    # Strip percentage values first so confidence numbers (e.g. "70%") are
    # never matched as the numeric answer.
    text = re.sub(r"\d+(?:\.\d+)?\s*(?:[-–]\s*\d+(?:\.\d+)?)?\s*%", "", text)

    num_pat = r"(\d+(?:,\d+)*(?:\.\d+)?)"
    unit_pat = rf"{unit}s?"

    def to_float(s):
        return float(s.replace(",", ""))

    # --- Precompute structural info --------------------------

    # Line boundaries — needed for "same line as label" scoring.
    line_starts = [0] + [m.end() for m in re.finditer(r"\n", text)]

    def line_bounds(pos):
        """(start, end) of the line containing `pos`."""
        line_start = max((s for s in line_starts if s <= pos), default=0)
        line_end_m = re.search(r"\n", text[pos:])
        line_end = pos + line_end_m.start() if line_end_m else len(text)
        # actual line end starts after pos; widen to whole line
        end_from_start = re.search(r"\n", text[line_start:])
        if end_from_start:
            return line_start, line_start + end_from_start.start()
        return line_start, len(text)

    # Bold spans — markdown **...**. Track each span as (start, end).
    bold_spans = []
    for m in re.finditer(r"\*\*([^*\n]+?)\*\*", text):
        bold_spans.append((m.start(), m.end()))

    def in_bold(pos):
        for bs, be in bold_spans:
            if bs <= pos < be:
                return (bs, be)
        return None

    # Label-keyword positions.
    label_re = re.compile(r"\b(?:grand\s+total|final\s+answer|total|answer)\b", re.IGNORECASE)
    label_positions = [m.start() for m in label_re.finditer(text)]

    def label_on_line(pos):
        """True if any label keyword occurs on the same line as `pos`."""
        ls, le = line_bounds(pos)
        return any(ls <= lp < le for lp in label_positions)

    # --- Build candidate list --------------------------

    # Each candidate: (score, position, value, range_midpoint_if_any)
    # range_midpoint_if_any: if the candidate is part of "<low>-<high> <unit>",
    # use the midpoint instead of the matched value.
    candidates = []

    # Pre-scan ranges so we can prefer midpoints when a range is detected.
    # Maps the .end() of the LAST num in a range to the midpoint value.
    range_endpoints = {}
    for m in re.finditer(
        rf"\b{num_pat}\s*(?:[-–]|\s+to\s+)\s*{num_pat}\s+{unit_pat}\b",
        text,
        re.IGNORECASE,
    ):
        low_s, high_s = m.group(1), m.group(2)
        midpoint = round((to_float(low_s) + to_float(high_s)) / 2, 4)
        range_endpoints[m.end()] = midpoint

    for m in re.finditer(rf"\b{num_pat}\s+{unit_pat}\b", text, re.IGNORECASE):
        # Rate filter: "<num> <unit> per <word>" is a rate, skip.
        tail = text[m.end() : m.end() + 25]
        if re.match(r"\s+per\s+\w", tail, re.IGNORECASE):
            continue

        value = to_float(m.group(1))
        # If this match is the high end of a range, prefer the midpoint.
        if m.end() in range_endpoints:
            value = range_endpoints[m.end()]

        pos = m.start()
        bold_info = in_bold(pos)
        in_label_line = label_on_line(pos)
        # Equals-sign signal: look back ~5 chars for '=' or '≈'.
        prefix = text[max(0, pos - 5) : pos]
        has_eq = bool(re.search(r"[=≈]\s*~?\s*$", prefix))

        # Score.
        if bold_info and label_re.search(text[bold_info[0] : bold_info[1]]):
            # Bold span itself contains a label keyword. Strongest signal —
            # this is the model's bolded answer-declaration.
            score = 4
        elif in_label_line and bold_info:
            score = 3
        elif in_label_line:
            score = 2
        elif bold_info:
            score = 2
        elif has_eq:
            score = 1
        else:
            score = 0

        candidates.append((score, pos, value, bold_info))

    if not candidates:
        return None

    # --- Resolve --------------------------

    max_score = max(c[0] for c in candidates)
    top = [c for c in candidates if c[0] == max_score]

    # Within the top tier, prefer the LAST candidate. Rationale: when the
    # model gives an intermediate "Total: X" early and refines to a final
    # "Estimated Total: Y" later, Y is the committed answer. Hedged
    # alternatives ("the answer would be roughly...") rarely tie at the
    # top tier because they lack the strong markers (bold-with-label-
    # inside, or "Grand total:" prefix) that the primary commitment has.
    return top[-1][2]


def grade_run(answer_text, known_answer, unit=None, truncated=False):
    """
    Grade a single run: extract the answer and compute accuracy vs truth.

    If `truncated` is True, the run is treated as having no usable answer
    regardless of what extract_answer would return — any number found in a
    truncated response is from an intermediate calculation, not the model's
    committed final answer. Returning None for extracted and accuracy keeps
    these runs from polluting the downstream means and stdevs.

    Returns a dict with:
      - extracted:   the parsed numeric value, or None if extraction failed
                     or the run was truncated
      - known:       the ground truth value (kept in the output for reference)
      - exact_match: True iff extracted == known
      - accuracy:    1 - abs(extracted - known) / known
                     (1.0 = perfect, 0.0 = 100% off, negative = worse than 100%)
                     None if extraction failed or the run was truncated
    """
    if truncated:
        return {
            "extracted": None,
            "known": known_answer,
            "exact_match": False,
            "accuracy": None,
        }

    extracted = extract_answer(answer_text, unit)

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
    Newer entries (post truncation-detection patch) also carry
    pre_stop_reason on the entry and stop_reason on each run. Legacy
    entries without these fields fall through to the text heuristic in
    is_truncated().

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

    # Pre-query: True only if the pre-answer (a) wasn't truncated and
    # (b) contains a recognisable game score (N-N or comma-separated form).
    # A truncated pre-answer that happens to contain a score-shaped digit
    # pair somewhere shouldn't count as coverage — the model never finished
    # what it was saying.
    pre_truncated = is_truncated(entry["pre_answer"], entry.get("pre_stop_reason"))
    pre_query_answered = (not pre_truncated) and has_score(entry["pre_answer"])

    # Grade each individual run.
    graded_runs = []
    for run in entry["runs"]:
        truncated = is_truncated(run["answer"], run.get("stop_reason"))
        graded = grade_run(run["answer"], known_answer, unit, truncated=truncated)

        # Truncation, refusal, and successful extraction are mutually
        # exclusive states. If the run was truncated, we suppress the
        # refusal-pattern check entirely — phrases like "I could not find"
        # may appear in mid-thought caveats inside a truncated response, but
        # the dominant signal is "the response was cut off", not "the model
        # refused". Refusal patterns only fire on un-truncated runs that
        # failed to produce an extractable answer.
        if truncated:
            refusal_pattern = None
            run_refused = False
        else:
            refusal_pattern = find_refusal_pattern(run["answer"], refusal_patterns)
            run_refused = refusal_pattern is not None and graded["extracted"] is None

        # Confidence is null when no number was extracted — a stated
        # confidence without an answer isn't meaningful. (Also null for
        # truncated runs, since graded["extracted"] is forced to None there.)
        confidence = extract_confidence(run["answer"]) if graded["extracted"] is not None else None

        graded_runs.append({
            "run": run["run"],
            "truncated": truncated,
            "run_refused": run_refused,
            "refusal_pattern": refusal_pattern if run_refused else None,
            **{k: v for k, v in graded.items() if k != "accuracy"},
            "confidence": confidence,
            "accuracy": graded["accuracy"],
        })

    # --- Summary metrics --------------------------

    # Accuracies from runs where extraction succeeded. Truncated runs were
    # already nulled out in grade_run, so this filter naturally excludes
    # them — no extra check needed here.
    valid_accuracies = [r["accuracy"] for r in graded_runs if r["accuracy"] is not None]
    mean_accuracy_of_extracted = (
        round(sum(valid_accuracies) / len(valid_accuracies), 4)
        if valid_accuracies else None
    )

    # Raw extracted numbers from runs where extraction succeeded (and that
    # weren't truncated — same filter, by construction).
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

    runs_truncated = sum(1 for r in graded_runs if r["truncated"])
    runs_with_refusals = sum(1 for r in graded_runs if r["run_refused"])

    valid_confidences = [r["confidence"] for r in graded_runs if r["confidence"] is not None]
    mean_confidence = (
        round(sum(valid_confidences) / len(valid_confidences), 1)
        if valid_confidences else None
    )

    summary = {
        "pre_query_answered": pre_query_answered,
        "pre_query_truncated": pre_truncated,
        "runs_graded": len(graded_runs),
        "runs_with_extraction": len(valid_accuracies),
        "runs_with_refusals": runs_with_refusals,
        "runs_truncated": runs_truncated,
        # Sanity check: every run should fall into exactly one of three
        # buckets — extracted-successfully, refused, or truncated. Anything
        # else is a parse-failure-by-other-means and worth surfacing.
        "all_runs_accounted_for": (
            len(valid_accuracies) + runs_with_refusals + runs_truncated
        ) == len(graded_runs),
        "mean_confidence": mean_confidence,
        "mean_accuracy_of_extracted": mean_accuracy_of_extracted,
        "stability_of_extracted": stability_of_extracted,
    }

    # Return a new entry mirroring the input shape with summary + graded runs.
    # pre_stop_reason is preserved on the way through so the graded file
    # carries the same provenance as the raw input.
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


# --- Plotting --------------------------

def _ensure_plotting_deps():
    """
    Lazy-install and import matplotlib + numpy.

    Returns (plt, np). Called at the top of generate_plots() so the deps
    are only required when actually plotting — running `grade` alone
    doesn't need them.
    """
    if importlib.util.find_spec("matplotlib") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib"], check=True)
    # numpy comes as a transitive dep of matplotlib, but check anyway for safety.
    if importlib.util.find_spec("numpy") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy"], check=True)

    import matplotlib
    matplotlib.use("Agg")  # file-only backend; no display or GUI framework needed
    import matplotlib.pyplot as plt
    import numpy as np
    return plt, np


def _fit_linear(x_values, y_values, np):
    """
    Fit a linear trend line to (x, y) data.

    Returns (slope, intercept, r_squared, x_line, y_line) where x_line and
    y_line are two-point arrays suitable for matplotlib's plot(). Returns
    None if fewer than 2 points (a line needs at least two).

    R² is computed as 1 - SS_residual / SS_total. R²=1 means the line
    explains all variance; R²=0 means the line is no better than the mean;
    negative R² would mean the line is worse than predicting the mean
    (can happen with bad fits but won't here since we're fitting to the
    same data we're scoring).
    """
    if len(x_values) < 2:
        return None

    x_arr = np.array(x_values)
    y_arr = np.array(y_values)

    # polyfit returns coefficients in descending-degree order. For degree 1
    # that's [slope, intercept].
    slope, intercept = np.polyfit(x_arr, y_arr, 1)

    # R² from sum-of-squares definition.
    y_predicted = slope * x_arr + intercept
    ss_residual = np.sum((y_arr - y_predicted) ** 2)
    ss_total = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 1.0

    # Generate two endpoints spanning the data's x range for plt.plot().
    x_line = np.array([x_arr.min(), x_arr.max()])
    y_line = slope * x_line + intercept

    return slope, intercept, r_squared, x_line, y_line


def _plot_scatter_with_trends(data_by_model, x_label, y_label, title, output_path, plt, np):
    """
    Scatter plot colored by model with per-model + combined linear trend lines.

    Parameters
    ----------
    data_by_model : dict
        {model_name: [(x1, y1), (x2, y2), ...]}. Points where either coord
        is None are filtered out per-model (an entry can contribute to one
        plot but not another if e.g. its confidence is None but its
        stability isn't).
    x_label, y_label, title : str
        Axes labels and plot title.
    output_path : Path
        Where to write the PNG.
    plt, np : modules
        Passed in from _ensure_plotting_deps so this function doesn't
        re-import them.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # tab10 colormap gives 10 visually distinct colors. Slice it to the
    # number of models we actually have so each gets its own color.
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(data_by_model), 1)))

    # Accumulate all valid points for the combined trend line at the end.
    all_x = []
    all_y = []

    for color, (model_name, points) in zip(colors, data_by_model.items()):
        # Filter out points with None on either axis. An entry might have a
        # mean_accuracy but no mean_confidence, for example, if every run
        # for that entry failed to state a confidence percentage.
        valid = [(x, y) for x, y in points if x is not None and y is not None]
        if not valid:
            continue

        x_vals = [p[0] for p in valid]
        y_vals = [p[1] for p in valid]

        # Scatter for this model.
        ax.scatter(x_vals, y_vals, color=color, label=model_name, alpha=0.75, s=70)

        # Per-model trend line — dashed to distinguish from the combined fit.
        fit = _fit_linear(x_vals, y_vals, np)
        if fit is not None:
            slope, _, r2, x_line, y_line = fit
            ax.plot(x_line, y_line, color=color, linestyle="--", alpha=0.6,
                    label=f"{model_name} fit  (slope={slope:.3f}, R²={r2:.3f})")

        all_x.extend(x_vals)
        all_y.extend(y_vals)

    # Combined trend across all models — solid black, thicker, drawn last
    # so it sits on top of the per-model lines.
    if len(all_x) >= 2:
        fit = _fit_linear(all_x, all_y, np)
        if fit is not None:
            slope, _, r2, x_line, y_line = fit
            ax.plot(x_line, y_line, color="black", linestyle="-", linewidth=2,
                    label=f"All models fit  (slope={slope:.3f}, R²={r2:.3f})")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # tight_layout adjusts margins so labels/legend don't get clipped.
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    # Close the figure explicitly — matplotlib keeps figures alive otherwise,
    # which leaks memory across multiple plot generations.
    plt.close(fig)


# --- Public API (plotting) --------------------------

def generate_plots(graded_path, spec, output_dir=None):
    """
    Generate three accuracy plots from a graded results file.

    Plots:
      1. accuracy_vs_confidence.png             — mean_accuracy vs mean_confidence
      2. accuracy_vs_stability.png              — mean_accuracy vs stability (all)
      3. accuracy_vs_stability_filtered.png     — mean_accuracy vs stability,
                                                   pre_query_answered=True only

    Each plot has:
      - Scatter points colored by model
      - Per-model linear trend lines (dashed, model-colored)
      - Combined linear trend across all models (solid black)
      - Legend showing slope and R² for every fit

    Parameters
    ----------
    graded_path : str | Path
        Path to the graded results JSON file.
    spec : Spec
        Loaded spec — used to put the eval name in plot titles.
    output_dir : str | Path, optional
        Where to write plots. Defaults to the same directory as graded_path.

    Returns
    -------
    list[Path]
        Paths to the three generated plots in the listed order.
    """
    # Lazy-load plotting deps and grab references to pass through to helpers.
    plt, np = _ensure_plotting_deps()

    graded_path = Path(graded_path)
    with open(graded_path) as f:
        graded = json.load(f)

    # Default output: alongside the graded file. Users can override this
    # to centralise plots from multiple evals if they want.
    if output_dir is None:
        output_dir = graded_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group data points by model. Three parallel dicts because each plot
    # projects the same source entries onto different (x, y) pairs.
    # setdefault(model, []).append(...) is the standard one-liner for
    # "group entries by key into a list".
    confidence_data = {}         # model → [(mean_confidence, mean_accuracy), ...]
    stability_data = {}          # model → [(stability, mean_accuracy), ...]
    stability_filtered = {}      # same, but only pre_query_answered=True

    for entry in graded["results"]:
        model = entry["model"]
        s = entry["summary"]

        # Plot 1: confidence on x, accuracy on y.
        confidence_data.setdefault(model, []).append(
            (s["mean_confidence"], s["mean_accuracy_of_extracted"])
        )

        # Plot 2: stability on x, accuracy on y.
        stability_data.setdefault(model, []).append(
            (s["stability_of_extracted"], s["mean_accuracy_of_extracted"])
        )

        # Plot 3: same as plot 2, but only topics where pre_query passed.
        # This is the "filtered" trend from your LessWrong post — the
        # high-R² version that includes only topics in the model's
        # training data coverage.
        if s["pre_query_answered"]:
            stability_filtered.setdefault(model, []).append(
                (s["stability_of_extracted"], s["mean_accuracy_of_extracted"])
            )

    # Generate the three plots. eval_name in the title makes each plot
    # self-identifying when viewed in isolation.
    eval_name = spec.name
    paths = []

    # Derive a filename prefix from the graded JSON stem (e.g.
    # "claude-haiku-4-5_graded" → "claude-haiku-4-5") so each plot is
    # clearly tied to the spec that produced it.
    stem = graded_path.stem
    file_prefix = stem[: -len("_graded")] if stem.endswith("_graded") else stem

    p1 = output_dir / f"{file_prefix}_accuracy_vs_confidence.png"
    _plot_scatter_with_trends(
        confidence_data,
        x_label="Mean stated confidence (%)",
        y_label="Mean accuracy of extracted answers",
        title=f"{eval_name}: Accuracy vs Confidence",
        output_path=p1,
        plt=plt, np=np,
    )
    paths.append(p1)

    p2 = output_dir / f"{file_prefix}_accuracy_vs_stability.png"
    _plot_scatter_with_trends(
        stability_data,
        x_label="Stability of extracted answers (1 - stdev/mean)",
        y_label="Mean accuracy of extracted answers",
        title=f"{eval_name}: Accuracy vs Stability (all topics)",
        output_path=p2,
        plt=plt, np=np,
    )
    paths.append(p2)

    p3 = output_dir / f"{file_prefix}_accuracy_vs_stability_filtered.png"
    _plot_scatter_with_trends(
        stability_filtered,
        x_label="Stability of extracted answers (1 - stdev/mean)",
        y_label="Mean accuracy of extracted answers",
        title=f"{eval_name}: Accuracy vs Stability (pre_query_answered = True)",
        output_path=p3,
        plt=plt, np=np,
    )
    paths.append(p3)

    return paths