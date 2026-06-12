"""
llm-accuracy-eval — main entry point.

Orchestrates the eval modules:

    spec.py              — parses YAML eval specs into a typed config object
    query.py             — runs the eval harness against the Claude API, writes raw results
    grader.py            — reads raw results, computes accuracy/stability metrics, writes
                           graded results and plots (numeric grader)
    grader_structured.py — alternate grader for system-prompted, terminal-block output,
                           selected by `grader.type: structured`

Each module is independently usable. This file chains them together based on
the subcommand the user picks. Run `python main.py --help` for a full list.

Available commands
------------------
run     <spec.yaml>
    Full pipeline: load spec → call API → grade → plot.
    Most common entry point. Produces a raw results JSON, a graded JSON, and
    accuracy plots (four for the numeric grader, five for the structured
    grader), all named after the spec file.

    Example:
        python main.py run specs/claude-sonnet-4-6_structured.yaml

collect <spec.yaml>
    Harness only — calls the API and writes raw results but skips grading.
    Use when you want to defer grading (e.g. the grader is still being tuned)
    or capture results for later analysis.

    Example:
        python main.py collect specs/claude-sonnet-4-6_structured.yaml

grade   <results.json> <spec.yaml>
    Grader only — no API calls. Reads an existing raw results file, computes
    metrics, writes a graded JSON and plots. Use to re-grade published results
    after changing the grader, or to verify someone else's numbers.

    Example:
        python main.py grade results/claude-sonnet-4-6_structured.json specs/claude-sonnet-4-6_structured.yaml

plot    [results_dir]
    Combined plots — merges every *_graded.json in the given directory (default:
    results/), grouped by grader kind, and generates a combined plot set per
    kind (numeric: 4 plots, structured: 5). Writes a combined_<kind>_graded.json
    and the matching combined_* PNG files to the same directory. Does not call
    the API or re-grade anything.

    Example:
        python main.py plot
        python main.py plot results/

splice  <base.json> <prequery.json>
    Fold a pre-query-only (*_prequery.json) re-run back into a full results
    file, keeping the full file's primary-query runs untouched. No API calls.
    Use --in-place or --out to control the destination. See docs/prequery_rerun.md.

    Example:
        python main.py splice results/foo.json results/foo_prequery.json --in-place
"""

# --- Imports --------------------------

import argparse       # CLI parser: subcommands, positional args, --help
import json           # reading/writing graded JSON for the `plot` command
import sys            # sys.exit() for clean error termination
from pathlib import Path            # object-oriented file paths
from types import SimpleNamespace   # lightweight object used as a stand-in spec
                                    # for the `plot` command (needs only .name)

# The four modules this file orchestrates. They don't know about each other —
# only main.py knows how to chain them.
from spec import load_spec                        # load_spec(path) -> Spec
from query import run_harness                     # run_harness(spec, spec_path) -> Path
from grader import grade_results as grade_numeric, generate_plots
from grader_structured import (
    grade_results as grade_structured,
    generate_plots as plots_structured,
)
                                                  # generate_plots(path, spec) -> list[Path]


def _grade_results(results_path, spec):
    # Only `structured` has its own grader. `exact`, `judge`, and `none` are
    # not yet implemented and currently fall back to the numeric grader.
    if spec.grader.type == "structured":
        return grade_structured(results_path, spec)
    return grade_numeric(results_path, spec)


def _generate_plots(graded_path, spec):
    if spec.grader.type == "structured":
        return plots_structured(graded_path, spec)
    return generate_plots(graded_path, spec)


# --- Subcommand handlers --------------------------

# Each subcommand is a function that takes the parsed args object and
# returns nothing. The CLI parser (built below) dispatches to the right
# handler based on which subcommand the user invoked.

def cmd_run(args: argparse.Namespace) -> None:
    """
    Full pipeline: load spec → call API → grade → generate plots.

    Most common entry point. Produces output files all named after the spec:
    a raw results JSON, a graded JSON, and accuracy PNG plots (four for the
    numeric grader, five for the structured grader).
    This is the expensive path — it makes live API calls and waits for
    rate-limit cooldowns between runs.
    """
    # Step 1: Load and validate the YAML spec. Fails fast if anything is
    # missing or malformed — better than crashing partway through an API run.
    spec = load_spec(args.spec_path)
    print(f"Loaded spec: {spec.name} v{spec.version}")

    # Step 2: Run the harness. API calls + rate-limit sleeps happen here.
    # The output filename is derived from the spec filename, not spec.output.path,
    # so claude-sonnet-4-6_structured.yaml → results/claude-sonnet-4-6_structured.json.
    print(f"Running harness — {len(spec.models)} model(s), "
          f"{len(spec.topics)} topic(s), n={spec.runs}")
    results_path = run_harness(spec, args.spec_path)
    print(f"Raw results written → {results_path}")

    # Step 3: Grade. No API calls — pure local computation over the JSON
    # written in Step 2. Appends _graded to the filename.
    print("Grading results...")
    graded_path = _grade_results(results_path, spec)
    print(f"Graded results written → {graded_path}")

    # Step 4: Generate the accuracy plots alongside the graded file.
    print("Generating plots...")
    plot_paths = _generate_plots(graded_path, spec)
    for p in plot_paths:
        print(f"  Wrote {p}")


def cmd_collect(args: argparse.Namespace) -> None:
    """
    Harness only: load spec → call API → write raw results (no grading).

    Use when you want to capture API responses but defer grading — for example
    if the grader is still being tuned, or you want to grade with different
    specs later. Prints the exact `grade` command to run when ready.
    """
    spec = load_spec(args.spec_path)
    print(f"Loaded spec: {spec.name} v{spec.version}")
    print(f"Running harness — {len(spec.models)} model(s), "
          f"{len(spec.topics)} topic(s), n={spec.runs}")
    results_path = run_harness(spec, args.spec_path)
    print(f"Raw results written → {results_path}")
    print(f"(Skipping grading — run `grade {results_path} {args.spec_path}` when ready)")


def cmd_plot(args: argparse.Namespace) -> None:
    """
    Combined plots: merge *_graded.json files in a directory and emit a
    separate combined plot set per grader kind.

    Why per-kind: the numeric and structured graders both populate a field
    called pre_query_answered, but with different semantics — numeric uses
    a free-text score regex, structured uses strict SCORE-field commitment.
    Mixing them on one axis is silently misleading. Splitting by grader
    kind also lets the structured group keep its 5-plot output rather than
    being downgraded to the numeric 4-plot view.

    Files are grouped by their top-level `grader_kind` field (absent ⇒
    numeric, "structured" ⇒ structured). For each non-empty group a
    combined_<kind>_graded.json is written and the matching plotter is
    invoked (numeric: 4 plots, structured: 5 plots). Any combined_*_graded.json
    already in the directory is excluded from inputs so re-runs don't fold
    prior combined outputs back in.

    Note: any stale `combined_graded.json` from the pre-split version of this
    command is now ignored on input but left on disk — delete it manually.
    """
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        sys.exit(f"Error: results directory not found: {results_dir}")

    graded_files = sorted(
        p for p in results_dir.glob("*_graded.json")
        if not p.name.startswith("combined_")
    )
    if not graded_files:
        sys.exit(f"No *_graded.json files found in {results_dir}")

    # Group inputs by grader_kind. Numeric grader doesn't write the field,
    # so treat a missing key as numeric.
    groups: dict[str, list] = {}
    print(f"Combining {len(graded_files)} graded file(s):")
    for p in graded_files:
        with open(p) as f:
            data = json.load(f)
        kind = data.get("grader_kind") or "numeric"
        groups.setdefault(kind, []).append((p, data.get("results", [])))
        print(f"  [{kind}] {p.name}")

    # Per group: write a combined file and call the right plotter.
    for kind in sorted(groups):
        combined_results = []
        for _p, rs in groups[kind]:
            combined_results.extend(rs)

        combined_path = results_dir / f"combined_{kind}_graded.json"
        payload = {"spec_name": f"combined-{kind}", "results": combined_results}
        if kind != "numeric":
            payload["grader_kind"] = kind   # round-trip for downstream readers
        with open(combined_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nCombined {kind} graded data written → {combined_path}")

        # Stand-in spec — both plotters only need spec.name for titles.
        spec_min = SimpleNamespace(name=f"All Models ({kind})")
        plot_fn = plots_structured if kind == "structured" else generate_plots
        print(f"Generating {kind} combined plots...")
        for pp in plot_fn(combined_path, spec_min):
            print(f"  Wrote {pp}")


def cmd_grade(args: argparse.Namespace) -> None:
    """
    Grader only: read an existing raw results JSON → compute metrics → write
    graded results and plots. No API calls are made.

    Use to re-grade after changing grader logic, or to verify someone else's
    published raw results without spending tokens. Requires the original spec
    for grading context: truth values, expected unit, and refusal patterns.
    """
    spec = load_spec(args.spec_path)
    results_path = Path(args.results_path)

    # Check the file exists before entering the grader — gives a clearer
    # error message than a raw FileNotFoundError from inside grade_results.
    if not results_path.exists():
        sys.exit(f"Error: results file not found: {results_path}")

    print(f"Grading {results_path} against spec {spec.name}...")
    graded_path = _grade_results(results_path, spec)
    print(f"Graded results written → {graded_path}")

    print("Generating plots...")
    plot_paths = _generate_plots(graded_path, spec)
    for p in plot_paths:
        print(f"  Wrote {p}")


def cmd_splice(args: argparse.Namespace) -> None:
    """
    Splice the pre-query section of a pre-query-only re-run into a full
    results file, keeping the (expensive) primary-query runs from the full
    file untouched. No API calls are made.

    Workflow this supports: re-run only the pre-query with an updated prompt
    (a spec with `queries.query.enabled: false`, which writes a cheap
    *_prequery.json), then fold those fresh pre-queries back into the
    original full results file so it reads as if everything was run together.

    Join key is (model, league, year). For each base entry, the pre_query,
    pre_answer, and pre_stop_reason fields are overwritten from the matching
    pre-query entry; everything else (query, n, runs, metadata) is preserved.
    Mismatches in either direction are reported, not silently dropped.

    Output is non-destructive by default (writes <base>_spliced.json). Pass
    --in-place to overwrite the base file, or --out to name the target.
    """
    base_path = Path(args.base_path)
    prequery_path = Path(args.prequery_path)
    for p in (base_path, prequery_path):
        if not p.exists():
            sys.exit(f"Error: file not found: {p}")

    with open(base_path) as f:
        base = json.load(f)
    with open(prequery_path) as f:
        pre = json.load(f)

    # Index the pre-query file by join key for O(1) lookup.
    pre_index = {
        (e["model"], e["league"], e["year"]): e for e in pre["results"]
    }
    base_keys = {
        (e["model"], e["league"], e["year"]) for e in base["results"]
    }

    updated = 0
    unchanged = 0
    for e in base["results"]:
        key = (e["model"], e["league"], e["year"])
        src = pre_index.get(key)
        if src is None:
            unchanged += 1
            print(f"  WARNING: no pre-query for {key} — base entry left unchanged")
            continue
        e["pre_query"] = src["pre_query"]
        e["pre_answer"] = src["pre_answer"]
        e["pre_stop_reason"] = src.get("pre_stop_reason")
        updated += 1

    # Surface pre-query entries that had no home in the base file — usually a
    # sign the two files came from different specs or topic lists.
    extra = sorted(k for k in pre_index if k not in base_keys)
    for k in extra:
        print(f"  WARNING: pre-query entry {k} not found in base — ignored")

    # Provenance breadcrumb. The grader and plotter ignore unknown top-level
    # keys, so this is non-breaking; delete it if you need byte-identical output.
    base["pre_query_spliced_from"] = prequery_path.name

    if args.in_place:
        out_path = base_path
    elif args.out:
        out_path = Path(args.out)
    else:
        out_path = base_path.with_name(base_path.stem + "_spliced" + base_path.suffix)

    with open(out_path, "w") as f:
        json.dump(base, f, indent=2)

    print(f"Spliced {updated} pre-query block(s); {unchanged} base entries unchanged; "
          f"{len(extra)} extra pre-query entries ignored.")
    print(f"Written → {out_path}")

def build_parser() -> argparse.ArgumentParser:
    """
    Construct the argparse parser with all five subcommands.

    Works like git: the subcommand name comes first, then its arguments.
    For example:
        python main.py run specs/claude-sonnet-4-6_structured.yaml

    Pulled into its own function so it's independently testable and easy
    to scan when adding new subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="llm-accuracy-eval",
        description="Black-box evaluation harness for measuring LLM accuracy and stability.",
    )

    # dest="command" stores the chosen subcommand name on args so the dispatch
    # table in main() can look up the right handler. required=True means the
    # user must pick a subcommand — no silent no-op on bare invocation.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- `run` subcommand ---
    # Full pipeline: spec → API → grade → plots. Most common entry point.
    run_parser = subparsers.add_parser(
        "run",
        help="Full pipeline: call API, grade results, and generate plots.",
    )
    run_parser.add_argument("spec_path", help="Path to the YAML eval spec.")

    # --- `collect` subcommand ---
    # API calls only — skips grading. Output filename is derived from spec name.
    collect_parser = subparsers.add_parser(
        "collect",
        help="Call the API and save raw results only (no grading or plots).",
    )
    collect_parser.add_argument("spec_path", help="Path to the YAML eval spec.")

    # --- `grade` subcommand ---
    # Re-grade an existing raw results file with no API calls.
    # Two positional arguments — order matters (results first, then spec).
    grade_parser = subparsers.add_parser(
        "grade",
        help="Grade existing raw results and generate plots (no API calls).",
    )
    grade_parser.add_argument("results_path", help="Path to the raw results JSON file.")
    grade_parser.add_argument("spec_path", help="Path to the YAML eval spec (provides truth values and grader config).")

    # --- `plot` subcommand ---
    # Merge all *_graded.json files in a directory into combined plots.
    # results_dir is optional — defaults to results/ if omitted.
    plot_parser = subparsers.add_parser(
        "plot",
        help="Merge all graded results in a directory and generate combined plots.",
    )
    plot_parser.add_argument(
        "results_dir",
        nargs="?",
        default="results",
        help="Directory containing *_graded.json files (default: results/).",
    )

    # --- `splice` subcommand ---
    # Fold a pre-query-only re-run back into a full results file. No API calls.
    splice_parser = subparsers.add_parser(
        "splice",
        help="Overwrite the pre-query section of a full results file from a "
             "pre-query-only (*_prequery.json) re-run, keeping its runs.",
    )
    splice_parser.add_argument(
        "base_path",
        help="Full results JSON whose primary-query runs are kept.",
    )
    splice_parser.add_argument(
        "prequery_path",
        help="Pre-query-only results JSON (*_prequery.json) supplying fresh pre-queries.",
    )
    splice_parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: <base>_spliced.json).",
    )
    splice_parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite base_path in place instead of writing a new file.",
    )

    return parser


# --- Entry point --------------------------

def main() -> None:
    """
    Parse CLI args and dispatch to the right subcommand handler.

    Uses a dispatch table (dict) rather than if/elif chains — adding a new
    subcommand is one line here plus one subparser in build_parser().
    """
    parser = build_parser()
    args = parser.parse_args()

    # Dispatch table: subcommand name → handler function.
    handlers = {
        "run":     cmd_run,
        "collect": cmd_collect,
        "grade":   cmd_grade,
        "plot":    cmd_plot,
        "splice":  cmd_splice,
    }

    try:
        handlers[args.command](args)
    except FileNotFoundError as e:
        sys.exit(f"Error: file not found — {e.filename}")
    except KeyboardInterrupt:
        # Ctrl+C — exit without a traceback.
        sys.exit("\nInterrupted by user")
    except Exception as e:
        # Any other error: print the message and exit non-zero so shell
        # scripts wrapping this tool can detect failure. Comment out this
        # block temporarily if you need the full traceback while debugging.
        sys.exit(f"Error: {e}")


# --- Python module idiom --------------------------

# This idiom lets the file act as both a runnable script AND an importable
# module. When run directly (`python main.py run specs/example.yaml`), the
# __name__ variable is "__main__" and main() runs. When imported by another
# file (`from main import build_parser`), __name__ is "main" and the call
# below is skipped.
#
# Without this guard, importing this file would also try to run main(),
# which would consume sys.argv unexpectedly and crash any importing code.

if __name__ == "__main__":
    main()