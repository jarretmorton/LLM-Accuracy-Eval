"""
llm-accuracy-eval — main entry point.

Orchestrates three independent modules:

    spec.py    — parses YAML eval specs into a typed config object
    query.py   — runs the eval harness against the Claude API, writes raw results
    grader.py  — reads raw results, computes accuracy/stability metrics, writes
                 graded results and plots

Each module is independently usable. This file chains them together based on
the subcommand the user picks. Run `python main.py --help` for a full list.

Available commands
------------------
run     <spec.yaml>
    Full pipeline: load spec → call API → grade → plot.
    Most common entry point. Produces a raw results JSON, a graded JSON, and
    three accuracy plots, all named after the spec file.

    Example:
        python main.py run specs/claude-sonnet-4-6.yaml

collect <spec.yaml>
    Harness only — calls the API and writes raw results but skips grading.
    Use when you want to defer grading (e.g. the grader is still being tuned)
    or capture results for later analysis.

    Example:
        python main.py collect specs/claude-sonnet-4-6.yaml

grade   <results.json> <spec.yaml>
    Grader only — no API calls. Reads an existing raw results file, computes
    metrics, writes a graded JSON and plots. Use to re-grade published results
    after changing the grader, or to verify someone else's numbers.

    Example:
        python main.py grade results/claude-sonnet-4-6.json specs/claude-sonnet-4-6.yaml

plot    [results_dir]
    Combined plots — merges every *_graded.json in the given directory (default:
    results/) and generates a single set of plots with all models overlaid.
    Writes combined_graded.json and three combined_* PNG files to the same
    directory. Does not call the API or re-grade anything.

    Example:
        python main.py plot
        python main.py plot results/
"""

# --- Imports --------------------------

import argparse       # CLI parser: subcommands, positional args, --help
import json           # reading/writing graded JSON for the `plot` command
import sys            # sys.exit() for clean error termination
from pathlib import Path            # object-oriented file paths
from types import SimpleNamespace   # lightweight object used as a stand-in spec
                                    # for the `plot` command (needs only .name)

# The three modules this file orchestrates. They don't know about each other —
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

    Most common entry point. Produces three output files all named after the
    spec: a raw results JSON, a graded JSON, and three accuracy PNG plots.
    This is the expensive path — it makes live API calls and waits for
    rate-limit cooldowns between runs.
    """
    # Step 1: Load and validate the YAML spec. Fails fast if anything is
    # missing or malformed — better than crashing partway through an API run.
    spec = load_spec(args.spec_path)
    print(f"Loaded spec: {spec.name} v{spec.version}")

    # Step 2: Run the harness. API calls + rate-limit sleeps happen here.
    # The output filename is derived from the spec filename, not spec.output.path,
    # so claude-sonnet-4-6.yaml → results/claude-sonnet-4-6.json.
    print(f"Running harness — {len(spec.models)} model(s), "
          f"{len(spec.topics)} topic(s), n={spec.runs}")
    results_path = run_harness(spec, args.spec_path)
    print(f"Raw results written → {results_path}")

    # Step 3: Grade. No API calls — pure local computation over the JSON
    # written in Step 2. Appends _graded to the filename.
    print("Grading results...")
    graded_path = _grade_results(results_path, spec)
    print(f"Graded results written → {graded_path}")

    # Step 4: Generate the three accuracy plots alongside the graded file.
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
    Combined plots: merge all *_graded.json files in a directory and generate
    a single set of plots with every model's data overlaid on the same axes.

    No API calls are made and nothing is re-graded. The command:
      1. Finds every *_graded.json in results_dir (skipping combined_graded.json
         itself so re-running doesn't double-count).
      2. Merges their results arrays into combined_graded.json.
      3. Generates three plots prefixed with 'combined_' so they sit alongside
         the single-model plots without overwriting them.
    """
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        sys.exit(f"Error: results directory not found: {results_dir}")

    combined_filename = "combined_graded.json"
    # Collect all per-model graded files, excluding the combined output itself
    # so re-running the command doesn't fold last run's combined data back in.
    graded_files = sorted(
        p for p in results_dir.glob("*_graded.json")
        if p.name != combined_filename
    )

    if not graded_files:
        sys.exit(f"No *_graded.json files found in {results_dir}")

    # Merge the results arrays from every graded file into a single list.
    print(f"Combining {len(graded_files)} graded file(s):")
    combined_results = []
    for p in graded_files:
        print(f"  {p.name}")
        with open(p) as f:
            combined_results.extend(json.load(f)["results"])

    # Write the merged data so generate_plots has a file to read from.
    combined_path = results_dir / combined_filename
    with open(combined_path, "w") as f:
        json.dump({"spec_name": "combined", "results": combined_results}, f, indent=2)
    print(f"Combined graded data written → {combined_path}")

    # Use a minimal stand-in spec — generate_plots only needs spec.name for
    # plot titles; everything else comes from the graded JSON.
    print("Generating combined plots...")
    spec = SimpleNamespace(name="All Models")
    plot_paths = generate_plots(combined_path, spec)
    for p in plot_paths:
        print(f"  Wrote {p}")


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


# --- CLI setup --------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Construct the argparse parser with all four subcommands.

    Works like git: the subcommand name comes first, then its arguments.
    For example:
        python main.py run specs/claude-sonnet-4-6.yaml

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