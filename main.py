"""
llm-accuracy-eval main entry point.

This file ties together three independent modules:

    spec.py    — parses YAML eval specs into a typed config object
    query.py   — runs the harness against the API and writes raw results
    grader.py  — reads raw results, computes accuracy/stability, writes graded results

Each module is independently usable. This file just orchestrates them based
on user intent expressed via CLI subcommands:

    llm-accuracy-eval run specs/example.yaml        # full pipeline
    llm-accuracy-eval collect specs/example.yaml    # harness only (uses API, no grading)
    llm-accuracy-eval grade results/example.json specs/example.yaml   # grader only (no API)

The `grade` subcommand is the value-add of keeping the modules separate —
it lets anyone re-grade your published results without spending API tokens.
"""

# --- Imports --------------------------

# argparse is Python's standard library CLI parser. It handles --flags,
# positional arguments, subcommands, and produces --help text automatically.
import argparse

# sys gives us sys.exit() — the conventional way to terminate a program with
# an exit code. 0 = success, non-zero = failure (shell scripts check this).
import sys

# pathlib provides an object-oriented path API. Path("a/b").exists(),
# Path("foo.json").with_name("foo_graded.json"), etc. — cleaner than os.path
# and treats paths as objects rather than strings.
from pathlib import Path

# Import the three modules we orchestrate. Each exposes the function(s) we
# need via its public API. The modules don't know about each other — only
# this file knows how to chain them. That's the separation we wanted.
from spec import load_spec        # load_spec(path: str) -> Spec
from query import run_harness     # run_harness(spec: Spec) -> Path
from grader import grade_results, generate_plots  # grade_results(results_path: Path, spec: Spec) -> Path


# --- Subcommand handlers --------------------------

# Each subcommand is a function that takes the parsed args object and
# returns nothing. The CLI parser (built below) dispatches to the right
# handler based on which subcommand the user invoked.

def cmd_run(args: argparse.Namespace) -> None:
    """
    Full pipeline: parse spec → run harness → grade results.

    The most common entry point. A reviewer running your code once gets
    both raw API outputs and graded metrics.
    """
    # Step 1: Load and validate the YAML spec. If anything's missing or
    # malformed, spec.py raises an exception and we exit here with a clear
    # error — better than crashing partway through an expensive run.
    spec = load_spec(args.spec_path)
    print(f"Loaded spec: {spec.name} v{spec.version}")

    # Step 2: Run the harness. This is the expensive step — API calls plus
    # rate-limit waits. Writes raw results to the path declared in spec.output.
    print(f"Running harness — {len(spec.models)} model(s), "
          f"{len(spec.topics)} topic(s), n={spec.runs}")
    results_path = run_harness(spec, args.spec_path)
    print(f"Raw results written → {results_path}")

    # Step 3: Grade. Pure local computation — no API calls, no waiting.
    # Reads the raw results file written in Step 2.
    print("Grading results...")
    graded_path = grade_results(results_path, spec)
    print(f"Graded results written → {graded_path}")

    print("Generating plots...")
    plot_paths = generate_plots(graded_path, spec)
    for p in plot_paths:
        print(f"  Wrote {p}")


def cmd_collect(args: argparse.Namespace) -> None:
    """
    Harness only: parse spec → run harness → write raw results.

    Use when you want to capture results but defer grading (grader is still
    being iterated on, or you want to grade differently later).
    """
    spec = load_spec(args.spec_path)
    print(f"Loaded spec: {spec.name} v{spec.version}")
    print(f"Running harness — {len(spec.models)} model(s), "
          f"{len(spec.topics)} topic(s), n={spec.runs}")
    results_path = run_harness(spec, args.spec_path)
    print(f"Raw results written → {results_path}")
    print(f"(Skipping grading — run `grade {results_path} {args.spec_path}` when ready)")


def cmd_grade(args: argparse.Namespace) -> None:
    """
    Grader only: read raw results → compute metrics → write graded results.

    No API calls. Re-grade existing results after a grader change, or verify
    someone else's published numbers. Requires the spec for grading context
    (refusal patterns, truth values, expected unit).
    """
    spec = load_spec(args.spec_path)
    results_path = Path(args.results_path)

    # Sanity check before we try to read the file — clearer error than
    # letting open() throw FileNotFoundError from inside the grader.
    if not results_path.exists():
        sys.exit(f"Error: results file not found: {results_path}")

    print(f"Grading {results_path} against spec {spec.name}...")
    graded_path = grade_results(results_path, spec)
    print(f"Graded results written → {graded_path}")

    print("Generating plots...")
    plot_paths = generate_plots(graded_path, spec)
    for p in plot_paths:
        print(f"  Wrote {p}")


# --- CLI setup --------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Construct the argparse parser with subcommands.

    argparse subcommands work like git: `git commit -m ...` has `commit` as
    the subcommand and `-m ...` as flags. Same shape here:
    `llm-accuracy-eval run specs/example.yaml` has `run` as the subcommand
    and `specs/example.yaml` as a positional argument.

    Pulled into its own function so it's testable in isolation and the
    structure is easy to scan.
    """
    # Top-level parser describes the program itself.
    parser = argparse.ArgumentParser(
        prog="llm-accuracy-eval",
        description="Black-box evaluation harness for measuring LLM accuracy and stability.",
    )

    # add_subparsers creates the subcommand machinery. dest="command" tells
    # argparse where to store the chosen subcommand name, so we can dispatch
    # on it below. required=True forces the user to pick one (otherwise
    # running with no args silently does nothing).
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- `run` subcommand ---
    # Each subparser is itself an ArgumentParser with its own arguments.
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full pipeline (harness + grader) from a YAML spec.",
    )
    # Positional argument — required, no flag prefix. User invokes as:
    #     llm-accuracy-eval run specs/example.yaml
    # The string "spec_path" becomes args.spec_path inside the handler.
    run_parser.add_argument("spec_path", help="Path to the YAML eval spec.")

    # --- `collect` subcommand ---
    collect_parser = subparsers.add_parser(
        "collect",
        help="Run the harness only (no grading). Useful for deferred grading.",
    )
    collect_parser.add_argument("spec_path", help="Path to the YAML eval spec.")

    # --- `grade` subcommand ---
    # This one takes two positional arguments. Order matters — argparse
    # assigns them by position in the command line.
    grade_parser = subparsers.add_parser(
        "grade",
        help="Grade existing raw results (no API calls).",
    )
    grade_parser.add_argument("results_path", help="Path to the raw results JSON file.")
    grade_parser.add_argument("spec_path", help="Path to the YAML eval spec (for grader context).")

    return parser


# --- Entry point --------------------------

def main() -> None:
    """
    Parse CLI args and dispatch to the right subcommand handler.

    The pattern (parse → dispatch dict → call handler) is standard for any
    CLI tool with multiple subcommands. The alternative — a chain of
    if/elif on args.command — works but doesn't scale as well.
    """
    parser = build_parser()
    args = parser.parse_args()

    # Dispatch table: maps subcommand name to its handler function.
    # Adding a new subcommand is a one-line change here plus adding the
    # subparser above.
    handlers = {
        "run": cmd_run,
        "collect": cmd_collect,
        "grade": cmd_grade,
    }

    handler = handlers[args.command]

    # Wrap the handler in try/except so we catch errors from any subcommand
    # and exit with a clean message instead of dumping a Python traceback
    # on the user. Tracebacks are useful for debugging but ugly for end
    # users; if you want them while developing, comment out the except
    # blocks temporarily.
    try:
        handler(args)
    except FileNotFoundError as e:
        sys.exit(f"Error: file not found — {e.filename}")
    except KeyboardInterrupt:
        # User pressed Ctrl+C. Exit cleanly without a traceback.
        sys.exit("\nInterrupted by user")
    except Exception as e:
        # Catch-all for unexpected errors. Print the message and exit
        # non-zero so shell scripts wrapping this tool can detect failure.
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