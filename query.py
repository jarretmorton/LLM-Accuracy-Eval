# --- Setup --------------------------

# Standard library
import importlib.util
import json
import subprocess
import time
from datetime import date
from pathlib import Path

# Auto-install the anthropic package if it's not already present.
# Convenient for first-time users; safe to leave in for a research tool.
if importlib.util.find_spec("anthropic") is None:
    subprocess.run(["pip", "install", "anthropic"], check=True)

# Third-party
from dotenv import load_dotenv
from anthropic import Anthropic

# Loads ANTHROPIC_API_KEY from your .env file into the environment.
# Safe to call at module import — load_dotenv() is idempotent.
load_dotenv()

# Anthropic client, lazily initialised on first use so importing query.py
# has no side effects (notably, doesn't require an API key). One instance
# is reused across all calls.
_client = None

def get_client():
    """Lazily construct the Anthropic client on first use."""
    global _client
    if _client is None:
        _client = Anthropic()
    return _client

# --- Constants --------------------------

# Seconds to pause between API calls. Tuned for the API's token-per-minute
# rate limit; conservative side. Could become spec fields in v1.1.
SLEEP_BETWEEN_RUNS = 60
SLEEP_BETWEEN_TOPICS = 60
SLEEP_BETWEEN_MODELS = 60

# Token budget per response. Large enough for chain-of-thought + answer +
# confidence statement on the queries this harness was built for.
MAX_TOKENS = 3000

# Retry budget for transient API errors (429 rate limit, 529 overload).
MAX_RETRIES = 5


# --- Helpers --------------------------

def add_user_message(messages, text):
    """Append a user turn to the conversation history."""
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    """Append an assistant turn to the conversation history."""
    messages.append({"role": "assistant", "content": text})


def chat(messages, model, system=None, temperature=1.0, stop_sequences=None, web_search=False):
    """
    Send messages to the Claude API and return the response text.

    Notes:
    - `model` is a required parameter (no module-level global)
    - `stop_sequences` defaults to None and is normalized to [] inside —
      avoids the classic Python mutable-default-argument bug where a
      shared list default leaks state across calls.
    """
    params = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences or [],
    }

    if system:
        params["system"] = system

    if web_search:
        params["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

    # Retry with linear backoff on rate limit / overload.
    # 60s, 120s, 180s, 240s, 300s — total max wait ~25 minutes across 5 attempts.
    for attempt in range(MAX_RETRIES):
        try:
            message = get_client().messages.create(**params)
            break
        except Exception as e:
            if hasattr(e, "status_code") and e.status_code in (429, 529):
                wait = 60 * (attempt + 1)
                print(f"  API error {e.status_code} — retrying in {wait}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError("Max retries exceeded")

    # Web-search responses may mix text and tool-use blocks; keep only text.
    text = " ".join(block.text for block in message.content if block.type == "text")
    # stop_reason tells us whether the model finished naturally ("end_turn"),
    # hit a stop sequence ("stop_sequence"), used a tool ("tool_use"), or was
    # cut off mid-generation by the token cap ("max_tokens"). The grader uses
    # this to flag truncated responses whose extracted value would reflect an
    # intermediate calculation rather than a committed answer.
    return text, message.stop_reason


# --- Public API --------------------------

def run_harness(spec, spec_path) -> Path:
    """
    Run the eval harness defined by `spec`.

    For each (model, topic) pair, runs:
      1. The pre_query once (web search disabled — coverage check)
      2. The primary query n times (web search enabled per spec)

    Writes a single consolidated JSON file at spec.output.path containing
    all (model, topic, run) results. Returns the path to that file.

    Required spec attributes (set by spec.py's load_spec()):
      spec.name                            (str)
      spec.version                         (str)
      spec.models                          (list[str])
      spec.topics                          (list of dicts with `league`, `year`)
      spec.runs                            (int — number of repeats per query)
      spec.temperature                     (float)
      spec.queries.pre_query.text          (str — format string with {league} {year})
      spec.queries.pre_query.web_search    (bool)
      spec.queries.query.text              (str — same format string convention)
      spec.queries.query.web_search        (bool)
      spec.output.path                     (str)

    This signature assumes spec.py produces an object with attribute access
    (dataclass, SimpleNamespace, pydantic model, etc.). If you go with
    plain dicts in spec.py, swap the dotted access for bracket access.
    """
    # Derive the output filename from the spec file's stem so the results
    # file is always named after the spec that produced it (e.g.
    # claude-sonnet-4-6.yaml → results/claude-sonnet-4-6.json).
    # The directory still comes from spec.output.path so the YAML controls
    # where files land without hard-coding the name.
    output_dir = Path(spec.output.path).parent
    # When the primary query is disabled (queries.query.enabled: false), this
    # is a cheap pre-query-only re-run. We don't pre-compute output_path here
    # because the destination depends on whether a full results file already
    # exists for this spec — handled at write time below.
    pre_query_only = not spec.queries.query.enabled
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect every (model, topic) block in memory and write once at the end.
    # With v1.0 spec sizes this is cheap; for much larger evals you'd want
    # to stream to disk per-topic so a crash doesn't lose everything.
    results = []

    # Resolve the soft token budget and system prompts once, before the
    # per-(model, topic) loop. None values for system_prompt indicate "use
    # the claude.ai baseline behavior" (system=None to chat()).
    soft_budget = (
        spec.soft_token_budget
        if getattr(spec, "soft_token_budget", None)
        else int(MAX_TOKENS / 1.2)
    )
    expected_unit = getattr(spec.grader, "expected_unit", "") or ""
    pre_system_template = spec.queries.pre_query.system_prompt or None
    query_system_template = spec.queries.query.system_prompt or None

    # Outer loop: models. Each model runs against every topic before moving
    # to the next model. Serial — no parallelism for v1.0.
    for model_idx, model_name in enumerate(spec.models):
        print(f"\n=== Model: {model_name} ({model_idx + 1}/{len(spec.models)}) ===")

        for topic_idx, topic in enumerate(spec.topics):
            league = topic["league"]
            year = topic["year"]
            print(f"\n[{league}] ({topic_idx + 1}/{len(spec.topics)})")

            # --- Pre-query (coverage check) ---
            # .format() substitutes {league} and {year} placeholders from
            # the spec's pre_query template into the actual prompt text.
            pre_query_text = spec.queries.pre_query.text.format(league=league, year=year)

            # Resolve system prompts for this topic. Same placeholder set
            # as the query text, plus {soft_token_budget} and {expected_unit}.
            fmt_kwargs = dict(
                league=league, year=year,
                soft_token_budget=soft_budget,
                expected_unit=expected_unit,
            )
            pre_system = pre_system_template.format(**fmt_kwargs) if pre_system_template else None
            query_system = query_system_template.format(**fmt_kwargs) if query_system_template else None

            pre_messages = []
            add_user_message(pre_messages, pre_query_text)
            pre_answer, pre_stop_reason = chat(
                pre_messages,
                model=model_name,
                system=pre_system,
                temperature=spec.temperature,
                web_search=spec.queries.pre_query.web_search,
            )
            print(f"  Pre-query done")
            time.sleep(SLEEP_BETWEEN_RUNS)

            # --- Primary query, repeated N times ---
            # Skipped entirely when the primary query is disabled — the
            # pre-query-only path leaves runs empty and n at 0, then
            # splices into the existing results file at write time.
            query_text = spec.queries.query.text.format(league=league, year=year)
            runs = []
            if not pre_query_only:
                for i in range(1, spec.runs + 1):
                    # Fresh message history each run — no context carried across runs.
                    # This is methodologically important for stability measurement.
                    messages = []
                    add_user_message(messages, query_text)
                    answer, stop_reason = chat(
                        messages,
                        model=model_name,
                        system=query_system,
                        temperature=spec.temperature,
                        web_search=spec.queries.query.web_search,
                    )
                    runs.append({"run": i, "answer": answer, "stop_reason": stop_reason})
                    print(f"  Run {i}/{spec.runs} done")
                    if i < spec.runs:
                        time.sleep(SLEEP_BETWEEN_RUNS)
            else:
                print("  Primary query disabled (queries.query.enabled: false) — skipping runs")

            # Capture this (model, topic) block in the results list.
            results.append({
                "model": model_name,
                "league": league,
                "year": year,
                "pre_query": pre_query_text,
                "pre_answer": pre_answer,
                "pre_stop_reason": pre_stop_reason,
                "query": query_text,
                "n": 0 if pre_query_only else spec.runs,
                "runs": runs,
            })

            # Pause between topics within a model run.
            if topic_idx < len(spec.topics) - 1:
                time.sleep(SLEEP_BETWEEN_TOPICS)

        # Pause between models.
        if model_idx < len(spec.models) - 1:
            time.sleep(SLEEP_BETWEEN_MODELS)

    # Write path depends on whether this was a pre-query-only re-run.
    #
    # Full run: write a fresh results file at <spec_stem>.json (overwriting
    # any prior version) — the standard behavior.
    #
    # Pre-query-only: splice the fresh pre-queries into the existing results
    # file at <spec_stem>.json, preserving the (expensive) primary-query runs
    # already there. If that file does not yet exist, fall back to writing a
    # standalone <spec_stem>_prequery.json so the work is not lost.
    target_path = output_dir / (Path(spec_path).stem + ".json")

    if pre_query_only and target_path.exists():
        # Splice fresh pre-queries into the existing full results file.
        with open(target_path) as f:
            existing = json.load(f)

        # Index fresh results by (model, league, year) for O(1) lookup.
        fresh_idx = {
            (r["model"], r["league"], r["year"]): r for r in results
        }
        existing_keys = {
            (e["model"], e["league"], e["year"]) for e in existing.get("results", [])
        }

        updated = 0
        for e in existing.get("results", []):
            key = (e["model"], e["league"], e["year"])
            src = fresh_idx.get(key)
            if src is None:
                continue  # entry not touched by this re-run; leave as-is
            e["pre_query"] = src["pre_query"]
            e["pre_answer"] = src["pre_answer"]
            e["pre_stop_reason"] = src["pre_stop_reason"]
            updated += 1

        unmatched = sorted(k for k in fresh_idx if k not in existing_keys)
        for k in unmatched:
            print(f"  WARNING: fresh pre-query for {k} has no entry in "
                  f"{target_path.name} — skipped (add the topic and re-run full)")

        # Provenance breadcrumb. The grader and plotter ignore unknown
        # top-level keys, so this is non-breaking.
        existing["pre_query_refreshed_date"] = str(date.today())

        with open(target_path, "w") as f:
            json.dump(existing, f, indent=2)

        print(f"\nRefreshed {updated} pre-query block(s) in {target_path.name}; "
              f"{len(unmatched)} unmatched.")
        return target_path

    if pre_query_only:
        # No existing target — fall back to a standalone file so the work
        # isn't lost. User can later splice manually via `main.py splice`
        # or run the full pipeline once and re-run pre-query-only after.
        fallback_path = output_dir / (Path(spec_path).stem + "_prequery.json")
        print(f"\nNote: no existing full results file at {target_path.name} — "
              f"writing standalone {fallback_path.name} instead.")
        output_path = fallback_path
    else:
        output_path = target_path

    # Wrap per-(model, topic) results with spec-level metadata so a reader
    # of the output file can tell what produced it without cross-referencing.
    output = {
        "spec_name": spec.name,
        "spec_version": spec.version,
        "run_date": str(date.today()),
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path