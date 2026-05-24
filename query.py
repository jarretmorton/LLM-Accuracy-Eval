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

# Module-level Anthropic client. One instance is reused across all calls.
client = Anthropic()


# --- Constants --------------------------

# Seconds to pause between API calls. Tuned for the API's token-per-minute
# rate limit; conservative side. Could become spec fields in v1.1.
SLEEP_BETWEEN_RUNS = 60
SLEEP_BETWEEN_TOPICS = 60
SLEEP_BETWEEN_MODELS = 60

# Token budget per response. Large enough for chain-of-thought + answer +
# confidence statement on the queries this harness was built for.
MAX_TOKENS = 1000

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

    Differences from the eval.py version:
    - `model` is now a required parameter (no module-level global)
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
            message = client.messages.create(**params)
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
    return " ".join(block.text for block in message.content if block.type == "text")


# --- Public API --------------------------

def run_harness(spec) -> Path:
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
    # Ensure the output directory exists BEFORE we start spending tokens.
    # Better to fail fast on a missing directory than after a 30-minute run.
    output_path = Path(spec.output.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect every (model, topic) block in memory and write once at the end.
    # With v1.0 spec sizes this is cheap; for much larger evals you'd want
    # to stream to disk per-topic so a crash doesn't lose everything.
    results = []

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
            pre_messages = []
            add_user_message(pre_messages, pre_query_text)
            pre_answer = chat(
                pre_messages,
                model=model_name,
                temperature=spec.temperature,
                web_search=spec.queries.pre_query.web_search,
            )
            print(f"  Pre-query done")
            time.sleep(SLEEP_BETWEEN_RUNS)

            # --- Primary query, repeated N times ---
            query_text = spec.queries.query.text.format(league=league, year=year)
            runs = []
            for i in range(1, spec.runs + 1):
                # Fresh message history each run — no context carried across runs.
                # This is methodologically important for stability measurement.
                messages = []
                add_user_message(messages, query_text)
                answer = chat(
                    messages,
                    model=model_name,
                    temperature=spec.temperature,
                    web_search=spec.queries.query.web_search,
                )
                runs.append({"run": i, "answer": answer})
                print(f"  Run {i}/{spec.runs} done")
                if i < spec.runs:
                    time.sleep(SLEEP_BETWEEN_RUNS)

            # Capture this (model, topic) block in the results list.
            results.append({
                "model": model_name,
                "league": league,
                "year": year,
                "pre_query": pre_query_text,
                "pre_answer": pre_answer,
                "query": query_text,
                "n": spec.runs,
                "runs": runs,
            })

            # Pause between topics within a model run.
            if topic_idx < len(spec.topics) - 1:
                time.sleep(SLEEP_BETWEEN_TOPICS)

        # Pause between models.
        if model_idx < len(spec.models) - 1:
            time.sleep(SLEEP_BETWEEN_MODELS)

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