# --- Setup --------------------------

import json
import os
import re
import statistics

# Resolve paths relative to this script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")

# Set the result file to grade — filename only, no path
input_file = "claude-haiku-4-5_Swedish_Hockey_League_2023_2026-05-17_5runs.json"

# Load ground truth — build a lookup keyed by (league, year)
with open(os.path.join(base_dir, "truth.json")) as f:
    truth_data = json.load(f)

truth_lookup = {
    (entry["league"], entry["year"]): entry["known_answer_hours"]
    for entry in truth_data["entries"]
}

# --- Helpers --------------------------

# Phrases that indicate the model couldn't answer — extend this list as new patterns are observed
REFUSAL_PATTERNS = [
    r"I don't have access",
    r"I cannot find",
    r"I'm not sure",
    r"I don't know",
    r"I don't have information about",
    r"I was unable to find",
    r"I'm unable to find",
]

def is_refusal(text):
    # Returns True if any refusal pattern matches anywhere in the text (case-insensitive)
    return any(re.search(p, text, re.IGNORECASE) for p in REFUSAL_PATTERNS)

def extract_unit_from_query(query):
    # Find "in <unit>" where the unit is a word, not a number (avoids matching "in 2023")
    match = re.search(r'\bin\s+([a-zA-Z]+)\b(?!\s+\d)', query)
    # Strip trailing "s" to normalize plural → singular for use in regex (e.g. "hours" → "hour")
    return match.group(1).rstrip("s") if match else None

def extract_last_number(text, unit=None):
    if unit:
        # Find numbers immediately followed by the unit word (singular or plural)
        matches = re.findall(rf'\b(\d+(?:,\d+)*(?:\.\d+)?)\s+{unit}s?\b', text, re.IGNORECASE)
    else:
        matches = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text)
    if not matches:
        return None
    # Remove commas from numbers like "1,400" before converting to float
    return float(matches[-1].replace(",", ""))

def grade_run(answer_text, known_answer, unit=None):
    extracted = extract_last_number(answer_text, unit)

    if extracted is None:
        return {"extracted": None, "known": known_answer, "exact_match": False, "accuracy": None, "reason": "No number found in response"}

    # Accuracy = 1 - abs(model - truth) / truth
    # 1.0 = perfect, 0.0 = off by 100%, negative = worse than 100% error
    accuracy = round(1 - abs(extracted - known_answer) / known_answer, 4)

    return {
        "extracted": extracted,
        "known": known_answer,
        "exact_match": extracted == known_answer,
        "accuracy": accuracy,
    }

# --- Grading --------------------------

filepath = os.path.join(results_dir, input_file)

with open(filepath) as f:
    result = json.load(f)

league = result["league"]
year = result["year"]

# Verify the query in the result file matches the truth file's template before grading
expected_query = truth_data["query_template"].format(league=league, year=year)
actual_query = result.get("query", "")
if actual_query != expected_query:
    raise ValueError(
        f"Query mismatch — grading aborted.\n"
        f"  Expected: {expected_query}\n"
        f"  Found:    {actual_query}"
    )

known_answer = truth_lookup.get((league, year))

if known_answer is None:
    raise ValueError(f"No truth entry found for ({league}, {year})")

# Extract the unit from the query so the grader looks for numbers paired with it (e.g. "1,400 hours")
unit = extract_unit_from_query(result["query"])

# Grade each individual run
graded_runs = []
for run in result["runs"]:
    graded = grade_run(run["answer"], known_answer, unit)
    graded_runs.append({"run": run["run"], "run_refused": graded["extracted"] is None and is_refusal(run["answer"]), **graded})

# Build list of accuracy scores from runs where extraction succeeded
valid_accuracies = [r["accuracy"] for r in graded_runs if r["accuracy"] is not None]

# Average accuracy across valid runs; None if no valid runs
mean_accuracy_of_extracted = round(sum(valid_accuracies) / len(valid_accuracies), 4) if valid_accuracies else None

# Build list of raw extracted numbers from runs where extraction succeeded
valid_extracted = [r["extracted"] for r in graded_runs if r["extracted"] is not None]

# Spread of extracted numbers; requires at least 2 samples
stdev_of_extracted = round(statistics.stdev(valid_extracted), 4) if len(valid_extracted) > 1 else None

# Average of extracted numbers; requires at least 2 samples
mean_of_extracted = round(statistics.mean(valid_extracted), 4) if len(valid_extracted) > 1 else None

# 1.0 = perfectly consistent answers, lower = more variance relative to mean
stability_of_extracted = round(1 - (stdev_of_extracted / mean_of_extracted), 4) if stdev_of_extracted is not None and mean_of_extracted else None

runs_with_refusals = sum(1 for r in graded_runs if r["run_refused"])

summary = {
    "runs_graded": len(graded_runs),
    "runs_with_extraction": len(valid_accuracies),
    "runs_with_refusals": runs_with_refusals,
    "all_runs_accounted_for": (len(valid_accuracies) + runs_with_refusals) == len(graded_runs),
    "mean_accuracy_of_extracted": mean_accuracy_of_extracted,
    "stability_of_extracted": stability_of_extracted,
}

output = {
    "model": result["model"],
    "query": result["query"],
    "league": league,
    "year": year,
    "n": result["n"],
    "summary": summary,
    "runs": graded_runs,
}

# Write graded file alongside the source with _graded suffix
graded_filename = input_file.replace(".json", "_graded.json")
graded_filepath = os.path.join(results_dir, graded_filename)
with open(graded_filepath, "w") as f:
    json.dump(output, f, indent=2)

print(f"Graded: {graded_filename}")
print(f"Runs graded:                {summary['runs_graded']}")
print(f"Runs with extraction:       {summary['runs_with_extraction']}")
print(f"Runs with refusals:         {summary['runs_with_refusals']}")
if not summary["all_runs_accounted_for"]:
    print("WARNING: runs_with_extraction + runs_with_refusals does not equal runs_graded")
print(f"Mean accuracy of extracted: {summary['mean_accuracy_of_extracted']}")
print(f"Stability of extracted:     {summary['stability_of_extracted']}")
