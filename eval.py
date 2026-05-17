# --- Setup --------------------------

# importlib lets us check if a package is installed before trying to import it
import importlib.util

# Auto-install the anthropic package if it's not already present
if importlib.util.find_spec("anthropic") is None:
    # subprocess lets us run shell commands (like pip) from within Python
    import subprocess
    # check=True raises an error if the install fails rather than silently continuing
    subprocess.run(["pip", "install", "anthropic"], check=True)

# load_dotenv reads a .env file and injects its values into the environment
from dotenv import load_dotenv
# Anthropic is the main client class for making API calls
from anthropic import Anthropic

# Loads ANTHROPIC_API_KEY from your .env file into the environment
load_dotenv()

# Creates the API client — automatically reads ANTHROPIC_API_KEY from the environment
client = Anthropic()

# The model to use for all API calls — swap this line to change models
model = "claude-haiku-4-5"
#model = "claude-opus-4-7"

# --- Helpers --------------------------

# Helper functions — reusable utilities for building messages and calling the API

# Appends a user turn to the conversation history
def add_user_message(messages, text):
    # The Anthropic API expects messages as dicts with "role" and "content" keys
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

# Appends an assistant turn to the conversation history
# Useful for injecting a prior reply to continue or steer a conversation
def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

# Sends the conversation to the Claude API and returns the response text
def chat(messages, system=None, temperature=1.0, stop_sequences=[], web_search=False):
    params = {
        # Model is set in the first cell so it can be changed in one place
        "model": model,
        "max_tokens": 1000,
        # The full conversation history — Claude uses this to maintain context
        "messages": messages,
        # Controls randomness: 0.0 = deterministic, 1.0 = more creative
        "temperature": temperature,
        # Optional strings that cause Claude to stop generating when encountered
        "stop_sequences": stop_sequences,
    }

    # System prompt sets Claude's persona and instructions — only added if provided
    if system:
        params["system"] = system

    # Attaches the web search tool so Claude can look up current information
    if web_search:
        params["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

    message = client.messages.create(**params)

    # Web search responses may mix text and tool-use blocks — join only the text parts
    return " ".join(block.text for block in message.content if block.type == "text")

# --- Query --------------------------

import json
import os
import time
from datetime import date

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)

# Uncomment one league to run — leave all others commented out
leagues = [
    #"National Football League",
    #"Major League Baseball",
    #"Major League Soccer",
    "National Hockey League",
    #"Basketball Africa League",
    #"Korean Baseball Organization League",
    #"Swedish Hockey League",
    #"Finnish Women's Basketball League",
]

# Number of times to run the query for the selected league
n = 2

for league in leagues:
    league_slug = league.replace(" ", "_")
    runs = []

    for i in range(1, n + 1):
        messages = []
        add_user_message(
            messages,
            f"What was the total playing time in hours for the {league} in the season ending in 2023? Include post season playoffs, but don't include any overtime."
        )

        answer = chat(messages, system="Make sure the last number in your response is the final answer in hours", temperature=1.0, web_search=True)
        runs.append({"run": i, "answer": answer})
        print(f"[{league}] Run {i}/{n} done")

        # Pause between requests to avoid hitting API rate limits
        if i < n:
            time.sleep(30)

    filename = os.path.join(results_dir, f"{model}_{league_slug}_{date.today()}_{n}runs.json")
    with open(filename, "w") as f:
        json.dump({"model": model, "league": league, "n": n, "runs": runs}, f, indent=2)
    print(f"Results written → {filename}")