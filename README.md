# LLM Accuracy Eval

> A black-box evaluation harness for measuring LLM accuracy and stability without white-box model access.

LLM Accuracy Eval is the runnable companion to the LessWrong post [*A Black-Box Procedure for LLM Confidence in Critical Applications*](https://www.lesswrong.com/posts/unaLT4A6hSTCLNGod/a-black-box-procedure-for-llm-confidence-in-critical#comments). The post argued that for critical-application use of LLMs, two black-box signals are tractable and useful: (1) **training coverage**, estimated by asking secondary questions with web search disabled and observing whether the model refuses; and (2) **answer stability**, measured by running the same query repeatedly and comparing responses.

This repo turns that procedure into a CLI and a YAML-spec-driven eval format that can be pointed at any chat-completion API.

## Thesis

Across 640 queries spanning 8 sports league topics and 4 frontier LLMs:

- Model self-reported confidence has near-zero predictive value for accuracy (R² ≈ 0.02).
- Answer stability strongly predicts accuracy (R² ≈ 0.995, after filtering topics where models refused secondary questions with search disabled).

The implication is that any team deploying LLMs in critical applications can get a useful confidence signal today, in production, without lab cooperation or model access — by querying carefully and counting agreement.

## What this is

A small Python package that:

- Accepts a YAML eval spec (prompt, expected behavior, grader).
- Runs the prompt N times across M models in parallel.
- Computes stability and accuracy metrics.
- Emits structured JSON results for downstream analysis.

See [`docs/architecture.md`](docs/architecture.md) for the design and [`specs/example.yaml`](specs/example.yaml) for the eval format.

## What this is not

Not a benchmark. Not a leaderboard. Not a replacement for HELM or lm-eval-harness. This is a focused harness for the specific evaluation pattern from the LessWrong post.

## Status

Pre-v0.1. README, architecture sketch, and example spec only.

## License

MIT. See [LICENSE](LICENSE).
