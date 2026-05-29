# Pre-query-only re-run — runbook

Use this when you've changed the **pre-query** prompt (e.g. the decoupled
field-commitment rules from Option A) and want to refresh the pre-query
section of an existing results file without paying for the expensive
search-enabled primary-query runs again.

## How it works

1. **`queries.query.enabled: false`** in a spec makes `run_harness` skip the
   primary query entirely.
2. At write time, run_harness looks for the existing results file at the
   spec's canonical output path (`<spec-stem>.json`). If it exists,
   run_harness **splices the fresh pre-queries into it in place** — keyed on
   `(model, league, year)` — and preserves every primary-query run untouched.
3. If no existing file is found (first run, renamed file, etc.), run_harness
   falls back to writing a standalone `<spec-stem>_prequery.json` so the
   work isn't lost. You can splice it later with `main.py splice`.

The toggle replaces the old two-step "run → splice" flow with a single
`run` invocation.

## Steps

```bash
# 1. Update the pre-query system prompt in the spec.
# 2. Flip queries.query.enabled to false in the spec.
# 3. Re-run — pre-queries refresh in place; grader and plots regenerate
#    against the now-updated file.
python main.py run specs/<spec>.yaml

# 4. Flip queries.query.enabled back to true for future full runs.
```

The console will print one of:

- `Refreshed N pre-query block(s) in <file>; M unmatched.` — splice happened in place.
- `Note: no existing full results file at <file> — writing standalone <file>_prequery.json instead.` — fallback path.

## Notes & caveats

- **The canonical target is `<spec-stem>.json`.** If your full file is
  named to match the spec it was originally run from, the splice is
  automatic. If you renamed the file or run the pre-query re-run from a
  *different* spec stem, the splice will write a standalone
  `_prequery.json` instead (or splice into the wrong file if a different
  spec happens to share the stem). Easiest practice: use the same spec
  file (just toggling `enabled`) that produced the original full run.

- **Join key is `(model, league, year)`.** Unmatched warnings fire when the
  spec has a topic the target file lacks. Targets entries that the spec
  doesn't cover are left alone — pre-existing runs for other topics or
  models in the same file are not touched.

- **Calibration heads-up on the decoupled prompt.** Telling the model to
  commit best-recall teams even when uncertain will raise the pre-query
  commitment rate — and some of those commitments will be wrong. That's the
  intent: it makes weak/partial coverage *visible* in the fields (and thus
  to `pre_query_partially_answered`) instead of hiding it in prose. Expect
  the `partially_answered` bucket to grow and to contain some incorrect
  team recalls; that's the signal you wanted to study, not noise to suppress.

- **Provenance.** A top-level `pre_query_refreshed_date` is written on
  every in-place refresh. The grader and plotter ignore unknown top-level
  keys, so this is non-breaking. Delete it if you need output byte-identical
  to a unified run.

- **The `splice` subcommand still exists** for the manual workflow: when
  you have an external pre-query file (different naming, different source,
  etc.) and want explicit control over which file goes where. The
  integrated mode is just a smarter default for the common case.
