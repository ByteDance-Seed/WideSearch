# Riveter on WideSearch

This directory runs WideSearch tasks on [Riveter](https://www.riveterhq.com)'s `/v1` API and writes the response files that this repository's evaluation stage grades. The evaluation code in `src/` is not changed.

## How a task maps to Riveter

Each task is one Riveter dataset.

1. **Row count.** One `POST /v1/search_agent` call reads the query and estimates how many rows a complete answer has. When the query fixes the number (for example, "the top 10 for each year from 2020 to 2024"), that number is used; otherwise the estimate times 1.25. The count caps the build.
2. **Recipe.** `POST /v1/enrichments` creates the dataset. Every column the query names is an input column, so the list builder writes the whole table. When the key (`evaluation.unique_columns`) has two or more columns, a `Row ID` column comes first, because a dataset deduplicates on its first input column. Each non-key column also gets a string output column, `<column> (verified)`, and one boolean column, `Belongs in the table`.
3. **Build.** `POST /v1/datasets/{id}/rows/build` with the query, unchanged, plus two fixed sentences (`widesearch_tasks.BUILD_PROMPT_SUFFIX`). The list builder searches and reads the web and writes the rows.
4. **Fill.** `POST /v1/enrich` with the dataset (or `auto_fill` on the build). One agent per row checks each draft value against a source and writes the verified value.
5. **Transform.** `to_harness.py` writes one Markdown table per task in the query's column names and checks that `WideSearchResponse.extract_dataframe` reads it back with the same rows and columns.

Riveter receives the query text and the column names only. The gold tables and the evaluation's metrics, criteria, and preprocessors are never sent.

One run is scored three ways:

| Output | Cells |
| --- | --- |
| `riveter-build` | The list builder's table. |
| `riveter-verified` | Each non-key cell from its `(verified)` column when the row agent filled it, else the build's value. |
| `riveter-verified-in-scope` | `riveter-verified` without the rows marked `false` in `Belongs in the table`. |

## Setup

1. Set up this repository's environment (`sh prepare-env.sh`, or `uv sync`).
2. Write `integrations/riveter/.env`:

   ```
   RIVETER_API_KEY=sk_riv_...
   OPENAI_API_KEY=sk-...
   ```

   `RIVETER_API_KEY` needs an account on Datasets. `OPENAI_API_KEY` is the judge's key.

The dataset downloads from Hugging Face into `integrations/riveter/hf_dataset/` on first use.

## Run

Run from `integrations/riveter/` with the repository's Python (`../../.venv/bin/python`).

```bash
python riveter_runner.py plan --tasks pilot          # the specs; no network
python riveter_runner.py run --run-name en-v1 --tasks all --parallel 20 --auto-fill --kickoff-only
python riveter_runner.py run --run-name en-v1 --tasks all --parallel 20 --auto-fill   # collects the rows
python riveter_runner.py status --run-name en-v1
```

`run` spends Riveter credits. With `--kickoff-only`, every task's count and build start and the command exits; Riveter starts each fill when the build's rows land. The same command without `--kickoff-only` saves the rows. State is kept in `runs/<run_name>/<instance_id>/state.json`, and a rerun resumes. `--tasks` takes `pilot`, `all`, or comma-separated instance ids.

## Grade

```bash
python to_harness.py --run-name en-v1
python grade.py --run-name en-v1 --output riveter-verified --tasks all
python analyze.py --run-name en-v1
```

`grade.py` runs `scripts/run_infer_and_eval_batching.py --stage eval` unchanged except for the judge's HTTP client: the harness builds an `AzureOpenAI` client, and `grade.py` sends the same requests (`default_eval_config`, `gpt-4.1-2025-04-14`, same messages and generate kwargs, same retries) through the OpenAI client.

`--retry-incomplete-judge-replies` is a separate, disclosed pass written to `<output>-rejudged`. It asks the judge again when an `llm_judge_column` reply does not score every pair. Without it, an unparseable reply, or a reply that lists only some `idx` keys, scores every missing pair 0. This happens on large columns.

`selftest_gold_roundtrip.py` checks the transform without network access: it passes every gold table through `to_harness.py` and the evaluation with the LLM calls stubbed out. 97 of the 100 English tables score 1.0. The other three are `ws_en_009` and `ws_en_049`, where the query names a column differently from the gold file (the evaluation's LLM column mapping aligns them), and `ws_en_093`, where a gold key value contains a line break.

## Results: English set, November 2026

| Output | Item F1 | Row F1 | Success rate |
| --- | ---: | ---: | ---: |
| `riveter-verified` | 81.47 | 59.49 | 14% |
| `riveter-build` | 79.80 | 56.28 | 15% |
| `riveter-verified-in-scope` | 80.87 | 59.46 | 14% |

- **Setup:** the 100 English tasks, one trial, run November 5–6, 2026, and graded with this repository at `9825ba7`. The models are Riveter's: `gpt-6.1-sol` for the list builder, `gpt-6-luna` for the row agent and the count step.
- **Second judge pass:** with `--retry-incomplete-judge-replies`, `riveter-verified` scores 81.66 item F1 and 59.64 row F1.
- **Contamination:** in `ws_en_048`, a row agent cited this benchmark's gold file on Hugging Face for two cells. Without that task, `riveter-verified` scores 81.44 item F1, 59.79 row F1, and 14 of 99 tables exact. The list builders' logs for all 100 tasks contain no reference to the gold files.
- **Comparability:** the published leaderboard reports Avg@4 over the 200-task set. These numbers are one trial over the English half.
