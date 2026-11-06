"""Pipeline check, no network: every gold table goes through to_harness as if Riveter had returned it, then through
the harness's own scorer with the LLM calls stubbed (key alignment = identity, llm_judge = 1). A lossless transform
scores 1.0 on every task; anything lower is the transform's fault (escaping, line breaks, pandas type inference).
Writes only under runs/_selftest-gold and responses/_selftest-gold.
"""

import json
import shutil
import sys

import pandas as pd

from to_harness import HARNESS_ROOT, OUTPUT_NAMES, transform_run
from widesearch_tasks import BELONGS_COLUMN_NAME, RUN_ROOT, load_task_specs, norm_column

sys.path.insert(0, str(HARNESS_ROOT))
import src.evaluation.metric_utils as metric_utils  # noqa: E402
import src.evaluation.evaluation as evaluation  # noqa: E402
from src.evaluation.data_loader import WideSearchQuery, WideSearchResponse  # noqa: E402

RUN_NAME = "_selftest-gold"
GOLD_ROOT = RUN_ROOT / "hf_dataset" / "widesearch_gold"


def gold_text_frame(instance_id):
    frame = pd.read_csv(GOLD_ROOT / f"{instance_id}.csv", dtype=str, keep_default_na=False)
    frame.columns = [norm_column(column.strip().lstrip("\ufeff")) for column in frame.columns]
    return frame


def fake_rows(spec, gold):
    rows = []
    for _, gold_row in gold.iterrows():
        values = {column.display_name: gold_row.get(column.required_name, "") for column in spec.columns}
        if spec.row_id_column_name:
            values[spec.row_id_column_name] = " | ".join(values[column.display_name] for column in spec.key_columns)
        cells = {column.verified_column_name: {"value": values[column.display_name], "state": "ok"} for column in spec.value_columns}
        cells[BELONGS_COLUMN_NAME] = {"value": True, "state": "ok"}
        rows.append({"input": values, "cells": cells})
    return rows


def write_fake_run(specs):
    run_dir = RUN_ROOT / "runs" / RUN_NAME
    shutil.rmtree(run_dir, ignore_errors=True)
    for instance_id, spec in specs.items():
        task_dir = run_dir / instance_id
        task_dir.mkdir(parents=True)
        rows = fake_rows(spec, gold_text_frame(instance_id))
        (task_dir / "state.json").write_text(json.dumps({"instance_id": instance_id}))
        (task_dir / "build_rows.json").write_text(json.dumps(rows, ensure_ascii=False))
        (task_dir / "fill_rows.json").write_text(json.dumps(rows, ensure_ascii=False))


def stub_llm_calls():
    metric_utils.primary_key_preprocess = lambda response, reference, model_config_name: {}
    evaluation.primary_key_preprocess = metric_utils.primary_key_preprocess

    def judge_everything_correct(response, target, criterion, model_config_name):
        return [1] * len(response), ["stub"] * len(response)

    evaluation.llm_judge_column = judge_everything_correct


def harness_query(instance_id, raw_task):
    """The harness's HF loader's answer frame (WideSearchDataLoaderHF.load_answer), read from the local copy."""
    answer = pd.read_csv(GOLD_ROOT / f"{instance_id}.csv")
    answer.columns = [norm_column(column.strip()) for column in answer.columns]
    answer = answer[raw_task["evaluation"]["required"]]
    return WideSearchQuery(instance_id=instance_id, query=raw_task["query"], evaluation=raw_task["evaluation"],
                           answer=answer, language="en")


def main():
    specs = load_task_specs()
    write_fake_run(specs)
    summary = transform_run(RUN_NAME, set(OUTPUT_NAMES))
    stub_llm_calls()
    from widesearch_tasks import load_raw_tasks
    raw_tasks = load_raw_tasks()
    imperfect = 0
    for instance_id in specs:
        response_path = RUN_ROOT / "responses" / RUN_NAME / "riveter-verified" / f"riveter-verified_{instance_id}_0_response.jsonl"
        response = WideSearchResponse(**json.loads(response_path.read_text()))
        result = evaluation.evaluate_single_query(harness_query(instance_id, raw_tasks[instance_id]), response)
        problems = summary[instance_id]["riveter-verified"]["problems"]
        if result.f1_by_item < 0.9999 or problems:
            imperfect += 1
            print(f"{instance_id} item_f1={result.f1_by_item:.4f} row_f1={result.f1_by_row:.4f} problems={problems} msg={result.msg[:300]}")
    print(f"{len(specs)} tasks, {imperfect} below item F1 1.0")


if __name__ == "__main__":
    main()
