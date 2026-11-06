"""Per-task and overall numbers for one run, read from the harness's eval results and the runner's state files.

Usage:
  python analyze.py --run-name pilot-v1
Gold row counts are read here only to report padding (rows returned vs gold rows); nothing here feeds a run.
"""

import argparse
import json
import statistics

import pandas as pd

from to_harness import OUTPUT_NAMES
from widesearch_tasks import RUN_ROOT

METRICS = ("f1_by_item", "precision_by_item", "recall_by_item", "f1_by_row", "precision_by_row", "recall_by_row", "score")
CREDIT_STEPS = ("count", "build", "fill")


def eval_result(run_name, output_name, instance_id, results_suffix):
    path = RUN_ROOT / "results" / run_name / f"{output_name}{results_suffix}" / f"{output_name}_{instance_id}_0_eval_result.json"
    return json.loads(path.read_text()) if path.exists() else None


def gold_row_count(instance_id):
    return len(pd.read_csv(RUN_ROOT / "hf_dataset" / "widesearch_gold" / f"{instance_id}.csv"))


def task_rows(run_name, results_suffix):
    transform_summary = json.loads((RUN_ROOT / "responses" / run_name / "transform_summary.json").read_text())
    rows = []
    for state_path in sorted((RUN_ROOT / "runs" / run_name).glob("*/state.json")):
        state = json.loads(state_path.read_text())
        instance_id = state["instance_id"]
        row = {
            "instance_id": instance_id,
            "gold_rows": gold_row_count(instance_id),
            "count_sent": state.get("count", {}).get("count"),
            **{f"credits_{step}": float(state.get(step, {}).get("credits_used") or 0) for step in CREDIT_STEPS},
        }
        for output_name in OUTPUT_NAMES:
            result = eval_result(run_name, output_name, instance_id, results_suffix)
            row[f"{output_name}:rows"] = transform_summary.get(instance_id, {}).get(output_name, {}).get("rows")
            for metric in METRICS:
                row[f"{output_name}:{metric}"] = result.get(metric) if result else None
        rows.append(row)
    return rows


def print_overall(rows):
    for output_name in OUTPUT_NAMES:
        graded = [row for row in rows if row.get(f"{output_name}:f1_by_item") is not None]
        if not graded:
            continue
        means = {metric: round(100 * statistics.mean(row[f"{output_name}:{metric}"] for row in graded), 2) for metric in METRICS}
        print(f"{output_name} ({len(graded)} tasks): " + "  ".join(f"{metric}={value}" for metric, value in means.items()))
    credits = {step: round(sum(row[f"credits_{step}"] for row in rows), 2) for step in CREDIT_STEPS}
    total = round(sum(credits.values()), 2)
    print(f"credits: {credits} total={total} per task={round(total / max(len(rows), 1), 2)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--rejudged", action="store_true", help="read the results of grade.py --retry-incomplete-judge-replies")
    arguments = parser.parse_args()
    results_suffix = "-rejudged" if arguments.rejudged else ""
    rows = task_rows(arguments.run_name, results_suffix)
    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    short_columns = ["instance_id", "gold_rows", "count_sent"] + [f"credits_{step}" for step in CREDIT_STEPS]
    for output_name in OUTPUT_NAMES:
        short_columns += [f"{output_name}:rows", f"{output_name}:f1_by_item", f"{output_name}:f1_by_row"]
    print(frame[short_columns].round(3).to_string(index=False))
    print_overall(rows)
    (RUN_ROOT / "results" / arguments.run_name).mkdir(parents=True, exist_ok=True)
    frame.to_csv(RUN_ROOT / "results" / arguments.run_name / f"analysis{results_suffix}.csv", index=False)


if __name__ == "__main__":
    main()
