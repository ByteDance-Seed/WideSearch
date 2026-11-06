"""Writes the WideSearch harness's response files from a runner's saved rows.

Three outputs from one run, each its own --model_config_name for the harness:
  riveter-build              the list builder's table: the dataset's input columns after the build
  riveter-verified           each non-key cell from its "(verified)" output column when the row agent filled
                             it, else the build's draft
  riveter-verified-in-scope  riveter-verified without the rows the row agent marked false in "Belongs in the table"

Usage:
  python to_harness.py --run-name pilot-v1
Writes responses/<run_name>/<output>/<output>_<instance_id>_0_response.jsonl and checks that the harness's own
parser (WideSearchResponse.extract_dataframe) reads every table back with the intended rows and columns.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from widesearch_tasks import BELONGS_COLUMN_NAME, RUN_ROOT, load_task_specs, norm_column

HARNESS_ROOT = RUN_ROOT.parent.parent
OUTPUT_NAMES = ("riveter-build", "riveter-verified", "riveter-verified-in-scope")
FILLED_CELL_STATES = {"ok", "manual"}
LINE_BREAKS = re.compile(r"\s*[\r\n\t]+\s*")


def cell_text(value, placeholder):
    """One table cell as the harness's parser will read it back: line breaks folded, a value with a pipe or a
    double quote CSV-quoted (the parser rejoins the pipes and hands the line to pandas.read_csv)."""
    if value is None or isinstance(value, (dict, list)):
        text = "" if value is None else json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = LINE_BREAKS.sub("; ", text.strip())
    if not text and placeholder:
        return placeholder
    if "|" in text or '"' in text:
        return '"' + text.replace('"', '""') + '"'
    return text


def verified_value(row, column):
    cell = (row.get("cells") or {}).get(column.verified_column_name) or {}
    value = cell.get("value")
    if cell.get("state") in FILLED_CELL_STATES and value not in (None, ""):
        return value
    return (row.get("input") or {}).get(column.display_name)


def build_value(row, column):
    return (row.get("input") or {}).get(column.display_name)


def marked_out_of_scope(row):
    cell = (row.get("cells") or {}).get(BELONGS_COLUMN_NAME) or {}
    return cell.get("state") in FILLED_CELL_STATES and cell.get("value") is False


def table_rows(spec, rows, value_for):
    """Rows as lists of raw values in the spec's column order; a row missing any key value is no table row."""
    table = []
    for row in rows:
        values = [value_for(row, column) for column in spec.columns]
        keys_present = all(str(values[index] or "").strip() for index, column in enumerate(spec.columns) if column.is_key)
        if keys_present:
            table.append(values)
    return table


def markdown_table(spec, table):
    header = "| " + " | ".join(column.display_name for column in spec.columns) + " |"
    divider = "|" + "---|" * len(spec.columns)
    lines = [header, divider]
    for values in table:
        lines.append("| " + " | ".join(cell_text(value, spec.missing_value_placeholder) for value in values) + " |")
    return "```markdown\n" + "\n".join(lines) + "\n```"


def tables_for_task(spec, task_dir):
    """{output name: raw table} for whichever row files the run saved."""
    tables = {}
    build_rows_path = task_dir / "build_rows.json"
    fill_rows_path = task_dir / "fill_rows.json"
    if build_rows_path.exists():
        tables["riveter-build"] = table_rows(spec, json.loads(build_rows_path.read_text()), build_value)
    if fill_rows_path.exists():
        fill_rows = json.loads(fill_rows_path.read_text())
        tables["riveter-verified"] = table_rows(spec, fill_rows, verified_value)
        in_scope_rows = [row for row in fill_rows if not marked_out_of_scope(row)]
        tables["riveter-verified-in-scope"] = table_rows(spec, in_scope_rows, verified_value)
    return tables


def write_response(output_dir, output_name, instance_id, response_text):
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {"instance_id": instance_id, "response": response_text, "messages": None, "trial_idx": 0}
    path = output_dir / f"{output_name}_{instance_id}_0_response.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def parse_check(spec, response_text, expected_row_count):
    """Problems the harness's parser would have with this response: wrong columns or a lost row."""
    sys.path.insert(0, str(HARNESS_ROOT))
    from loguru import logger
    from src.evaluation.data_loader import WideSearchResponse

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    frame = WideSearchResponse(instance_id=spec.instance_id, response=response_text).extract_dataframe()
    if frame is None:
        return ["harness parser found no table"] if expected_row_count else []
    problems = []
    parsed_columns = {norm_column(column) for column in frame.columns}
    expected_columns = {column.required_name for column in spec.columns}
    if parsed_columns != expected_columns:
        problems.append(f"columns differ from required (the harness maps them with an LLM): "
                        f"extra={sorted(parsed_columns - expected_columns)} missing={sorted(expected_columns - parsed_columns)}")
    if len(frame) != expected_row_count:
        problems.append(f"parser read {len(frame)} rows, wrote {expected_row_count}")
    return problems


def transform_run(run_name, output_names):
    run_dir = RUN_ROOT / "runs" / run_name
    instance_ids = sorted(path.parent.name for path in run_dir.glob("*/state.json"))
    specs = load_task_specs(instance_ids)
    summary = {}
    for instance_id, spec in specs.items():
        tables = tables_for_task(spec, run_dir / instance_id)
        for output_name, table in tables.items():
            if output_name not in output_names:
                continue
            response_text = markdown_table(spec, table)
            write_response(RUN_ROOT / "responses" / run_name / output_name, output_name, instance_id, response_text)
            problems = parse_check(spec, response_text, len(table))
            summary.setdefault(instance_id, {})[output_name] = {"rows": len(table), "problems": problems}
    (RUN_ROOT / "responses" / run_name).mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "responses" / run_name / "transform_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--outputs", default=",".join(OUTPUT_NAMES))
    arguments = parser.parse_args()
    summary = transform_run(arguments.run_name, set(arguments.outputs.split(",")))
    for instance_id, outputs in summary.items():
        line = "  ".join(f"{name}={detail['rows']}" + (f" PROBLEMS {detail['problems']}" if detail["problems"] else "")
                         for name, detail in outputs.items())
        print(f"{instance_id}  {line}")


if __name__ == "__main__":
    main()
