"""Runs WideSearch tasks on Riveter's /v1 API, one dataset per task.

Per task: (1) count — one search_agent call reads the query and estimates the table's row count (1 credit);
(2) recipe — POST /v1/enrichments: every table column is an input column (the list builder writes them all), and
one "<column> (verified)" output column per non-key column plus "Belongs in the table" (free); (3) build —
POST /v1/datasets/{id}/rows/build with the query (paid, list builder); (4) fill — POST /v1/enrich with the
dataset (paid, one row agent per row). Rows are saved after the build and after the fill, so the transform
scores the list builder's table alone and the verified table from one run.

State lives in runs/<run_name>/<instance_id>/state.json; a rerun skips finished steps and resumes polling runs
already started. Usage:

  python riveter_runner.py plan --tasks pilot
  python riveter_runner.py run --run-name pilot-v1 --tasks pilot --parallel 4
  python riveter_runner.py run --run-name pilot-v1 --tasks pilot --parallel 10 --auto-fill --kickoff-only
  python riveter_runner.py status --run-name pilot-v1

--auto-fill with --kickoff-only leaves every paid step on Riveter's side once the builds start, so this machine can
sleep; the same command without --kickoff-only later saves the rows of the finished builds and fills.
"""

import argparse
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from widesearch_tasks import PILOT_INSTANCE_IDS, RUN_ROOT, load_task_specs

API_BASE = os.environ.get("RIVETER_API_BASE", "https://api.riveterhq.com/v1")
USER_AGENT = "widesearch-riveter/1.0"
HTTP_TIMEOUT_SECONDS = 90
LONG_POLL_SECONDS = 50
POLL_FLOOR_SECONDS = 10
COUNT_DEADLINE_SECONDS = 15 * 60
BUILD_DEADLINE_SECONDS = 3 * 60 * 60
FILL_DEADLINE_SECONDS = 6 * 60 * 60
ROWS_PAGE_LIMIT = 1000
TERMINAL_STATUSES = {"success", "stopped", "failed", "error"}

COUNT_FALLBACK_ROWS = 150
COUNT_MARGIN_WHEN_OPEN = 1.25
COUNT_MIN_ROWS = 3
COUNT_MAX_ROWS = 2000
BUILD_BUDGET_FLOOR_CREDITS = 100
BUILD_BUDGET_CREDITS_PER_ROW = 2
BUILD_BUDGET_MAX_CREDITS = 5000

COUNT_PROMPT_TEMPLATE = """Below is a request for a data table. Estimate how many rows a complete and correct answer to it has. Do not build the table.

If the request fixes the number of rows (for example "the top 10 for each year from 2020 to 2024" is 50 rows, "the 10 most populous cities" is 10), give that number and set count_is_fixed_by_request to true. Otherwise research the set the request describes (how many items exist that meet its conditions) and give your best estimate, with count_is_fixed_by_request false. In basis, say in one sentence where the number comes from.

REQUEST:
{query}"""

COUNT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "row_count": {"type": "integer", "minimum": 1},
        "count_is_fixed_by_request": {"type": "boolean"},
        "basis": {"type": "string"},
    },
}

LOG_LOCK = threading.Lock()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(instance_id, message):
    with LOG_LOCK:
        print(f"{utc_now()} {instance_id} {message}", flush=True)


def load_api_key():
    env_path = Path(os.environ.get("WIDESEARCH_ENV_FILE", RUN_ROOT / ".env"))
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "RIVETER_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    if os.environ.get("RIVETER_API_KEY"):
        return os.environ["RIVETER_API_KEY"]
    sys.exit(f"RIVETER_API_KEY is not set: add it to {env_path}")


class RiveterApiError(Exception):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {json.dumps(body)[:800]}")
        self.status = status
        self.body = body


class RiveterClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def request(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f"{API_BASE}{path}", data=data, method=method, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        })
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            raw = error.read()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"raw": raw.decode(errors="replace")[:800]}
            raise RiveterApiError(error.code, parsed) from None

    def get_with_retry(self, path, attempts=5):
        for attempt in range(attempts):
            try:
                return self.request("GET", path)
            except (urllib.error.URLError, TimeoutError, ConnectionError, RiveterApiError) as error:
                retriable = not isinstance(error, RiveterApiError) or error.status in (429, 500, 502, 503, 504, 520, 524)
                if not retriable or attempt == attempts - 1:
                    raise
                time.sleep(5 * (attempt + 1))

    def wait_for_run(self, instance_id, run_id, deadline_seconds, step):
        started = time.monotonic()
        last_progress = None
        while True:
            loop_started = time.monotonic()
            run = self.get_with_retry(f"/runs/{run_id}?wait={LONG_POLL_SECONDS}")
            if run.get("status") in TERMINAL_STATUSES:
                return run
            progress = json.dumps(run.get("progress"), sort_keys=True)
            if progress != last_progress:
                log(instance_id, f"{step} {run_id} {run.get('status')} progress={progress} credits={run.get('credits_used')}")
                last_progress = progress
            if time.monotonic() - started > deadline_seconds:
                raise TimeoutError(f"{step} run {run_id} not finished after {deadline_seconds}s")
            time.sleep(max(0, POLL_FLOOR_SECONDS - (time.monotonic() - loop_started)))

    def start_keyed_run(self, path, body):
        """POST a run with a run_key; a 409 for that key means an earlier attempt started it, so adopt that run."""
        try:
            return self.request("POST", path, body)
        except RiveterApiError as error:
            if error.status == 409 and "run_key" in json.dumps(error.body):
                return {"id": f"run_{body['run_key']}"}
            raise

    def existing_run_of_type(self, dataset_id, run_type):
        dataset = self.get_with_retry(f"/datasets/{dataset_id}")
        runs = [run for run in dataset.get("runs") or [] if run.get("type") == run_type]
        return runs[0] if runs else None

    def all_rows(self, dataset_id):
        rows, cursor = [], None
        while True:
            query = f"?limit={ROWS_PAGE_LIMIT}" + (f"&cursor={cursor}" if cursor is not None else "")
            page = self.get_with_retry(f"/datasets/{dataset_id}/rows{query}")
            rows.extend(page.get("rows", []))
            cursor = page.get("next_cursor")
            if cursor is None:
                return rows


class TaskState:
    def __init__(self, run_dir, instance_id):
        self.dir = run_dir / instance_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "state.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"instance_id": instance_id}

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))
        temporary.replace(self.path)

    def step(self, name):
        return self.data.setdefault(name, {})

    def write_rows(self, filename, rows):
        (self.dir / filename).write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def run_envelope_summary(run):
    return {
        "run_id": run.get("id"),
        "status": run.get("status"),
        "stopped_reason": run.get("stopped_reason"),
        "error": run.get("error"),
        "credits_used": run.get("credits_used"),
        "credit_estimate": (run.get("credit_detail") or {}).get("estimate"),
        "progress": run.get("progress"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
    }


def count_for_estimate(estimate):
    row_count = estimate.get("row_count") if isinstance(estimate, dict) else None
    if not isinstance(row_count, int) or row_count < 1:
        return COUNT_FALLBACK_ROWS
    if estimate.get("count_is_fixed_by_request") is not True:
        row_count = math.ceil(row_count * COUNT_MARGIN_WHEN_OPEN)
    return max(COUNT_MIN_ROWS, min(COUNT_MAX_ROWS, row_count))


def build_budget_for(count):
    return min(BUILD_BUDGET_MAX_CREDITS, BUILD_BUDGET_FLOOR_CREDITS + BUILD_BUDGET_CREDITS_PER_ROW * count)


def fill_budget_cap(spec, row_count):
    return round(row_count * 1.5 * len(spec.output_columns()), 2)


def run_count_step(client, spec, state, run_key_prefix):
    step = state.step("count")
    if step.get("count"):
        return step["count"]
    if not step.get("run_id"):
        body = {
            "prompt": COUNT_PROMPT_TEMPLATE.format(query=spec.query.strip()),
            "output_schema": COUNT_OUTPUT_SCHEMA,
            "wait": 0,
            "run_key": f"{run_key_prefix}-count-{spec.instance_id}",
            "max_credits": 1,
        }
        run = client.start_keyed_run("/search_agent", body)
        step.update(run_id=run["id"], started_at=utc_now())
        state.save()
    run = client.wait_for_run(spec.instance_id, step["run_id"], COUNT_DEADLINE_SECONDS, "count")
    result = client.get_with_retry(f"/runs/{step['run_id']}/result")
    estimate = (result.get("output") or {}).get("result")
    step.update(run_envelope_summary(run), estimate=estimate, count=count_for_estimate(estimate))
    state.save()
    log(spec.instance_id, f"count {step['count']} from {json.dumps(estimate, ensure_ascii=False)[:300]}")
    return step["count"]


def run_recipe_step(client, spec, state, run_name):
    step = state.step("recipe")
    if step.get("dataset_id"):
        return step["dataset_id"]
    body = {
        "name": f"WideSearch {run_name} {spec.instance_id}",
        "prompt": spec.enrichment_prompt(),
        "input_columns": spec.input_column_names,
        "output": spec.output_columns(),
    }
    created = client.request("POST", "/enrichments", body)
    step.update(enrichment_id=created["id"], dataset_id=created["dataset_id"], app_url=created.get("app_url"),
                budget_credits_per_row=created.get("budget_credits_per_row"), created_at=utc_now())
    state.save()
    log(spec.instance_id, f"recipe {created['id']} dataset {created['dataset_id']} {created.get('app_url')}")
    return step["dataset_id"]


def start_build(client, spec, step, dataset_id, count, auto_fill):
    """With auto_fill, Riveter starts the fill itself when the rows land, so nothing waits on this machine;
    the estimate's maximum then covers the build budget plus every row's fill budget."""
    budget = build_budget_for(count)
    step.update(count=count, budget=budget, auto_fill=auto_fill)
    earlier_build = client.existing_run_of_type(dataset_id, "dataset_build")
    if earlier_build:
        step.update(run_id=earlier_build["id"], adopted_existing_run=True, started_at=utc_now())
        return
    body = {"prompt": spec.build_prompt(), "count": count, "budget": budget, "auto_fill": auto_fill}
    estimate = client.request("POST", f"/datasets/{dataset_id}/rows/build", {**body, "estimate_cost": True})
    step.update(estimate=estimate.get("credit_detail"))
    maximum = ((estimate.get("credit_detail") or {}).get("estimate") or {}).get("maximum", budget)
    started = client.request("POST", f"/datasets/{dataset_id}/rows/build", {**body, "max_credits": maximum})
    step.update(run_id=started["run"]["id"], started_at=utc_now())


def ensure_build_started(client, spec, state, dataset_id, count, auto_fill):
    step = state.step("build")
    if step.get("run_id"):
        return step
    start_build(client, spec, step, dataset_id, count, auto_fill)
    state.save()
    log(spec.instance_id, f"build {step['run_id']} count={count} budget={step['budget']} auto_fill={auto_fill}")
    return step


def run_build_step(client, spec, state, dataset_id, count, auto_fill):
    step = state.step("build")
    if "rows_saved" in step:
        return step["rows_saved"]
    ensure_build_started(client, spec, state, dataset_id, count, auto_fill)
    run = client.wait_for_run(spec.instance_id, step["run_id"], BUILD_DEADLINE_SECONDS, "build")
    rows = client.all_rows(dataset_id)
    state.write_rows("build_rows.json", rows)
    step.update(run_envelope_summary(run), rows_saved=len(rows), saved_at=utc_now())
    state.save()
    log(spec.instance_id, f"build done status={run.get('status')} rows={len(rows)} credits={run.get('credits_used')}")
    return len(rows)


def start_fill(client, spec, step, dataset_id, run_key_prefix):
    earlier_fill = client.existing_run_of_type(dataset_id, "enrichment")
    if earlier_fill:
        step.update(run_id=earlier_fill["id"], adopted_existing_run=True, started_at=utc_now())
        return
    estimate = client.request("POST", "/enrich", {"dataset_id": dataset_id, "estimate_cost": True})
    maximum = ((estimate.get("credit_detail") or {}).get("estimate") or {}).get("maximum")
    step.update(estimate=estimate.get("credit_detail"))
    body = {"dataset_id": dataset_id, "run_key": f"{run_key_prefix}-fill-{spec.instance_id}"}
    if maximum is not None:
        body["max_credits"] = maximum
    started = client.start_keyed_run("/enrich", body)
    step.update(run_id=started["id"], started_at=utc_now())


def run_fill_step(client, spec, state, dataset_id, run_key_prefix):
    step = state.step("fill")
    if "rows_saved" in step:
        return
    if not step.get("run_id"):
        start_fill(client, spec, step, dataset_id, run_key_prefix)
        state.save()
        log(spec.instance_id, f"fill {step['run_id']} estimate_max={((step.get('estimate') or {}).get('estimate') or {}).get('maximum')}")
    run = client.wait_for_run(spec.instance_id, step["run_id"], FILL_DEADLINE_SECONDS, "fill")
    rows = client.all_rows(dataset_id)
    state.write_rows("fill_rows.json", rows)
    step.update(run_envelope_summary(run), rows_saved=len(rows), saved_at=utc_now())
    state.save()
    log(spec.instance_id, f"fill done status={run.get('status')} rows={len(rows)} credits={run.get('credits_used')}")


def run_task(client, spec, run_dir, run_name, options):
    state = TaskState(run_dir, spec.instance_id)
    run_key_prefix = f"ws-{run_name}"
    try:
        count = run_count_step(client, spec, state, run_key_prefix)
        dataset_id = run_recipe_step(client, spec, state, run_name)
        if options.kickoff_only:
            ensure_build_started(client, spec, state, dataset_id, count, options.auto_fill)
        else:
            built_rows = run_build_step(client, spec, state, dataset_id, count, options.auto_fill)
            if built_rows and not options.skip_fill:
                run_fill_step(client, spec, state, dataset_id, run_key_prefix)
        state.data.pop("last_error", None)
    except Exception as error:  # one task's failure must not stop the others; the rerun resumes it
        state.data["last_error"] = {"at": utc_now(), "message": str(error)[:2000]}
        log(spec.instance_id, f"ERROR {error}")
    state.save()


def selected_specs(tasks_arg):
    if tasks_arg == "pilot":
        return load_task_specs(PILOT_INSTANCE_IDS)
    if tasks_arg == "all":
        return load_task_specs()
    return load_task_specs([instance_id.strip() for instance_id in tasks_arg.split(",")])


def command_plan(arguments):
    specs = selected_specs(arguments.tasks)
    print(f"{len(specs)} tasks. Per task: count 1 credit; build budget {BUILD_BUDGET_FLOOR_CREDITS} + "
          f"{BUILD_BUDGET_CREDITS_PER_ROW} x count (max {BUILD_BUDGET_MAX_CREDITS}); fill cap rows x 1.5 x output columns.")
    for spec in specs.values():
        problems = f"  PROBLEMS: {spec.problems}" if spec.problems else ""
        print(f"{spec.instance_id}  inputs={len(spec.input_column_names)}  outputs={len(spec.output_columns())}  "
              f"keys={[column.display_name for column in spec.key_columns]}  placeholder={spec.missing_value_placeholder!r}{problems}")


def command_run(arguments):
    specs = selected_specs(arguments.tasks)
    run_dir = RUN_ROOT / "runs" / arguments.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs.values():
        (run_dir / spec.instance_id).mkdir(exist_ok=True)
        (run_dir / spec.instance_id / "spec.json").write_text(json.dumps(spec.to_json(), indent=2, ensure_ascii=False))
    client = RiveterClient(load_api_key())
    account = client.get_with_retry("/account")
    log("-", f"account balance before run: {json.dumps(account)[:400]}")
    with ThreadPoolExecutor(max_workers=arguments.parallel) as executor:
        for spec in specs.values():
            executor.submit(run_task, client, spec, run_dir, arguments.run_name, arguments)
    account = client.get_with_retry("/account")
    log("-", f"account balance after run: {json.dumps(account)[:400]}")


def command_status(arguments):
    run_dir = RUN_ROOT / "runs" / arguments.run_name
    totals = {"count": 0.0, "build": 0.0, "fill": 0.0}
    for state_path in sorted(run_dir.glob("*/state.json")):
        state = json.loads(state_path.read_text())
        cells = []
        for step_name in totals:
            step = state.get(step_name, {})
            totals[step_name] += float(step.get("credits_used") or 0)
            cells.append(f"{step_name}={step.get('status', '-')}/{step.get('credits_used', '-')}")
        rows = f"rows={state.get('build', {}).get('rows_saved', '-')} count={state.get('count', {}).get('count', '-')}"
        error = f"  ERROR: {state['last_error']['message'][:160]}" if state.get("last_error") else ""
        print(f"{state['instance_id']}  {'  '.join(cells)}  {rows}{error}")
    print(f"credits: {json.dumps({name: round(value, 2) for name, value in totals.items()})} total={round(sum(totals.values()), 2)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="print the task specs; no network")
    plan.add_argument("--tasks", default="pilot", help="pilot, all, or comma-separated instance ids")
    run = commands.add_parser("run", help="count, recipe, build, fill for each task (spends credits)")
    run.add_argument("--run-name", required=True)
    run.add_argument("--tasks", default="pilot")
    run.add_argument("--parallel", type=int, default=4)
    run.add_argument("--skip-fill", action="store_true", help="stop after the build")
    run.add_argument("--auto-fill", action="store_true", help="Riveter starts each fill itself when the build's rows land")
    run.add_argument("--kickoff-only", action="store_true",
                     help="start every task's count and build, then exit; a later run without it collects the rows")
    status = commands.add_parser("status", help="per-task state and credits of a run; no network")
    status.add_argument("--run-name", required=True)
    arguments = parser.parse_args()
    {"plan": command_plan, "run": command_run, "status": command_status}[arguments.command](arguments)


if __name__ == "__main__":
    main()
