"""Grades response files with the WideSearch harness's own eval stage, unchanged except for the judge's HTTP client:
the harness builds an AzureOpenAI client, so the judge calls here go through the plain OpenAI client with the same
model (default_eval_config, gpt-4.1-2025-04-14), messages, and generate kwargs, and the harness's retry policy.

With --retry-incomplete-judge-replies (a disclosed deviation, its own results directory <output>-rejudged): an
llm_judge reply that does not score every pair — unparseable JSON, or a JSON that lists only some idx keys, which the
harness reads as 0 for every missing pair — is asked again, up to JUDGE_ATTEMPTS times. On a 533-row column the
harness's single call does both.

Usage:
  python grade.py --run-name pilot-v1 --output riveter-verified --tasks pilot
Needs OPENAI_API_KEY in .env (the judge spend is on that key). Results land in results/<run_name>/<output>/.
"""

import argparse
import os
import runpy
import sys

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_incrementing

from to_harness import HARNESS_ROOT
from widesearch_tasks import PILOT_INSTANCE_IDS, RUN_ROOT, load_raw_tasks

JUDGE_TIMEOUT_SECONDS = 300
JUDGE_ATTEMPTS = 3
REJUDGED_SUFFIX = "-rejudged"


def load_openai_key():
    env_path = RUN_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENAI_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    sys.exit(f"OPENAI_API_KEY is not set: add it to {env_path}")


def patch_judge_client(api_key):
    sys.path.insert(0, str(HARNESS_ROOT))
    os.chdir(HARNESS_ROOT)
    import src.utils.llm as harness_llm

    client = OpenAI(api_key=api_key, timeout=JUDGE_TIMEOUT_SECONDS)

    @retry(stop=stop_after_attempt(8), wait=wait_incrementing(8, 8))
    def openai_complete(base_url, api_key, messages, tools=None, model_name="gpt-4o-2024-05-13", retry_if_empty=False, **generate_kwargs):
        completion = client.chat.completions.create(messages=messages, model=model_name, tools=tools, **generate_kwargs)
        message = completion.choices[0].message if completion.choices else None
        if retry_if_empty and message is not None and not message.content and not message.tool_calls:
            raise RuntimeError("[openai_complete] Got message, but content and toolcalls is empty, retry")
        return message

    harness_llm.openai_complete = openai_complete


def judge_reply_scores_every_pair(msg_list, pair_count):
    from src.evaluation.metric_utils import parse_markdown_json

    scores = parse_markdown_json(str(msg_list[0])) if msg_list else None
    return isinstance(scores, dict) and all(f"idx_{index}" in scores for index in range(pair_count))


def retry_incomplete_judge_replies():
    import src.evaluation.evaluation as harness_evaluation

    harness_llm_judge_column = harness_evaluation.llm_judge_column

    def llm_judge_column_with_retry(response, target, criterion, model_config_name):
        for _ in range(JUDGE_ATTEMPTS):
            score_list, msg_list = harness_llm_judge_column(response, target, criterion, model_config_name)
            if judge_reply_scores_every_pair(msg_list, len(response)):
                break
        return score_list, msg_list

    harness_evaluation.llm_judge_column = llm_judge_column_with_retry


def instance_ids_for(tasks_arg):
    if tasks_arg == "pilot":
        return PILOT_INSTANCE_IDS
    if tasks_arg == "all":
        return sorted(load_raw_tasks())
    return [instance_id.strip() for instance_id in tasks_arg.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", required=True, help="riveter-build, riveter-verified, or riveter-verified-in-scope")
    parser.add_argument("--tasks", default="pilot", help="pilot, all, or comma-separated instance ids")
    parser.add_argument("--thread-num", type=int, default=4)
    parser.add_argument("--retry-incomplete-judge-replies", action="store_true")
    arguments = parser.parse_args()

    response_root = RUN_ROOT / "responses" / arguments.run_name / arguments.output
    result_suffix = REJUDGED_SUFFIX if arguments.retry_incomplete_judge_replies else ""
    result_root = RUN_ROOT / "results" / arguments.run_name / f"{arguments.output}{result_suffix}"
    result_root.mkdir(parents=True, exist_ok=True)
    patch_judge_client(load_openai_key())
    if arguments.retry_incomplete_judge_replies:
        retry_incomplete_judge_replies()
    sys.argv = [
        "run_infer_and_eval_batching.py", "--stage", "eval", "--trial_num", "1",
        "--model_config_name", arguments.output,
        "--response_root", str(response_root), "--result_save_root", str(result_root),
        "--instance_id", ",".join(instance_ids_for(arguments.tasks)),
        "--thread_num", str(arguments.thread_num),
    ]
    runpy.run_path(str(HARNESS_ROOT / "scripts" / "run_infer_and_eval_batching.py"), run_name="__main__")


if __name__ == "__main__":
    main()
