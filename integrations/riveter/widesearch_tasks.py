"""One WideSearch task as the Riveter recipe the runner sends, derived from the task file alone.

Read from a task: the query text, and the evaluation's `required` and `unique_columns` names (the table's
column names and its key, both also stated by the query). Never read: the gold CSV, the evaluation's
metrics, criteria, or preprocessors.
"""

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parent
DATASET_DIR = RUN_ROOT / "hf_dataset"
DATASET_PATH = DATASET_DIR / "widesearch.jsonl"
DATASET_REPO_ID = "ByteDance-Seed/WideSearch"

PILOT_INSTANCE_IDS = [
    "ws_en_011", "ws_en_028", "ws_en_021", "ws_en_033", "ws_en_014",
    "ws_en_015", "ws_en_026", "ws_en_039", "ws_en_013", "ws_en_006",
]

VERIFIED_COLUMN_SUFFIX = " (verified)"
BELONGS_COLUMN_NAME = "Belongs in the table"
MISSING_VALUE_TOKENS = ("/", "-", "–", "—")
MAX_RIVETER_COLUMN_NAME_LENGTH = 100
FUZZY_DISPLAY_NAME_MIN_RATIO = 0.8

BUILD_PROMPT_SUFFIX = (
    "\n\nEvery column named above is part of the result: write a value in each column for every row, "
    "taken from the sources you read. Leave a cell empty only when no source gives it."
)

ENRICHMENT_TASK_PROMPT_TEMPLATE = """This dataset is the table the request below asks for, one dataset row per table row. A list-building pass wrote each row: the input columns hold the row's key ({key_names}) and a draft value for every other column of the table.

For this row, fill each output column with the correct value of the table column it names. Check the draft against a source. If the draft is right, give it again unchanged; if it is wrong or empty, give the correct value. Write every value in the format the request asks for. Never change the row's key.

REQUEST:
{query}"""

VERIFIED_COLUMN_PROMPT_TEMPLATE = "The row's “{display_name}”, as the request defines and formats it. The draft is in the input column “{display_name}”."

BELONGS_COLUMN_PROMPT = (
    "true when this row belongs in the table the request asks for: its key names a real item that meets every "
    "condition in the request (time range, place, category, ranking). false when it does not."
)

GENERAL_MISSING_VALUE_PHRASE = re.compile(
    r"\b(any|all|some)\b.*\b(information|data|field|fields|content|cell|cells|statistic|value|values)\b"
    r"|\b(can't|cannot|can not|could not|couldn't|not)\b.*\b(find|found|retrieved|located|available)\b"
    r"|\bno relevant\b",
    re.IGNORECASE,
)
QUOTED_TOKEN = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{1,4})[\"'“”‘’]")
CONDITION_START = re.compile(r"\b(if|for|when)\b", re.IGNORECASE)


def norm_column(name):
    """The harness's column normalization (src/utils/utils.py)."""
    return name.strip().lower().replace(" ", "")


def riveter_header_key(name):
    """How Datasets::ImportListGenerationResult matches the list builder's headers to input columns."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass
class TaskColumn:
    required_name: str
    display_name: str
    is_key: bool

    @property
    def verified_column_name(self):
        return f"{self.display_name}{VERIFIED_COLUMN_SUFFIX}"


@dataclass
class TaskSpec:
    instance_id: str
    query: str
    columns: list
    row_id_column_name: str | None
    missing_value_placeholder: str | None
    problems: list = field(default_factory=list)

    @property
    def key_columns(self):
        return [column for column in self.columns if column.is_key]

    @property
    def value_columns(self):
        return [column for column in self.columns if not column.is_key]

    @property
    def input_column_names(self):
        leading = [self.row_id_column_name] if self.row_id_column_name else []
        keys_first = self.key_columns + self.value_columns
        return leading + [column.display_name for column in keys_first]

    def build_prompt(self):
        prompt = self.query.strip()
        if self.row_id_column_name:
            key_names = " and ".join(f"“{column.display_name}”" for column in self.key_columns)
            prompt += f'\n\nThe column “{self.row_id_column_name}” holds the row\'s {key_names} joined with " | ".'
        return prompt + BUILD_PROMPT_SUFFIX

    def enrichment_prompt(self):
        key_names = ", ".join(column.display_name for column in self.key_columns)
        return ENRICHMENT_TASK_PROMPT_TEMPLATE.format(key_names=key_names, query=self.query.strip())

    def output_columns(self):
        verified = [
            {
                "name": column.verified_column_name,
                "prompt": VERIFIED_COLUMN_PROMPT_TEMPLATE.format(display_name=column.display_name),
                "output_schema": {"type": "string"},
            }
            for column in self.value_columns
        ]
        belongs = {"name": BELONGS_COLUMN_NAME, "prompt": BELONGS_COLUMN_PROMPT, "output_schema": {"type": "boolean"}}
        return verified + [belongs]

    def to_json(self):
        return {
            "instance_id": self.instance_id,
            "columns": [vars(column) for column in self.columns],
            "row_id_column_name": self.row_id_column_name,
            "missing_value_placeholder": self.missing_value_placeholder,
            "input_column_names": self.input_column_names,
            "build_prompt": self.build_prompt(),
            "enrichment_prompt": self.enrichment_prompt(),
            "output_columns": self.output_columns(),
            "problems": self.problems,
        }


def ensure_dataset_downloaded():
    if DATASET_PATH.exists():
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset", local_dir=DATASET_DIR)


def load_raw_tasks(language="en"):
    ensure_dataset_downloaded()
    tasks = {}
    with open(DATASET_PATH) as dataset_file:
        for line in dataset_file:
            task = json.loads(line)
            if task["language"] == language:
                task["evaluation"] = json.loads(task["evaluation"])
                tasks[task["instance_id"]] = task
    return tasks


def display_name_pattern(required_name):
    spaced_characters = r"\s*".join(re.escape(character) for character in required_name)
    return re.compile(r"(?<![A-Za-z0-9])" + spaced_characters + r"(?![A-Za-z0-9])", re.IGNORECASE)


def column_list_line(query, required_names):
    """The query line that names the most table columns: the line the column names are read from first."""
    def matches_in(line):
        return sum(1 for name in required_names if display_name_pattern(name).search(line))

    return max(query.split("\n"), key=matches_in)


def split_outside_parentheses(text, separators):
    parts, current, depth = [], [], 0
    for character in text:
        depth += {"(": 1, ")": -1}.get(character, 0)
        if character in separators and depth <= 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(character)
    return parts + ["".join(current)]


def query_segments(query):
    segments = []
    for line in query.split("\n"):
        for piece in re.split(r"[:：]", line):
            for part in split_outside_parentheses(piece, ",;，|"):
                segment = re.sub(r"^(and|or)\s+", "", part.strip().rstrip(".").strip(), flags=re.IGNORECASE)
                if segment:
                    segments.append(segment)
    return segments


def fuzzy_display_name(required_name, query):
    scored = [
        (difflib.SequenceMatcher(None, norm_column(segment), required_name).ratio(), segment)
        for segment in query_segments(query)
    ]
    best_ratio, best_segment = max(scored, default=(0.0, None))
    return best_segment if best_ratio >= FUZZY_DISPLAY_NAME_MIN_RATIO else None


def display_name_for(required_name, query, preferred_line):
    pattern = display_name_pattern(required_name)
    for text in (preferred_line, query):
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(0).strip()
    return fuzzy_display_name(required_name, query) or required_name


def row_id_column_name_for(key_columns):
    if len(key_columns) < 2:
        return None
    return "Row ID (" + " + ".join(column.display_name for column in key_columns) + ")"


def names_a_column(sentence, columns):
    lowered = sentence.lower()
    return any(len(column.display_name) >= 4 and column.display_name.lower() in lowered for column in columns)


def condition_before(sentence, token_start):
    """The clause that says when the token applies: from the last if / for / when before the token, else the sentence."""
    starts = [match.start() for match in CONDITION_START.finditer(sentence, 0, token_start)]
    return sentence[starts[-1]:token_start] if starts else sentence


def missing_value_placeholder_for(query, columns):
    """The token the query asks for in every cell it cannot fill ("For any missing information, fill in '/'").
    NA-style tokens are left out: the harness reads them and an empty cell as the same missing value."""
    tokens = set()
    for sentence in re.split(r"[.\n]", query):
        for match in QUOTED_TOKEN.finditer(sentence):
            token = match.group(1).strip()
            condition = condition_before(sentence, match.start())
            if token in MISSING_VALUE_TOKENS and GENERAL_MISSING_VALUE_PHRASE.search(condition) and not names_a_column(sentence, columns):
                tokens.add(token)
    return tokens.pop() if len(tokens) == 1 else None


def spec_problems(spec):
    problems = []
    header_keys = [riveter_header_key(name) for name in spec.input_column_names]
    if len(set(header_keys)) != len(header_keys):
        problems.append(f"input column names collide after header normalization: {spec.input_column_names}")
    too_long = [name for name in spec.input_column_names + [c["name"] for c in spec.output_columns()]
                if len(name) > MAX_RIVETER_COLUMN_NAME_LENGTH]
    if too_long:
        problems.append(f"column names over {MAX_RIVETER_COLUMN_NAME_LENGTH} characters: {too_long}")
    unmatched = [column.required_name for column in spec.columns if norm_column(column.display_name) != column.required_name]
    if unmatched:
        problems.append(f"query names these columns differently from the harness: {unmatched}")
    not_in_query = [column.display_name for column in spec.columns if column.display_name not in spec.query]
    if not_in_query:
        problems.append(f"display names not found in the query: {not_in_query}")
    return problems


def task_spec(raw_task):
    query = raw_task["query"]
    required_names = raw_task["evaluation"]["required"]
    unique_names = set(raw_task["evaluation"]["unique_columns"])
    preferred_line = column_list_line(query, required_names)
    columns = [
        TaskColumn(required_name=name, display_name=display_name_for(name, query, preferred_line), is_key=name in unique_names)
        for name in required_names
    ]
    spec = TaskSpec(
        instance_id=raw_task["instance_id"],
        query=query,
        columns=columns,
        row_id_column_name=row_id_column_name_for([column for column in columns if column.is_key]),
        missing_value_placeholder=missing_value_placeholder_for(query, columns),
    )
    spec.problems = spec_problems(spec)
    return spec


def load_task_specs(instance_ids=None, language="en"):
    raw_tasks = load_raw_tasks(language)
    selected = instance_ids or sorted(raw_tasks)
    return {instance_id: task_spec(raw_tasks[instance_id]) for instance_id in selected}
