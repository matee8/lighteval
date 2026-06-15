import dataclasses
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib import request
from urllib.request import Request

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

logger = logging.getLogger(__name__)

__all__ = ["TASKS_TABLE"]

_MOCK_JUDGE_WARNING_SHOWN = False


def discover_local_datasets(data_dir: Path) -> Dict[str, List[str]]:
    if not data_dir.is_dir():
        raise NotADirectoryError("Dataset directory does not exist or is not a directory: "
                                 f"{data_dir}")

    data_files: Dict[str, List[str]] = {}

    for file_path in data_dir.glob("*.json"):
        file_name = file_path.stem.lower()
        if "emelt" in file_name:
            level = "advanced"
        elif "közép" in file_name or "kozep" in file_name:
            level = "standard"
        else:
            logger.warning("File %s missing level (emelt/közép). Skipping.", file_path)
            continue

        if "matematika" in file_name or "matek" in file_name:
            subject = "math"
        elif "fizika" in file_name:
            if "első" in file_name or "elso" in file_name:
                subject = "physics_part1"
            elif "második" in file_name or "masodik" in file_name:
                subject = "physics_part2"
            else:
                logger.warning("Physics file %s missing part (első/második). Skipping.", file_path)
                continue
        else:
            logger.warning("File %s missing subject (matematika/fizika). Skipping.", file_path)
            continue

        key = f"{subject}_{level}"
        if key not in data_files:
            data_files[key] = []

        data_files[key].append(str(file_path.absolute()))

    if not data_files:
        logger.warning("No valid dataset files found in %s.", data_dir)

    return data_files


def preprocess_datasets(data_files: Dict[str, List[str]]) -> Dict[str, List[str]]:
    preprocessed_files: Dict[str, List[str]] = {}

    temp_dir = Path(tempfile.gettempdir()) / "humatura_preprocessed"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for task_key, file_paths in data_files.items():
        preprocessed_files[task_key] = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from %s.", file_path)
                    continue

            merged_tasks: Dict[str, Any] = {}
            for item in data:
                if item.get("skipped"):
                    continue

                task_id = item.get("task")
                if not task_id:
                    logger.warning("Item missing 'task' key in %s. Skipping.", file_path)
                    continue

                if task_id not in merged_tasks:
                    merged_item = item.copy()

                    merged_item["solutions"] = [item["solution"]]
                    merged_item.pop("solution", None)
                    merged_tasks[task_id] = merged_item
                else:
                    merged_tasks[task_id]["solutions"].append(item["solution"])

            out_path = temp_dir / f"{Path(file_path).stem}_preprocessed.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(list(merged_tasks.values()), f, ensure_ascii=False, indent=2)

            preprocessed_files[task_key].append(str(out_path.absolute()))

    return preprocessed_files


def math_prompt_fn(line: Dict[str, Any], task_name: str = "") -> Doc:
    instruction = (
        "Kérlek, oldd meg a következő matematikai feladatot. "
        "Gondoljuk végig lépésről lépésre, részletesen. "
        "A végső választ pontosan egy \\boxed{} formátumba írd be "
        "(például \\boxed{42} vagy \\boxed{x=3})."
    )

    description: str = line.get("description", "")
    query = f"{instruction}\n\nFeladat:\n{description}\n\nMegoldás:\n"

    solutions = [sol.get("solution", sol) for sol in line.get("solutions", [])]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[],
        gold_index=[],
        specific={"solutions": solutions},
    )


def physics_part1_prompt_fn(line: Dict[str, Any], task_name: str = "") -> Doc:
    instruction = (
        "Kérlek, válaszolj a következő feleletválasztós fizika feladatra. "
        "A végső válaszod betűjelét (A, B, C vagy D) pontosan egy \\boxed{} formátumba írd be "
        "(például \\boxed{A} vagy \\boxed{C})."
    )

    description: str = line.get("description", "")

    first_solution = line.get("solutions", [{}])[0]
    answers: Dict[str, str] = first_solution.get("answers", {})
    correct_answer: str = first_solution.get("correct_answer", "")

    choices_text = "\n".join(f"{letter}) {text}" for letter, text in answers.items())

    query = (f"{instruction}\n\nFeladat:\n{description}\n\n"
            f"Válaszlehetőségek:\n{choices_text}\n\nMegoldás:\n")

    return Doc(
        task_name=task_name,
        query=query,
        choices=[],
        gold_index=[],
        specific={"correct_answer": correct_answer},
    )


def physics_part2_prompt_fn(line: Dict[str, Any], task_name: str = "") -> Doc:
    instruction = (
        "Kérlek, oldd meg a következő fizika feladatot. "
        "Gondoljuk végig lépésről lépésre, részletesen, és add meg a végső választ."
    )

    description: str = line.get("description", "")
    query = f"{instruction}\n\nFeladat:\n{description}\n\nMegoldás:\n"

    solutions = [sol.get("solution", sol) for sol in line.get("solutions", [])]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[],
        gold_index=[],
        specific={"solutions": solutions},
    )


def extract_boxed_answer(text: str) -> Optional[str]:
    box_tag = "\\boxed{"
    start_idx = text.find(box_tag)

    if start_idx == -1:
        return None

    start_idx += len(box_tag)
    brace_count = 1

    for i in range(start_idx, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1

        if brace_count == 0:
            return text[start_idx:i]

    return None


class PhysicsMultipleChoiceMatch(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs: Any) -> Dict[str, float]:
        if not model_response.final_text:
            return {"exact_match": 0.0}

        if doc.specific is None or "correct_answer" not in doc.specific:
            logger.debug("Formatted doc is missing the specific correct_answer field.")
            return {"exact_match": 0.0}

        prediction = model_response.final_text[0]
        extracted_pred = extract_boxed_answer(prediction)

        if extracted_pred is None:
            logger.debug("Failed to extract boxed answer from multiple choice prediction.")
            return {"exact_match": 0.0}

        gold_answer: str = str(doc.specific["correct_answer"]).strip().upper()
        pred_clean: str = extracted_pred.strip().upper()

        score: float = 1.0 if pred_clean == gold_answer else 0.0

        return {"exact_match": score}


def fallback_mock_judge(prediction: str, solutions: List[Any]) -> bool:
    global _MOCK_JUDGE_WARNING_SHOWN
    if not _MOCK_JUDGE_WARNING_SHOWN:
        logger.warning("JUDGE_API_KEY is not set. Falling back to local substring matching.")
        _MOCK_JUDGE_WARNING_SHOWN = True

    pred_clean = prediction.replace(" ", "").lower()

    for sol in solutions:
        gold_answer = str(sol.get("final-answer", "")).replace(" ", "").strip("$").lower()
        if gold_answer and gold_answer in pred_clean:
            return True

    return False

def call_llm_judge(query: str, prediction: str, solutions: List[Any]) -> bool:
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_url = os.environ.get("JUDGE_API_URL", "https://api.openai.com/v1/chat/completions")
    model_name = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")

    if not api_key:
        return fallback_mock_judge(prediction, solutions)

    formatted_solutions = ""
    for idx, sol in enumerate(solutions, 1):
        final_ans = sol.get("final-answer", "")
        steps = sol.get("steps", [])
        steps_text = "\n".join(f"- {step.get("step-description", "")}" for step in steps)
        formatted_solutions += (f"Option {idx}:\n  Final Answer: {final_ans}\n  "
                                f"Steps:\n{steps_text}\n\n")

    prompt = (
        "You are an expert academic grader. Your task is to grade a student's answer to a mathematical or physics question.\n\n"
        f"Question:\n{query}\n\n"
        f"Reference Solution(s):\n{formatted_solutions}\n"
        f"Student's Answer:\n{prediction}\n\n"
        "Determine if the student's final answer is correct and mathematically/physically equivalent to any of the reference solutions.\n"
        "Ignore step-by-step reasoning or calculation differences as long as the final conclusion/result is correct.\n\n"
        "Response format:\n"
        "Respond in exactly the following JSON format:\n"
        "{\n"
        '    "correct": true,\n'
        '    "explanation": "Brief explanation of your grading decision"\n'
        "}\n"
        "Do not include any other markdown formatting, code block markers, or text outside of the JSON block."
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }

    try:
        req = Request(api_url,
                      data=json.dumps(payload).encode("utf-8"),
                      headers=headers,
                      method="POST")

        with request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"].strip()

            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            result = json.loads(content)
            return bool(result.get("correct", False))
    except Exception as error:
        logger.error("LLM Judge API call failed: %s. Falling back to local heuristics.", error)
        return fallback_mock_judge(prediction, solutions)


class LLMJudgeEquivalence(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs: Any) -> Dict[str, float]:
        if not model_response.final_text:
            return {"llm_judge_score": 0.0}

        if doc.specific is None or "solutions" not in doc.specific:
            logger.debug("Formatted doc is missing the specific solutions array.")
            return {"llm_judge_score": 0.0}

        prediction = model_response.final_text[0]
        solutions: List[Any] = doc.specific["solutions"]

        if not solutions:
            return {"llm_judge_score": 0.0}

        first_sol = solutions[0]
        steps = first_sol.get("steps", [])

        if steps:
            max_points = float(sum(step.get("points", 0) for step in steps))
        else:
            max_points = float(first_sol.get("points", 1.0))

        is_correct = call_llm_judge(doc.query, prediction, solutions)
        score = max_points if is_correct else 0.0

        return {"llm_judge_score": score}


def aggregate_exact_match_scores(scores: List[Any]) -> float:
    if not scores:
        return 0.0
    if isinstance(scores[0], dict):
        unpacked_scores = [s.get("exact_match", 0.0) for s in scores]
        return sum(unpacked_scores) / len(unpacked_scores)
    return sum(scores) / len(scores)


def aggregate_judge_scores(scores: List[Any]) -> float:
    if not scores:
        return 0.0
    if isinstance(scores[0], dict):
        unpacked_scores = [s.get("llm_judge_score", 0.0) for s in scores]
        return sum(unpacked_scores) / len(unpacked_scores)
    return sum(scores) / len(scores)


physics_mc_metric: SampleLevelMetric = SampleLevelMetric(
    metric_name="physics_multiple_choice_accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=PhysicsMultipleChoiceMatch(),
    corpus_level_fn=aggregate_exact_match_scores,
)

llm_judge_metric: SampleLevelMetric = SampleLevelMetric(
    metric_name="llm_judge_score",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=LLMJudgeEquivalence(),
    corpus_level_fn=aggregate_judge_scores,
)

@dataclasses.dataclass
class SubjectTemplate:
    prompt_fn: Callable[[Dict[str, Any], str], Doc]
    metrics: List[Any]


_SUBJECT_TEMPLATES: Dict[str, SubjectTemplate] = {
    "math": SubjectTemplate(prompt_fn=math_prompt_fn, metrics=[llm_judge_metric]),
    "physics_part1": SubjectTemplate(prompt_fn=physics_part1_prompt_fn, metrics=[physics_mc_metric]),
    "physics_part2": SubjectTemplate(prompt_fn=physics_part2_prompt_fn, metrics=[llm_judge_metric]),
}

TASKS_TABLE: List[LightevalTaskConfig] = []

_DATA_DIR = Path(os.environ.get("HUMATURA_DATA_DIR", "./data"))

try:
    _discovered_files = discover_local_datasets(_DATA_DIR)
    _discovered_files = preprocess_datasets(_discovered_files)
except NotADirectoryError:
    logger.warning("Data directory %s not found. Tasks will be registered without local files.",
                   _DATA_DIR)
    _discovered_files = {}

for task_key, all_files_for_subset in _discovered_files.items():
    if not all_files_for_subset:
        continue

    subject, level = task_key.rsplit("_", 1)

    template = _SUBJECT_TEMPLATES.get(subject)
    if not template:
        logger.warning("No configuration template found for subject %s. Skipping.", subject)
        continue

    task_name = f"humatura:{subject}:{level}"


    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=task_name,
            prompt_function=template.prompt_fn,
            hf_repo="json",
            hf_subset="default",
            hf_data_files={"validation": all_files_for_subset},
            hf_avail_splits=["validation"],
            evaluation_splits=["validation"],
            metrics=template.metrics,
        )
    )
