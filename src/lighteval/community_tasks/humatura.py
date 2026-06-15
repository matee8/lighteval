import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

logger = logging.getLogger(__name__)

__all__ = ["TASKS_TABLE"]


def discover_local_datasets(data_dir: Path) -> Dict[str, List[str]]:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist or is not a directory: {data_dir}")

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

    first_solution = line.get("solutions", [{}])[0]
    solution_obj = first_solution.get("solution", first_solution)

    return Doc(
        task_name=task_name,
        query=query,
        choices=[],
        gold_index=[],
        specific={"solution": solution_obj},
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

    query = f"{instruction}\n\nFeladat:\n{description}\n\nVálaszlehetőségek:\n{choices_text}\n\nMegoldás:\n"

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


class HungarianMathEquivalence(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs: Any) -> Dict[str, float]:
        if not model_response.final_text:
            return {"math_equivalence": 0.0}

        if doc.specific is None or "solution" not in doc.specific:
            logger.debug("Formatted doc is missing the specific solution dictionary.")
            return {"math_equivalence": 0.0}

        prediction = model_response.final_text[0]
        extracted_pred = extract_boxed_answer(prediction)

        if extracted_pred is None:
            logger.debug("Failed to extract boxed answer from prediction.")
            return {"math_equivalence": 0.0}

        solution_dict: Any = doc.specific["solution"]
        if not isinstance(solution_dict, dict):
            return {"math_equivalence": 0.0}

        gold_answer: str = solution_dict.get("final-answer", "").strip("$")

        pred_clean: str = extracted_pred.replace(" ", "").lower()
        gold_clean: str = gold_answer.replace(" ", "").lower()

        score: float = 1.0 if pred_clean == gold_clean else 0.0

        return {"math_equivalence": score}


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


def aggregate_scores(scores: List[Any]) -> float:
    if not scores:
        return 0.0

    if isinstance(scores[0], dict):
        unpacked_scores = [s.get("math_equivalence", 0.0) for s in scores]
        return sum(unpacked_scores) / len(unpacked_scores)

    return sum(scores) / len(scores)


def aggregate_exact_match_scores(scores: List[Any]) -> float:
    if not scores:
        return 0.0

    if isinstance(scores[0], dict):
        unpacked_scores = [s.get("exact_match", 0.0) for s in scores]
        return sum(unpacked_scores) / len(unpacked_scores)

    return sum(scores) / len(scores)


hungarian_math_metric: SampleLevelMetric = SampleLevelMetric(
    metric_name="hungarian_math_final_answer",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=HungarianMathEquivalence(),
    corpus_level_fn=aggregate_scores,
)

physics_mc_metric: SampleLevelMetric = SampleLevelMetric(
    metric_name="physics_multiple_choice_accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=PhysicsMultipleChoiceMatch(),
    corpus_level_fn=aggregate_exact_match_scores,
)

_PROMPT_FNS: Dict[str, Callable[[Dict[str, Any], str], Doc]] = {
    "math": math_prompt_fn,
    "physics_part1": physics_part1_prompt_fn,
    "physics_part2": physics_part2_prompt_fn,
}

TASKS_TABLE: List[LightevalTaskConfig] = []

_DATA_DIR = Path(os.environ.get("HUMATURA_DATA_DIR", "./data"))

try:
    _discovered_files = discover_local_datasets(_DATA_DIR)
    _discovered_files = preprocess_datasets(_discovered_files)
except NotADirectoryError:
    logger.warning("Data directory %s not found. Tasks will be registered without local files.", _DATA_DIR)
    _discovered_files = {}

for task_key, all_files_for_subset in _discovered_files.items():
    if not all_files_for_subset:
        continue

    subject, level = task_key.rsplit("_", 1)
    task_name = f"humatura:{subject}:{level}"

    prompt_fn = _PROMPT_FNS.get(subject, math_prompt_fn)

    task_metrics: List[Any] = [physics_mc_metric] if subject == "physics_part1" else [hungarian_math_metric]

    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=task_name,
            prompt_function=prompt_fn,
            hf_repo="json",
            hf_subset="default",
            hf_data_files={"validation": all_files_for_subset},
            hf_avail_splits=["validation"],
            evaluation_splits=["validation"],
            metrics=task_metrics,
        )
    )
