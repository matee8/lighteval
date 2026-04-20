import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

logger = logging.getLogger(__name__)


def discover_local_datasets(data_dir: Path) -> Dict[str, List[str]]:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist or is not a directory: {data_dir}")

    data_files: Dict[str, List[str]] = {"emelt": [], "kozep": []}

    for file_path in data_dir.glob("*.json"):
        file_name = file_path.stem.lower()
        if "emelt" in file_name:
            data_files["emelt"].append(str(file_path.absolute()))
        elif "kozep" in file_name:
            data_files["kozep"].append(str(file_path.absolute()))
        else:
            logger.warning("File %s does not match emelt or kozep subsets. Skipping.", file_path)

    # Since I create the datasets earlier, it could be left empty, which is not compatible with
    # LightEval.
    data_files = {subset: paths for subset, paths in data_files.items() if paths}

    if not data_files:
        logger.warning("No emelt or kozep JSON dataset files found in %s.", data_dir)

    return data_files


def hungarian_math_prompt_fn(line: Dict[str, Any], task_name: str = "") -> Doc:
    instruction = (
        "Kérlek, oldd meg a következő matematikai feladatot. "
        "Gondoljuk végig lépésről lépésre, részletesen. "
        "A végső választ pontosan egy \\boxed{} formátumba írd be "
        "(például \\boxed{42} vagy \\boxed{x=3})."
    )

    description: str = line["description"]
    query = f"{instruction}\n\nFeladat:\n{description}\n\nMegoldás:\n"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[],
        gold_index=[],
        specific={"solution": line["solution"]["solution"]},
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


def aggregate_scores(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


hungarian_math_metric: SampleLevelMetric = SampleLevelMetric(
    metric_name="hungarian_math_final_answer",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=HungarianMathEquivalence(),
    corpus_level_fn=aggregate_scores,
)

TASKS_TABLE: Dict[str, LightevalTaskConfig] = {}

_DATA_DIR = Path(os.environ.get("HUMATURA_DATA_DIR", "./data"))

try:
    _discovered_files = discover_local_datasets(_DATA_DIR)
except NotADirectoryError:
    logger.warning("Data directory %s not found. Tasks will be registered without local files.", _DATA_DIR)
    _discovered_files = {"emelt": [], "kozep": []}

for subset in ["emelt", "kozep"]:
    task_name: str = f"hungarian_math:{subset}"

    TASKS_TABLE[task_name] = LightevalTaskConfig(
        name=task_name,
        prompt_function=hungarian_math_prompt_fn,
        hf_repo="json",
        hf_subset=subset,
        metrics=[hungarian_math_metric],
        hf_data_files=_discovered_files if _discovered_files else None,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
    )
