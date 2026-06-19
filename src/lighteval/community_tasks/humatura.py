import dataclasses
import json
import logging
import os
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


def _parse_solution_steps(steps_str: str) -> List[Dict[str, Any]]:
    if not steps_str:
        return []
    try:
        return json.loads(steps_str)
    except json.JSONDecodeError:
        logger.error("Failed to parse solution steps: %s.", steps_str)
        return []


def math_prompt_fn(line: Dict[str, Any], task_name: str = "") -> Doc:
    instruction = (
        "Kérlek, oldd meg a következő matematikai feladatot. "
        "Gondoljuk végig lépésről lépésre, részletesen. "
        "A végső választ pontosan egy \\boxed{} formátumba írd be "
        "(például \\boxed{42} vagy \\boxed{x=3})."
    )

    description: str = line.get("description", "")
    query = f"{instruction}\n\nFeladat:\n{description}\n\nMegoldás:\n"

    solutions = [{
        "final-answer": line.get("final_answer", ""),
        "steps": _parse_solution_steps(line.get("solution_steps", "")),
        "points": line.get("point", 1.0)
    }]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[],
        gold_index=[],
        specific={"solutions": solutions},
    )


def physics_multiple_choice_prompt_fn(line: Dict[str, Any], task_name: str = "") -> Doc:
    instruction = (
        "Kérlek, válaszolj a következő feleletválasztós fizika feladatra. "
        "A végső válaszod betűjelét (A, B, C vagy D) pontosan egy \\boxed{} formátumba írd be "
        "(például \\boxed{A} vagy \\boxed{C})."
    )

    description: str = line.get("description", "")
    query = f"{instruction}\n\nFeladat:\n{description}\n\nMegoldás:\n"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[],
        gold_index=[],
        specific={"correct_answer": line.get("final_answer", "")},
    )


def physics_open_ended_prompt_fn(line: Dict[str, Any], task_name: str = "") -> Doc:
    instruction = (
        "Kérlek, oldd meg a következő fizika feladatot. "
        "Gondoljuk végig lépésről lépésre, részletesen, és add meg a végső választ."
    )

    description: str = line.get("description", "")
    query = f"{instruction}\n\nFeladat:\n{description}\n\nMegoldás:\n"

    solutions = [{
        "final-answer": line.get("final_answer", ""),
        "steps": _parse_solution_steps(line.get("solution_steps", "")),
        "points": line.get("point", 1.0)
    }]

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
    "physics-multiple-choice": SubjectTemplate(prompt_fn=physics_multiple_choice_prompt_fn, metrics=[physics_mc_metric]),
    "physics-open-ended": SubjectTemplate(prompt_fn=physics_open_ended_prompt_fn, metrics=[llm_judge_metric]),
}

TASKS_TABLE: List[LightevalTaskConfig] = []



for subset, template in _SUBJECT_TEMPLATES.items():
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"humatura:{subset}",
            prompt_function=template.prompt_fn,
            hf_repo="NYTK/HuMatura",
            hf_subset=subset,
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            metrics=template.metrics,
        )
    )
