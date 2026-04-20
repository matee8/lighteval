import logging
from pathlib import Path
from typing import Any, Dict

from lighteval.tasks.requests import Doc

logger = logging.getLogger(__name__)


def discover_local_datasets(data_dir: Path) -> Dict[str, str]:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist or is not a directory: {data_dir}")

    data_files: Dict[str, str] = {}

    for file_path in data_dir.glob("*.json"):
        subset_name = file_path.stem.replace("-matematika", "")
        data_files[subset_name] = str(file_path.absolute())

    if not data_files:
        logger.warning("No JSON dataset files found in %s.", data_dir)

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
