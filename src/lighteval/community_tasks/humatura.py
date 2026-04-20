import logging
from pathlib import Path
from typing import Any, Dict, List

from lighteval.tasks.requests import Doc

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
