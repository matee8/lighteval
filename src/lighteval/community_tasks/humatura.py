import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def discover_local_datasets(data_dir: Path) -> Dict[str, str]:
    if not data_dir.is_dir():
        raise NotADirectoryError('Dataset directory does not exist or is not a directory: '
                                 f'{data_dir}')

    data_files = {}

    for file_path in data_dir.glob('*.json'):
        subset_name = file_path.stem.replace('-matematika', '')
        data_files[subset_name] = str(file_path.absolute())

    if not data_files:
        logger.warning('No JSON dataset files found in %s.', data_dir)

    return data_files
