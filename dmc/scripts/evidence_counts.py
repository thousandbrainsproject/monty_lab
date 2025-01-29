import json
import os
from pathlib import Path

import numpy as np
from data_utils import (
    DMC_RESULTS_DIR,
)


def describe_dict(data, level=0):
    """
    Recursively describe the contents of a nested dictionary.

    :param data: The dictionary to describe.
    :param level: Current depth level in the nested dictionary.
    """
    if not isinstance(data, dict):
        print(f"{'  ' * level}- Not a dictionary: {type(data).__name__}")
        return

    for key in sorted(data.keys()):
        value = data[key]
        print(f"{'  ' * (level + 1)}'{key}': {type(value).__name__}")
        if isinstance(value, dict):
            # Recursively describe nested dictionaries
            describe_dict(value, level + 1)


class DetailedJSONStats:
    def __init__(self, path: os.PathLike):
        self._path = Path(path)
        self._index = None
        self._initialized = False

    @property
    def path(self) -> os.PathLike:
        return self._path

    def _check_initialized(self):
        if self._initialized:
            return
        length = 0
        with open(self._path, "r") as f:
            length = sum(1 for _ in f)
        self._index = np.arange(length)
        self._initialized = True

    def _get_line(self, num: int):
        with open(self._path, "r") as f:
            for i, line in enumerate(f):
                if i == num:
                    return json.loads(line)[str(i)]

    def __iter__(self):
        with open(self._path, "r") as f:
            for i, line in enumerate(f):
                yield json.loads(line)[str(i)]

    def __len__(self) -> int:
        self._check_initialized()
        return len(self._index)

    def __getitem__(self, episode: int):
        self._check_initialized()
        assert np.isscalar(episode)
        episode = self._index[episode]
        ln = self._get_line(episode)
        return ln


experiment_dir = DMC_RESULTS_DIR / "dist_agent_1lm_randrot_noise_10simobj"
stats = DetailedJSONStats(experiment_dir / "detailed_run_stats.json")
