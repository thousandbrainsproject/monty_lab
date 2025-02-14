import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
from data_utils import (
    DMC_RESULTS_DIR,
)


def describe_dict(data: Mapping, level: int = 0):
    """
    Recursively describe the contents of a nested dictionary. For visualizing the
    structure of detailed JSON stats. Can be removed when no longer useful.

    Args:
        data (dict): The dictionary to describe.
        level (int): Current depth level in the nested dictionary.
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


class DetailedJSONStatsInterface:
    """Convenience interface to detailed JSON stats.

    This class is a dict-like interface to detailed JSON stats files that loads
    episodes one at a time. An episode can be loaded via `stats[episode_num]`
    (or, equivalently `stats.read_episode(episode_num)`), which takes about
    1.5 - 6.5 seconds per episode. If you plan on loading all episodes eventually,
    the most efficient method is to iterate over a `DetailedJSONStatsInterface`.

    Example:
        >>> stats = DetailedJSONStatsInterface("detailed_stats.json")
        >>> last_episode_data = stats[-1]  # Get data for the last episode.
        >>> # Iterate over all episodes.
        >>> for i, episode_data in enumerate(stats):
        ...     # Process episode data
        ...     pass
    """

    def __init__(self, path: os.PathLike):
        self._path = Path(path)
        self._index = None  # Just used to convert possibly negative indices

    @property
    def path(self) -> os.PathLike:
        return self._path

    def read_episode(self, episode: int) -> Mapping:
        self._check_initialized()
        assert np.isscalar(episode)
        episode = self._index[episode]
        with open(self._path, "r") as f:
            for i, line in enumerate(f):
                if i == episode:
                    return json.loads(line)[str(i)]

    def _check_initialized(self):
        if self._index is not None:
            return
        length = 0
        with open(self._path, "r") as f:
            length = sum(1 for _ in f)
        self._index = np.arange(length)

    def __iter__(self):
        with open(self._path, "r") as f:
            for i, line in enumerate(f):
                yield json.loads(line)[str(i)]

    def __len__(self) -> int:
        self._check_initialized()
        return len(self._index)

    def __getitem__(self, episode: int) -> Mapping:
        """Get the stats for a given episode.

        Args:
            episode (int): The episode number.

        Returns:
            Mapping: The stats for the episode.
        """
        return self.read_episode(episode)



experiment_dir = DMC_RESULTS_DIR / "dist_agent_1lm_randrot_noise_10simobj"
stats = DetailedJSONStatsInterface(experiment_dir / "detailed_run_stats.json")
ep = stats[0]
