from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PartialWorld:
    arrays: dict[str, np.ndarray]
    world_id: int
