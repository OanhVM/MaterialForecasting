from typing import List

import numpy as np
from numpy import ndarray


def evaluate_naive(inputs: List[ndarray], label_width: int) -> List[ndarray]:
    return [np.repeat(_input[-1], label_width) for _input in inputs]
