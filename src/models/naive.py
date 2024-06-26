from typing import List

import numpy as np
from numpy import ndarray


def naive_forecast(inputs: List[ndarray], label_width: int, **kwargs) -> List[ndarray]:
    return [np.repeat(_input[-1], label_width) for _input in inputs]
