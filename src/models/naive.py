from typing import List, Tuple

import numpy as np
from numpy import ndarray


def evaluate_naive(cont_seqs: List[ndarray]) -> Tuple[float, float]:
    preds = np.asarray([c[-2] for c in cont_seqs])
    labels = np.asarray([c[-1] for c in cont_seqs])

    rmse = np.mean(labels ** 2 - preds ** 2) ** 0.5

    return rmse
