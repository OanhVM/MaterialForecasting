from typing import List

import numpy as np
from numpy import ndarray


def evaluate_naive(cont_seqs: List[ndarray], horizons: List[int]) -> List[float]:
    max_horizon = max(horizons)
    horizons = np.asarray(horizons)

    preds = np.asarray([
        cont_seq[:-1][-max_horizon:] for cont_seq in cont_seqs
    ])
    labels = np.asarray([
        cont_seq[-max_horizon:] for cont_seq in cont_seqs
    ])

    squared_errs = (labels - preds) ** 2
    rmses = [
        np.mean(squared_errs[:, :horizon]) ** 0.5
        for horizon in horizons
    ]
    return rmses
