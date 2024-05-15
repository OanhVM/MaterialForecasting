import logging
import warnings
from typing import List

import numpy as np
from numpy import ndarray
from numpy.linalg import LinAlgError
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

warnings.simplefilter('ignore', ConvergenceWarning)


def evaluate_arima(cont_seqs: List[ndarray], horizons: List[int], lag: int, diff: int) -> List[float]:
    max_horizon = max(horizons)
    horizons = np.asarray(horizons)

    preds = []
    labels = []
    for idx, cont_seq in enumerate(cont_seqs):
        print(f"evaluate_arima: idx = {idx:4d}/{len(cont_seqs)}")
        try:
            arima_result = ARIMA(cont_seq[:-max_horizon], order=(lag, diff, 0)).fit()
            preds.append(
                arima_result.forecast(steps=max_horizon)[horizons - 1]
            )
            labels.append(
                cont_seq[-max_horizon:][horizons - 1]
            )
        except LinAlgError as e:
            logging.error(
                f"len(cont_seq) = {len(cont_seq):3d}; cont_seq =\n{cont_seq}\n"
                f"err = {e}"
            )
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    squared_errs = (labels - preds) ** 2
    rmses = [
        np.mean(squared_errs[:, :horizon]) ** 0.5
        for horizon in horizons
    ]
    return rmses
