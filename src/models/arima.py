import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

warnings.simplefilter('ignore', ConvergenceWarning)


def evaluate_arima(cont_seqs: List[ndarray], lag: int, max_workers: int = 4) -> Tuple[float, float]:
    # TODO: is here the place for cont_seqs filtering?
    cont_seqs = [s for s in cont_seqs if len(s) > lag]

    def _pred_arima(cont_seq: ndarray) -> Tuple[float, float]:
        return ARIMA(cont_seq[:-1], order=(lag, 0, 0)).fit().forecast(steps=1)[0]

    preds = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_pred_arima, cont_seq=cont_seq)
            for cont_seq in cont_seqs
        ]

        for future in as_completed(futures):
            try:
                preds.append(future.result())
            except Exception as e:
                logging.exception(e)

    preds = np.asarray(preds)
    labels = np.asarray([c[-1] for c in cont_seqs])

    rmse = np.mean(labels ** 2 - preds ** 2) ** 0.5

    return rmse
