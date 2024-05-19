import logging
import warnings
from typing import List

import numpy as np
from numpy import ndarray
from numpy.linalg import LinAlgError
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)


def arima_forecast(inputs: List[ndarray], label_width: int, lag: int, diff: int, **kwargs) -> List[ndarray]:
    preds = []
    for idx, _input in enumerate(inputs):

        if idx + 1 % 100 == 0 or idx == len(inputs) - 1:
            print(f"evaluate_arima: lag = {lag}; diff = {diff}; idx = {idx + 1:4d}/{len(inputs)}")

        try:
            preds.append(
                ARIMA(_input, order=(lag, diff, 0)).fit().forecast(steps=label_width)
            )
        except LinAlgError as e:
            logging.error(f"{e}: len(_input) = {len(_input):3d}; _input = {_input}")
            # Default to naive on failure
            preds.append(
                np.repeat(_input[-1], label_width)
            )

    return preds
