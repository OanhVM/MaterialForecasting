import logging
import warnings
from typing import List

from numpy import ndarray
from numpy.linalg import LinAlgError
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)


def evaluate_arima(inputs: List[ndarray], label_width: int, lag: int, diff: int) -> List[ndarray]:
    preds = []
    for idx, _input in enumerate(inputs):

        if idx % 100 == 0 or idx == len(inputs) - 1:
            print(f"evaluate_arima: lag = {lag}; diff = {diff}; idx = {idx:4d}/{len(inputs)}")

        try:
            preds.append(
                ARIMA(_input, order=(lag, diff, 0)).fit().forecast(steps=label_width)
            )
        except LinAlgError as e:
            logging.error(f"{e}: len(_input) = {len(_input):3d}; _input = {_input}")

    return preds
