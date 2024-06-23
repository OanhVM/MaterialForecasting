from os.path import basename, isdir
from typing import List, Tuple

import numpy as np
from keras.models import clone_model
from numpy import ndarray

from common import save_model, read_model, parse_model_file_name, get_model_file_path
from models.lstm import model_forecast

_ALL_COMPANY_NAMES = ["A", "B", "C"]


def _get_relevant_model_file_paths(model_file_path: str) -> Tuple[List[str], str]:
    fed_model_name, _, col_name, min_cont_length, label_width = parse_model_file_name(
        model_file_name=basename(model_file_path),
    )

    model_file_paths = [
        get_model_file_path(
            model_name=fed_model_name[1:],  # e.g. FLSTM32_F -> LSTM32
            company_name=company_name, col_name=col_name,
            min_cont_length=min_cont_length, label_width=label_width,
        )
        for company_name in _ALL_COMPANY_NAMES
    ]

    fed_model_file_path = get_model_file_path(
        model_name=fed_model_name,
        company_name="".join(_ALL_COMPANY_NAMES), col_name=col_name,
        min_cont_length=min_cont_length, label_width=label_width,
    )

    return model_file_paths, fed_model_file_path


def forecast_lstm_fed(inputs: List[ndarray], model_file_path: str, **kwargs) -> List[ndarray]:
    model_file_paths, fed_model_file_path = _get_relevant_model_file_paths(model_file_path=model_file_path)

    if not isdir(fed_model_file_path):
        models = [read_model(model_file_path=model_file_path) for model_file_path in model_file_paths]

        fed_model = clone_model(models[0])
        fed_model.set_weights(
            np.mean([np.asarray(model.get_weights(), dtype=object) for model in models], axis=0)
        )

        save_model(model=fed_model, model_file_path=fed_model_file_path)

    else:
        fed_model = read_model(model_file_path=fed_model_file_path)

    preds = model_forecast(model=fed_model, inputs=inputs)

    return preds
