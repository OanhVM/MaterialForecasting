from argparse import ArgumentParser
from enum import Enum
from functools import partial
from os.path import isfile
from typing import List, Tuple

from numpy import ndarray

from common import read_cont_seqs_csv, save_forecast_data, get_forecast_data_file_path, read_forecast_data, \
    get_model_file_path
from models.arima import arima_forecast
from models.lstm import train_and_forecast_lstm
from models.lstm_fed import forecast_lstm_fed
from models.naive import naive_forecast


class ForecastModel(Enum):
    NAIVE = (naive_forecast, 1)

    ARMA3 = (partial(arima_forecast, lag=3, diff=0), 3)
    ARMA6 = (partial(arima_forecast, lag=6, diff=0), 6)

    ARIMA3 = (partial(arima_forecast, lag=3, diff=1), 3)
    ARIMA6 = (partial(arima_forecast, lag=6, diff=1), 6)

    LSTM8 = (partial(train_and_forecast_lstm, n_neuron=8, do_balance=False), 1)
    LSTM32 = (partial(train_and_forecast_lstm, n_neuron=32, do_balance=False), 1)

    LSTM8B = (partial(train_and_forecast_lstm, n_neuron=8, do_balance=True), 1)
    LSTM32B = (partial(train_and_forecast_lstm, n_neuron=32, do_balance=True), 1)

    # TODO: this is a bit ugly/duplicated? (partial() is cheating)
    LSTM8F = (partial(forecast_lstm_fed), 1)
    LSTM32F = (partial(forecast_lstm_fed), 1)

    LSTM8BF = (partial(forecast_lstm_fed), 1)
    LSTM32BF = (partial(forecast_lstm_fed), 1)

    def __init__(self, forecast_func: callable, min_input_width: int):
        self._forecast_func: callable = forecast_func
        self._min_input_width: int = min_input_width

    @property
    def min_input_width(self):
        return self._min_input_width

    def __call__(self, *args, **kwargs) -> List[ndarray]:
        return self._forecast_func(*args, **kwargs)


def _make_inputs_and_labels(cont_seqs: List[ndarray], label_width: int) -> Tuple[List[ndarray], List[ndarray]]:
    inputs, labels = [], []
    for cont_seq in cont_seqs:
        inputs.append(cont_seq[:-label_width])
        labels.append(cont_seq[-label_width:])

    return inputs, labels


def _get_inputs_and_labels(
        company_name: str, col_name: str, min_cont_length: int, label_width: int,
        data_dir_path: str = "data",
) -> Tuple[List[ndarray], List[ndarray]]:
    inputs_file_path, labels_file_path = [
        get_forecast_data_file_path(
            forecast_data_type=forecast_data_type,
            company_name=company_name, col_name=col_name,
            min_cont_length=min_cont_length, label_width=label_width,
            data_dir_path=data_dir_path,
        )
        for forecast_data_type in ("inputs", "labels")
    ]

    if isfile(inputs_file_path) and isfile(labels_file_path):
        inputs, labels = [
            read_forecast_data(forecast_data_file_path)
            for forecast_data_file_path in (inputs_file_path, labels_file_path)
        ]

    else:
        cont_seqs = read_cont_seqs_csv(
            company_name=company_name, col_name=col_name,
            min_cont_length=min_cont_length,
            data_dir_path=data_dir_path,
        )

        inputs, labels = _make_inputs_and_labels(cont_seqs=cont_seqs, label_width=label_width)

        for forecast_data, forecast_data_file_path in [
            (inputs, inputs_file_path),
            (labels, labels_file_path),
        ]:
            save_forecast_data(forecast_data=forecast_data, forecast_data_file_path=forecast_data_file_path)

    return inputs, labels


def _forecast(
        model_names: List[str],
        company_name: str, col_name: str,
        min_cont_length: int, horizons: List[int],
        data_dir_path: str = "data",
):
    if model_names == ["all"]:
        forecast_models: List[ForecastModel] = list(ForecastModel)
    else:
        forecast_models: List[ForecastModel] = [ForecastModel[model_name.upper()] for model_name in model_names]

    label_width = max(horizons)
    min_input_width = max(eval_model.min_input_width for eval_model in forecast_models)
    actual_min_cont_length = max(min_cont_length, min_input_width + label_width)
    print(
        f"min_input_width = {min_input_width}; label_width = {label_width}; "
        f"actual_min_cont_length = {actual_min_cont_length}"
    )

    inputs, labels = _get_inputs_and_labels(
        company_name=company_name, col_name=col_name,
        min_cont_length=actual_min_cont_length, label_width=label_width,
        data_dir_path=data_dir_path,
    )

    for forecast_model in forecast_models:
        preds_file_path = get_forecast_data_file_path(
            forecast_data_type=f"preds_{forecast_model.name.lower()}",
            company_name=company_name, col_name=col_name,
            min_cont_length=min_cont_length, label_width=label_width,
            data_dir_path=data_dir_path,
        )

        if not isfile(preds_file_path):
            preds = forecast_model(
                inputs=inputs, label_width=label_width, labels=labels,
                # ML models -specific parameters
                model_file_path=get_model_file_path(
                    model_name=forecast_model.name.lower(),
                    company_name=company_name, col_name=col_name,
                    min_cont_length=min_cont_length, label_width=label_width,
                ),
            )
            save_forecast_data(forecast_data=preds, forecast_data_file_path=preds_file_path)

        else:
            print(f"{preds_file_path} already exists - model forecasting skipped.")


def _main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("company_names", type=str, nargs="+")
    arg_parser.add_argument(
        "--model-names", "-M", type=str, nargs="+",
        choices=[*[m.name.lower() for m in ForecastModel], "all"], default=["all"],
    )
    arg_parser.add_argument("--horizons", "-H", metavar="", type=int, nargs="+", default=[1, 3, 6])
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--min-cont-length", "-l", metavar="", type=int, default=13)

    args = arg_parser.parse_args()

    for company_name in args.company_names:
        _forecast(
            model_names=args.model_names,
            company_name=company_name,
            col_name=args.col_name,
            min_cont_length=args.min_cont_length,
            horizons=args.horizons,
        )


if __name__ == "__main__":
    _main()
