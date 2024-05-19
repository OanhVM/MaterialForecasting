from argparse import ArgumentParser
from enum import Enum
from functools import partial
from os.path import isfile
from typing import List, Tuple

from numpy import ndarray

from common import read_cont_seqs_csv, save_eval_results, get_eval_results_file_path, read_eval_results
from models.arima import evaluate_arima
from models.lstm import train_and_eval_lstm
from models.naive import evaluate_naive


def _make_arima_eval_func(lag: int, diff: int):
    def _arima_eval_func(inputs: List[ndarray], _, label_width: int):
        return evaluate_arima(inputs=inputs, label_width=label_width, lag=lag, diff=diff)

    return _arima_eval_func


class EvalModel(Enum):
    NAIVE = (lambda inputs, _, label_width: evaluate_naive(inputs=inputs, label_width=label_width), 1)

    ARMA3 = (_make_arima_eval_func(lag=3, diff=0), 3)
    ARMA6 = (_make_arima_eval_func(lag=6, diff=0), 6)

    ARIMA3 = (_make_arima_eval_func(lag=6, diff=1), 3)
    ARIMA6 = (_make_arima_eval_func(lag=6, diff=1), 6)

    LSTM4 = (partial(train_and_eval_lstm, n_neuron=4, do_balance=False), 1)
    LSTM8 = (partial(train_and_eval_lstm, n_neuron=8, do_balance=False), 1)
    LSTM16 = (partial(train_and_eval_lstm, n_neuron=16, do_balance=False), 1)
    LSTM32 = (partial(train_and_eval_lstm, n_neuron=32, do_balance=False), 1)

    LSTM4B = (partial(train_and_eval_lstm, n_neuron=4, do_balance=True), 1)
    LSTM8B = (partial(train_and_eval_lstm, n_neuron=8, do_balance=True), 1)
    LSTM16B = (partial(train_and_eval_lstm, n_neuron=16, do_balance=True), 1)
    LSTM32B = (partial(train_and_eval_lstm, n_neuron=32, do_balance=True), 1)

    ALL = (None, None)

    def __init__(self, eval_func: callable, min_input_width: int):
        self._eval_func: callable = eval_func
        self._min_input_width: int = min_input_width

    @property
    def min_input_width(self):
        return self._min_input_width

    def __call__(self, inputs: List[ndarray], labels: List[ndarray], label_width: int):
        return self._eval_func(inputs, labels, label_width)


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
        get_eval_results_file_path(
            result_type=result_type,
            company_name=company_name, col_name=col_name,
            min_cont_length=min_cont_length, label_width=label_width,
            data_dir_path=data_dir_path,
        )
        for result_type in ("inputs", "labels")
    ]

    if isfile(inputs_file_path) and isfile(labels_file_path):
        inputs, labels = [
            read_eval_results(eval_results_file_path)
            for eval_results_file_path in (inputs_file_path, labels_file_path)
        ]

    else:
        cont_seqs = read_cont_seqs_csv(
            company_name=company_name, col_name=col_name,
            min_cont_length=min_cont_length,
            data_dir_path=data_dir_path,
        )

        inputs, labels = _make_inputs_and_labels(cont_seqs=cont_seqs, label_width=label_width)

        for eval_results, eval_results_file_path in [
            (inputs, inputs_file_path),
            (labels, labels_file_path),
        ]:
            save_eval_results(eval_results=eval_results, eval_results_file_path=eval_results_file_path)

    return inputs, labels


def _eval_global(model_names: List[str],
                 company_name: str, col_name: str,
                 min_cont_length: int,
                 horizons: List[int],
                 data_dir_path: str = "data",
                 ):
    # TODO: run this only if no cached results exists
    eval_models: List[EvalModel] = [EvalModel[model_name.upper()] for model_name in model_names]

    label_width = max(horizons)
    min_input_width = max(eval_model.min_input_width for eval_model in eval_models)
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

    for eval_model in eval_models:
        preds_file_path = get_eval_results_file_path(
            result_type=f"preds_{eval_model.name.lower()}",
            company_name=company_name, col_name=col_name,
            min_cont_length=min_cont_length, label_width=label_width,
            data_dir_path=data_dir_path,
        )

        if not isfile(preds_file_path):
            preds = eval_model(inputs=inputs, labels=labels, label_width=label_width)
            save_eval_results(eval_results=preds, eval_results_file_path=preds_file_path)

        else:
            print(f"{preds_file_path} already exists - model forecasting skipped.")


def _main():
    arg_parser = ArgumentParser()

    # TODO: add more `model_name`
    arg_parser.add_argument("company_names", type=str, nargs="+")
    arg_parser.add_argument(
        "--model-names", "-m", type=str, nargs="+", required=True,
        choices=[m.name.lower() for m in EvalModel],
    )
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--min-cont-length", "-l", metavar="", type=int, default=2)
    arg_parser.add_argument("--horizons", "-H", metavar="", type=int, nargs="+")

    args = arg_parser.parse_args()

    for company_name in args.company_names:
        _eval_global(
            model_names=args.model_names,
            company_name=company_name,
            col_name=args.col_name,
            min_cont_length=args.min_cont_length,
            horizons=args.horizons,
        )


if __name__ == "__main__":
    _main()
