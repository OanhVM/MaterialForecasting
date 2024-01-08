from argparse import ArgumentParser
from typing import List

import numpy as np
from numpy import ndarray

from common import read_cont_seqs_csv
from models.arima import evaluate_arima
from models.naive import evaluate_naive

EVAL_FUNC_PER_MODEL_NAME = {
    "naive": evaluate_naive,
    "arima_6": lambda *args, **kwargs: evaluate_arima(*args, **kwargs, lag=6),
}


def eval_model(model_name: str, cont_seqs: List[ndarray]):
    return EVAL_FUNC_PER_MODEL_NAME[model_name](cont_seqs=cont_seqs)


def _eval_global(model_name: str,
                 company_name: str, col_name: str,
                 min_cont_length: int,
                 do_diff: bool,
                 data_dir_path: str = "data",
                 ):
    cont_seqs = read_cont_seqs_csv(
        company_name=company_name, min_cont_length=min_cont_length, col_name=col_name,
        data_dir_path=data_dir_path,
    )

    cont_seqs = [np.diff(seq) for seq in cont_seqs] if do_diff else cont_seqs

    rmse = eval_model(model_name=model_name, cont_seqs=cont_seqs)
    print(
        f"model_name = {model_name:12}; company = {company_name}; rmse = {rmse:10.6f}"
    )


def _main():
    arg_parser = ArgumentParser()

    # TODO: add more `model_name`
    arg_parser.add_argument("model_name", type=str, choices=EVAL_FUNC_PER_MODEL_NAME.keys())
    arg_parser.add_argument("company_names", type=str, nargs="+")
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--min-cont-length", metavar="", type=int, default=2)
    arg_parser.add_argument("--do-diff", action="store_true")

    args = arg_parser.parse_args()

    for company_name in args.company_names:
        _eval_global(
            model_name=args.model_name,
            company_name=company_name,
            col_name=args.col_name,
            min_cont_length=args.min_cont_length,
            do_diff=args.do_diff,
        )


if __name__ == "__main__":
    _main()
