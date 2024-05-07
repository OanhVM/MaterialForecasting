from argparse import ArgumentParser
from enum import Enum
from typing import List

import numpy as np

from common import read_cont_seqs_csv, update_global_stats_results_csv_file_name
from models.arima import evaluate_arima
from models.naive import evaluate_naive


class StatsModel(Enum):
    NAIVE = (evaluate_naive, 1)
    ARIMA_3 = (lambda *args, **kwargs: evaluate_arima(*args, **kwargs, lag=3), 3)
    ARIMA_6 = (lambda *args, **kwargs: evaluate_arima(*args, **kwargs, lag=6), 6)

    def __init__(self, eval_func: callable, min_offset: int):
        self.eval_func: callable = eval_func
        self.min_offset: int = min_offset

    def __call__(self, *args, **kwargs):
        return self.eval_func(*args, **kwargs)


def _eval_global(model_name: str,
                 company_name: str, col_name: str,
                 min_cont_length: int,
                 do_diff: bool,
                 horizons: List[int],
                 data_dir_path: str = "data",
                 ):
    stats_model: StatsModel = StatsModel[model_name.upper()]

    min_cont_length = max(
        min_cont_length,
        max(horizons) + stats_model.min_offset + (1 if do_diff else 0),
    )

    cont_seqs = read_cont_seqs_csv(
        company_name=company_name,
        min_cont_length=min_cont_length,
        col_name=col_name,
        data_dir_path=data_dir_path,
    )
    cont_seqs = [np.diff(s) for s in cont_seqs] if do_diff else cont_seqs

    rmses = stats_model(cont_seqs=cont_seqs, horizons=horizons)

    update_global_stats_results_csv_file_name(
        model_name=model_name,
        company_name=company_name, col_name=col_name,
        min_cont_length=min_cont_length,
        do_diff=do_diff,
        horizons=horizons, rmses=rmses,
    )


def _main():
    arg_parser = ArgumentParser()

    # TODO: add more `model_name`
    arg_parser.add_argument("model_name", type=str, choices=[m.name.lower() for m in StatsModel])
    arg_parser.add_argument("company_names", type=str, nargs="+")
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--min-cont-length", "-m", metavar="", type=int, default=2)
    arg_parser.add_argument("--do-diff", "-d", action="store_true")
    arg_parser.add_argument("--horizons", "-H", metavar="", type=int, nargs="+")

    args = arg_parser.parse_args()

    for company_name in args.company_names:
        _eval_global(
            model_name=args.model_name,
            company_name=company_name,
            col_name=args.col_name,
            min_cont_length=args.min_cont_length,
            do_diff=args.do_diff,
            horizons=args.horizons,
        )


if __name__ == "__main__":
    _main()
