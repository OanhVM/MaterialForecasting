from argparse import ArgumentParser
from enum import Enum
from functools import partial
from typing import List

import numpy as np
from numpy import ndarray
from sktime.performance_metrics.forecasting import mean_squared_error, mean_absolute_error, median_absolute_error, \
    mean_absolute_scaled_error, median_absolute_scaled_error

from common import get_forecast_data_file_path, read_forecast_data, update_global_results_csv_file_name
from models.arima import arima_forecast
from run_forecast import ForecastModel


def _make_arima_forecast_func(lag: int, diff: int):
    def _arima_forecast_func(inputs: List[ndarray], _, label_width: int, __):
        return arima_forecast(inputs=inputs, label_width=label_width, lag=lag, diff=diff)

    return _arima_forecast_func


class EvalMetrics(partial, Enum):
    MAE = partial(mean_absolute_error, multioutput="raw_values")
    MdAE = partial(median_absolute_error, multioutput="raw_values")

    RMSE = partial(mean_squared_error, multioutput="raw_values", square_root=True)

    MASE = partial(mean_absolute_scaled_error, multioutput="raw_values")
    MdASE = partial(median_absolute_scaled_error, multioutput="raw_values")


def _evaluate(
        metric_names: List[str],
        model_names: List[str],
        company_name: str, col_name: str,
        min_cont_length: int, horizons: List[int],
        data_dir_path: str = "data",
):
    if metric_names == ["all"]:
        eval_metrics: List[EvalMetrics] = list(EvalMetrics)
    else:
        eval_metrics: List[EvalMetrics] = [EvalMetrics[metric_name] for metric_name in metric_names]

    if model_names == ["all"]:
        forecast_models: List[ForecastModel] = list(ForecastModel)
    else:
        forecast_models: List[ForecastModel] = [ForecastModel[model_name.upper()] for model_name in model_names]

    label_width = max(horizons)

    labels = read_forecast_data(
        forecast_data_file_path=get_forecast_data_file_path(
            forecast_data_type="labels",
            company_name=company_name, col_name=col_name,
            min_cont_length=min_cont_length, label_width=label_width,
            data_dir_path=data_dir_path,
        ),
    )
    labels = np.asarray(labels)

    for forecast_model in forecast_models:
        preds = read_forecast_data(
            forecast_data_file_path=get_forecast_data_file_path(
                forecast_data_type=f"preds_{forecast_model.name.lower()}",
                company_name=company_name, col_name=col_name,
                min_cont_length=min_cont_length, label_width=label_width,
                data_dir_path=data_dir_path,
            ),
        )
        preds = np.asarray(preds)

        metric_results_per_metric_name = {
            eval_metric.name: eval_metric(labels, preds, y_train=labels)
            for eval_metric in eval_metrics
        }
        metric_result_per_metric_full_name = {
            f"{metric_name}_{horizon}": metric_results[horizon - 1]
            for metric_name, metric_results in metric_results_per_metric_name.items()
            for horizon in horizons
        }

        update_global_results_csv_file_name(
            model_name=forecast_model.name.lower(),
            company_name=company_name, col_name=col_name,
            min_cont_length=min_cont_length,
            metric_result_per_metric_full_name=metric_result_per_metric_full_name,
        )


def _main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("company_names", type=str, nargs="+")
    arg_parser.add_argument(
        "--metric-names", "-m", type=str, nargs="+", required=True,
        choices=[*[m.name.lower() for m in EvalMetrics], "all"],
    )
    arg_parser.add_argument(
        "--model-names", "-M", type=str, nargs="+", required=True,
        choices=[*[m.name.lower() for m in ForecastModel], "all"],
    )
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--min-cont-length", "-l", metavar="", type=int, default=2)
    arg_parser.add_argument("--horizons", "-H", metavar="", type=int, nargs="+")

    args = arg_parser.parse_args()

    for company_name in args.company_names:
        _evaluate(
            metric_names=args.metric_names,
            model_names=args.model_names,
            company_name=company_name,
            col_name=args.col_name,
            min_cont_length=args.min_cont_length,
            horizons=args.horizons,
        )


if __name__ == "__main__":
    _main()
