import logging
from argparse import ArgumentParser
from colorsys import rgb_to_hls, hls_to_rgb
from os.path import join
from typing import List, Union, Tuple

import numpy as np
from matplotlib import pyplot as plt, colormaps
from numpy import ndarray

from common import read_global_results_csv_file_name
from run_eval import EvalMetrics
from run_forecast import ForecastModel

def adjust_lightness(color: Union[Tuple[int, int, int], ndarray], amount: float = 0.5):
    c = rgb_to_hls(*color)
    return hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


_COLOR_PER_MODEL_NAME = {
    _m.name.lower(): adjust_lightness(colormaps["tab20c"].colors[_ci], amount=0.9) for _m, _ci in [
        (ForecastModel.NAIVE, 16),

        (ForecastModel.ARMA3, 4),
        (ForecastModel.ARMA6, 5),
        (ForecastModel.ARIMA3, 6),
        (ForecastModel.ARIMA6, 7),

        (ForecastModel.LSTM8, 0),
        (ForecastModel.LSTM32, 1),
        (ForecastModel.LSTM8B, 2),
        (ForecastModel.LSTM32B, 3),

        (ForecastModel.LSTM8F, 8),
        (ForecastModel.LSTM32F, 9),
        (ForecastModel.LSTM8BF, 10),
        (ForecastModel.LSTM32BF, 11),
    ]
}

_YLIM_PER_METRIC_NAME = {
    EvalMetrics.MAE.name: 0.69,
    EvalMetrics.MdAE.name: 0.69,
    EvalMetrics.RMSE.name: 0.69,
    EvalMetrics.MASE.name: 1.55,
    EvalMetrics.MdASE.name: 1.55,
}


def plot_metric_results(
        company_name: str, col_name: str, min_cont_length: int,
        metric_names: List[str], model_names: List[str],
        scale: float = 3.5,
):
    if metric_names == ["all"]:
        metric_names = [m.name for m in EvalMetrics]

    if model_names == ["all"]:
        model_names = [m.name.lower() for m in ForecastModel]

    results_df = read_global_results_csv_file_name(
        company_name=company_name, col_name=col_name, min_cont_length=min_cont_length,
    )

    for horizon, per_horizon_results_df in results_df.groupby("horizon"):
        per_horizon_results_df = per_horizon_results_df.loc[
            [(horizon, model_name) for model_name in model_names],
            tuple(metric_names),
        ]

        fig, ax = plt.subplots(figsize=(5 * scale, 1 * scale), dpi=100)
        ax.set_title(f"Company {company_name} {horizon}-month Forecasting Errors ({', '.join(metric_names)})")

        xs = np.arange(len(metric_names))

        if len(model_names) > len(_COLORS):
            logging.warning(
                f"The number of models ({len(model_names)}) exceeded the"
                f"number of supported colors ({len(_COLORS)})."
            )

        bar_width = 0.7 / len(model_names)
        max_ylim = max(_YLIM_PER_METRIC_NAME[metric_name] for metric_name in metric_names)

        for model_idx, (model_name, metric_values) in enumerate(zip(model_names, per_horizon_results_df.values)):
            offset = bar_width * (model_idx - len(model_names) / 2.) * 1.2
            ax.bar(
                xs + offset, metric_values,
                width=bar_width, label=model_name,
                color=_COLOR_PER_MODEL_NAME[model_name],
            )

        ax.set_xticks(xs, metric_names)
        ax.set_ylim((0, max_ylim))
        ax.set_yticks(np.arange(0., max_ylim, 0.1))

        ax.legend(loc="upper center", ncols=len(model_names))
        fig.tight_layout()

        fig_file_path = join("results", f"metrics_{company_name}_{horizon}.{'_'.join(metric_names)}.png")
        print(f"Writing to {fig_file_path}...")
        plt.savefig(join("results", f"metrics_{company_name}_{horizon}.{'_'.join(metric_names)}.png"))
        print(f"Writing to {fig_file_path}... DONE!")

        plt.close()


def _main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("company_names", type=str, nargs="+")
    arg_parser.add_argument(
        "--metric-names", "-m", type=str, nargs="+",
        choices=[*[m.name for m in EvalMetrics], "all"], default=["all"],
    )
    arg_parser.add_argument(
        "--model-names", "-M", type=str, nargs="+",
        choices=[*[m.name.lower() for m in ForecastModel], "all"], default=["all"],
    )
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--min-cont-length", "-l", metavar="", type=int, default=13)

    args = arg_parser.parse_args()

    for company_name in args.company_names:
        plot_metric_results(
            company_name=company_name,
            col_name=args.col_name,
            min_cont_length=args.min_cont_length,
            metric_names=args.metric_names,
            model_names=args.model_names,
        )


if __name__ == '__main__':
    _main()
