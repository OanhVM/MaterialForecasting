import logging
from argparse import ArgumentParser
from os.path import join
from typing import List

import numpy as np
from matplotlib import pyplot as plt, colormaps

from common import read_global_results_csv_file_name
from run_eval import EvalMetrics
from run_forecast import ForecastModel

_COLORS = np.roll(colormaps["tab20"].colors, 1, axis=0)


def plot_metric_results(
        company_name: str, col_name: str, min_cont_length: int,
        metric_names: List[str], model_names: List[str],
        scale: float = 1.0,
):
    results_df = read_global_results_csv_file_name(
        company_name=company_name, col_name=col_name, min_cont_length=min_cont_length,
    )

    for horizon, per_horizon_results_df in results_df.groupby("horizon"):
        per_horizon_results_df = per_horizon_results_df.loc[
            [(horizon, model_name) for model_name in model_names],
            tuple(metric_names),
        ]

        fig, ax = plt.subplots(figsize=(12 * scale, 4 * scale), dpi=150)
        ax.set_title(f"Company {company_name} {horizon}-month Forecasting Errors ({', '.join(metric_names)})")

        xs = np.arange(len(metric_names))

        if len(model_names) > len(_COLORS):
            logging.warning(
                f"The number of models ({len(model_names)}) exceeded the"
                f"number of supported colors ({len(_COLORS)})."
            )

        bar_width = 0.7 / len(model_names)

        for model_idx, (model_name, metric_values) in enumerate(zip(model_names, per_horizon_results_df.values)):
            offset = bar_width * (model_idx - len(model_names) / 2.) * 1.2
            ax.bar(xs + offset, metric_values, bar_width, label=model_name, color=_COLORS[model_idx])

        ax.set_xticks(xs, metric_names)
        ax.set_ylim((0, per_horizon_results_df.values.max() * 1.15))

        ax.legend(loc="upper left", ncols=len(model_names))
        fig.tight_layout()

        plt.savefig(join("results", f"metrics_{company_name}_{horizon}.{'_'.join(metric_names)}.png"))
        plt.close()


def _main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("company_names", type=str, nargs="+")
    arg_parser.add_argument(
        "--metric-names", "-m", type=str, nargs="+", required=True,
        choices=[*[m.name for m in EvalMetrics], "all"],
    )
    arg_parser.add_argument(
        "--model-names", "-M", type=str, nargs="+", required=True,
        choices=[*[m.name.lower() for m in ForecastModel], "all"],
    )
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--min-cont-length", "-l", metavar="", type=int, default=2)

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
