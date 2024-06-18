from argparse import ArgumentParser
from os import makedirs
from os.path import join, dirname
from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from common import get_forecast_data_file_path, read_forecast_data
from run_forecast import ForecastModel


def _get_err_distr(errs: ndarray) -> Tuple[List[Tuple[int, int]], ndarray]:
    bins = np.linspace(0., 1., num=11)
    bin_edges = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
    bin_sizes, _ = np.histogram(errs, bins=bins)

    return bin_edges, bin_sizes


def _make_plot(company_name: str, forecast_model: ForecastModel, horizons: List[int], errs: ndarray,
               scale: float = 4,
               ):
    fig, axes = plt.subplots(
        nrows=1, ncols=len(horizons), sharey="all",
        figsize=(len(horizons) * scale, scale), dpi=100,
    )

    for ax, horizon in zip(axes, horizons):
        bin_edges, bin_sizes = _get_err_distr(errs=errs[:, horizon - 1])

        ax.bar(np.arange(len(bin_sizes)) + 0.5, bin_sizes, width=0.75)
        ax.bar_label(
            ax.containers[0],
            labels=[f"{bin_size / 1000:.0f}K" if bin_size > 1e4 else bin_size for bin_size in bin_sizes],
            label_type="edge",
        )

        # ax.set_xticks([0.5, *[i + 0.5 for i in range(len(bin_sizes))]])
        ax.set_xticks(np.arange(11))

        ax.set_xticklabels(np.arange(11) / 10.)

        ax.set_title(f"{horizon}-month Horizon")
        ax.set_xlabel("Error Ranges")

    fig.suptitle(f"Company {company_name} Error Distribution - {forecast_model.display_name}")
    fig.supylabel("Frequency", ha="center")

    fig.tight_layout()

    fig_file_path = join("results", "error_distr", f"error_distr_{company_name}_{forecast_model.name.lower()}.png")
    print(f"Writing to {fig_file_path}...")
    makedirs(dirname(fig_file_path), exist_ok=True)
    plt.savefig(fig_file_path)
    print(f"Writing to {fig_file_path}... DONE!")

    plt.close()


def plot_err_distr(model_names: List[str], company_name: str, col_name: str, min_cont_length: int, horizons: List[int],
                   data_dir_path: str = "data",
                   ):
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
        errs = np.abs(labels - preds)

        _make_plot(
            company_name=company_name, forecast_model=forecast_model,
            horizons=horizons, errs=errs,
        )


def _main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("company_names", type=str, nargs="+")
    arg_parser.add_argument("--horizons", "-H", metavar="", type=int, nargs="+", default=[1, 3, 6])
    arg_parser.add_argument(
        "--model-names", "-M", type=str, nargs="+",
        choices=[*[m.name.lower() for m in ForecastModel], "all"], default=["all"],
    )
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--min-cont-length", "-l", metavar="", type=int, default=13)

    args = arg_parser.parse_args()

    for company_name in args.company_names:
        plot_err_distr(
            model_names=args.model_names,
            company_name=company_name,
            col_name=args.col_name,
            min_cont_length=args.min_cont_length,
            horizons=args.horizons,
        )


if __name__ == '__main__':
    _main()
