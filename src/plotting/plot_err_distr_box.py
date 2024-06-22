from argparse import ArgumentParser
from os import makedirs
from os.path import join, dirname
from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from plotting.plot_err_distr import get_errs_per_forecast_model
from plotting.plot_eval_results import COLOR_PER_MODEL_NAME, adjust_lightness
from run_forecast import ForecastModel


def _make_plot(company_name: str, horizons: List[int], errs_per_forecast_model: Dict[ForecastModel, ndarray],
               scale: float = 4,
               ):
    fig, axes = plt.subplots(
        nrows=1, ncols=len(horizons), sharey="all",
        figsize=(len(horizons) * scale * 1.5, scale), dpi=100,
        layout="constrained",
    )

    colors = [COLOR_PER_MODEL_NAME[m.name.lower()] for m in errs_per_forecast_model.keys()]

    boxes = []
    for ax, horizon in zip(axes, horizons):
        box_plot = ax.boxplot(
            np.asarray(list(errs_per_forecast_model.values()))[:, :, horizon - 1].T,
            widths=0.3,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black"),
        )

        boxes = box_plot["boxes"]
        for patch, color in zip(boxes, colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(adjust_lightness(color, amount=0.75))

        ax.set_title(f"{horizon}-month Horizon")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylim((-0.05, 1.05))

    fig.legend(
        boxes, [m.name.lower() for m in errs_per_forecast_model.keys()],
        loc="outside lower center",
        ncols=len(errs_per_forecast_model.keys()),
    )
    fig.suptitle(f"Company {company_name} Forecasting Error Distribution")

    fig_file_path = join("results", "error_distr_box", f"error_distr_box_{company_name}.png")
    print(f"Writing to {fig_file_path}...")
    makedirs(dirname(fig_file_path), exist_ok=True)
    plt.savefig(fig_file_path)
    print(f"Writing to {fig_file_path}... DONE!")

    plt.close()


def plot_err_distr_box(model_names: List[str], company_name: str, col_name: str, min_cont_length: int,
                       horizons: List[int],
                       data_dir_path: str = "data",
                       ):
    if model_names == ["all"]:
        forecast_models: List[ForecastModel] = list(ForecastModel)
    else:
        forecast_models: List[ForecastModel] = [ForecastModel[model_name.upper()] for model_name in model_names]

    errs_per_forecast_model = get_errs_per_forecast_model(
        forecast_models=forecast_models,
        company_name=company_name,
        col_name=col_name,
        min_cont_length=min_cont_length,
        horizons=horizons,
        data_dir_path=data_dir_path,
    )

    _make_plot(
        company_name=company_name,
        errs_per_forecast_model=errs_per_forecast_model,
        horizons=horizons,
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
        plot_err_distr_box(
            model_names=args.model_names,
            company_name=company_name,
            col_name=args.col_name,
            min_cont_length=args.min_cont_length,
            horizons=args.horizons,
        )


if __name__ == '__main__':
    _main()
