from argparse import ArgumentParser
from os import makedirs
from os.path import join, dirname
from typing import Tuple, List, Dict

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from models.lstm import get_input_label_pairs_per_input_width
from run_forecast import get_inputs_and_labels


def _make_plot(company_name: str,
               original_freq_per_input_width: Dict[int, int],
               balanced_freq_per_input_width: Dict[int, int],
               scale: float = 4,
               ):
    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharey="all",
        figsize=(3 * scale, scale), dpi=150,
    )

    for ax, _freq_per_input_width, ax_title in zip(
            axes,
            [original_freq_per_input_width, balanced_freq_per_input_width],
            ["Original", "Balanced"],
    ):
        freqs = np.asarray(list(_freq_per_input_width.values()))
        xs = np.arange(len(freqs))

        ax.bar(xs, freqs, width=0.7)

        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.set_title(f"{ax_title}")
        ax.set_xlabel("Input Width Categories")

    fig.suptitle(f"Company {company_name} Training Input Width Distribution")
    fig.supylabel("Frequency", ha="center")

    fig.tight_layout()

    fig_file_path = join("results", "input_width_distr", f"input_width_distr_{company_name}.png")
    print(f"Writing to {fig_file_path}...")
    makedirs(dirname(fig_file_path), exist_ok=True)
    plt.savefig(fig_file_path)
    print(f"Writing to {fig_file_path}... DONE!")

    plt.close()


def _get_freq_per_input_width(
        input_label_pairs_per_input_width: Dict[int, List[Tuple[ndarray, ndarray]]],
) -> Dict[int, int]:
    return dict(sorted(
        {
            input_width: len(input_label_pairs)
            for input_width, input_label_pairs in input_label_pairs_per_input_width.items()
        }.items()
    ))


def plot_err_distr(company_name: str, col_name: str, min_cont_length: int, label_width: int,
                   data_dir_path: str = "data",
                   ):
    inputs, labels = get_inputs_and_labels(
        company_name=company_name, col_name=col_name,
        min_cont_length=min_cont_length, label_width=label_width,
        data_dir_path=data_dir_path,
    )

    original_freq_per_input_width = _get_freq_per_input_width(
        input_label_pairs_per_input_width=get_input_label_pairs_per_input_width(
            inputs=inputs, labels=labels, do_balance=False,
        )
    )
    balanced_freq_per_input_width = _get_freq_per_input_width(
        input_label_pairs_per_input_width=get_input_label_pairs_per_input_width(
            inputs=inputs, labels=labels, do_balance=True,
        )
    )

    assert list(original_freq_per_input_width.keys()) == list(balanced_freq_per_input_width.keys())

    _make_plot(
        company_name=company_name,
        original_freq_per_input_width=original_freq_per_input_width,
        balanced_freq_per_input_width=balanced_freq_per_input_width,
    )


def _main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("company_names", type=str, nargs="+")
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--min-cont-length", "-l", metavar="", type=int, default=13)
    arg_parser.add_argument("--label-width", metavar="", type=int, default=6)

    args = arg_parser.parse_args()

    for company_name in args.company_names:
        plot_err_distr(
            company_name=company_name,
            col_name=args.col_name,
            min_cont_length=args.min_cont_length,
            label_width=args.label_width,
        )


if __name__ == '__main__':
    _main()
