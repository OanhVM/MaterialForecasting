from argparse import ArgumentParser
from os import makedirs
from os.path import join, dirname
from typing import Tuple, List, Dict

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from data_balancer import BalanceStrategy, balance_input_label_pairs
from run_forecast import get_inputs_and_labels


def _make_plot(company_name: str,
               balance_strategy: BalanceStrategy,
               balance_labels: List[str], original_idx_bins: List[ndarray], balanced_idx_bins: List[ndarray],
               scale: float = 5,
               ):
    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharey="all",
        figsize=(3 * scale, scale), dpi=100,
    )

    balance_strategy_dname = balance_strategy.name.lower().capitalize()

    for ax, idx_bins, ax_title in zip(
            axes,
            [original_idx_bins, balanced_idx_bins],
            ["Original", f"Balanced"],
    ):
        freqs = np.asarray([len(idx_bin) for idx_bin in idx_bins])
        xs = np.arange(len(freqs))

        ax.bar(xs, freqs, width=0.7)

        ax.set_xticks(xs)
        ax.set_xticklabels(balance_labels)

        ax.set_title(f"{ax_title}")
        ax.set_xlabel(f"{balance_strategy_dname} Categories")

    fig.suptitle(f"Company {company_name} Training Data Balancing by {balance_strategy_dname}")
    fig.supylabel("Frequency", ha="center")

    fig.tight_layout()

    fig_file_path = join(
        "results", "balance_distr", f"balance_distr_{company_name}.{balance_strategy.name.lower()}.png",
    )
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


def plot_err_distr(company_name: str, balance_strategy: BalanceStrategy,
                   col_name: str, min_cont_length: int, label_width: int,
                   data_dir_path: str = "data",
                   ):
    inputs, labels = get_inputs_and_labels(
        company_name=company_name, col_name=col_name,
        min_cont_length=min_cont_length, label_width=label_width,
        data_dir_path=data_dir_path,
    )

    _, (balance_labels, original_idx_bins, balanced_idx_bins) = balance_input_label_pairs(
        inputs=inputs, labels=labels, strategy=balance_strategy,
    )

    _make_plot(
        company_name=company_name,
        balance_strategy=balance_strategy,
        balance_labels=balance_labels,
        original_idx_bins=original_idx_bins,
        balanced_idx_bins=balanced_idx_bins,
    )


def _main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("company_names", type=str, nargs="+")
    arg_parser.add_argument(
        "--balance-strategy", "-b", type=str, required=True,
        choices=[s.name.lower() for s in BalanceStrategy],
    )
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--min-cont-length", "-l", metavar="", type=int, default=13)
    arg_parser.add_argument("--label-width", metavar="", type=int, default=6)

    args = arg_parser.parse_args()

    for company_name in args.company_names:
        plot_err_distr(
            company_name=company_name,
            balance_strategy=BalanceStrategy[args.balance_strategy.upper()],
            col_name=args.col_name,
            min_cont_length=args.min_cont_length,
            label_width=args.label_width,
        )


if __name__ == '__main__':
    _main()
