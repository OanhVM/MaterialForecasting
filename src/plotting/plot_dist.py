from argparse import ArgumentParser
from os.path import join
from typing import Tuple, List, Literal

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from common import read_source_csv, read_selected_data_csv

_READ_FUNC_PER_TAG = {
    "raw": read_source_csv,
    "selected": read_selected_data_csv,
}


def _get_dist(company_name: str, tag: str) -> Tuple[List[Tuple[int, int]], ndarray]:
    data = _READ_FUNC_PER_TAG[tag](company_name=company_name)
    counts = data.groupby(["MaterialNo", "MaterialGroupNo"])["EpochM"].count().values

    bins = [2, 12, 24, 36, 48, counts.max() + 1]
    bin_edges = [(bins[i], bins[i + 1] - 1) for i in range(len(bins) - 1)]
    bin_sizes, _ = np.histogram(counts, bins=bins)

    return bin_edges, bin_sizes


def _make_plot(company_name: str, bin_edges: List[Tuple[int, int]], bin_sizes: ndarray, max_bin_size: int,
               tag: Literal["raw", "selected", "continuous"], do_show_numbers: bool,
               ):
    fig, ax = plt.subplots(figsize=(6 * 0.8, 8 * 0.8), dpi=150)

    ax.bar(range(len(bin_sizes)), bin_sizes, width=0.8)

    ax.set_xticks([i for i in range(len(bin_sizes))])
    ax.set_xticklabels([f"{bin_edge[0]} - {bin_edge[1]}" for bin_edge in bin_edges])

    ax.set_ylim([0, max_bin_size * 1.1])
    y_ticks = range(0, int(max_bin_size * 1.1), 25000)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y // 1000:4}K" if y > 0 else f"{y:4}" for y in y_ticks])

    tag_title_text = f" {tag.capitalize()}" if tag != "raw" else ""
    ax.set_title(
        f"Frequency distribution of{tag_title_text} SKU series lengths\n"
        f"Company {company_name} "
    )
    ax.set_xlabel("Series Length Brackets")
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    if do_show_numbers:
        ax.bar_label(ax.containers[0], label_type="edge")
        plot_file_name = f"freq_dist_series_length_{tag}_numbered_{company_name}.png"
    else:
        plot_file_name = f"freq_dist_series_length_{tag}_{company_name}.png"

    plt.savefig(join("data", company_name, plot_file_name))
    plt.close()


def _plot_dist(company_names: List[str], tag: Literal["raw", "selected", "continuous"], do_show_numbers: bool):
    dist_per_company_name = {
        company_name: _get_dist(company_name=company_name, tag=tag)
        for company_name in company_names
    }

    max_bin_size = np.concatenate([bin_sizes for _, bin_sizes in dist_per_company_name.values()]).max()
    print(f"max_bin_size = {max_bin_size}")

    for company_name, (bin_edges, bin_sizes) in dist_per_company_name.items():
        _make_plot(
            company_name=company_name,
            bin_edges=bin_edges, bin_sizes=bin_sizes, max_bin_size=max_bin_size,
            tag=tag, do_show_numbers=do_show_numbers,
        )


def _main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("tag", type=str, choices=["raw", "selected", "continuous"])
    arg_parser.add_argument("company_names", type=str, nargs="+", help="company names")
    arg_parser.add_argument("--show-numbers", action="store_true")

    args = arg_parser.parse_args()

    _plot_dist(
        company_names=args.company_names,
        tag=args.tag,
        do_show_numbers=args.show_numbers,
    )


if __name__ == '__main__':
    _main()
