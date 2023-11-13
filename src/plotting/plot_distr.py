from argparse import ArgumentParser
from os.path import join
from typing import Tuple, List, Literal

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from common import read_source_csv, read_selected_data_csv, read_cont_seqs_csv

_COUNT_FUNC_PER_TAG = {
    "raw": lambda company_name: read_source_csv(
        company_name=company_name,
    ).groupby(["MaterialNo", "MaterialGroupNo"])["EpochM"].count().values,
    "selected": lambda company_name: read_selected_data_csv(
        company_name=company_name,
    ).groupby(["MaterialNo", "MaterialGroupNo"])["EpochM"].count().values,
    "continuous": lambda company_name: np.asarray([
        len(cont_seq) for cont_seq in read_cont_seqs_csv(company_name=company_name)
    ]),
}


def _get_distr(company_name: str, tag: str) -> Tuple[List[Tuple[int, int]], ndarray]:
    counts = _COUNT_FUNC_PER_TAG[tag](company_name=company_name)

    bins = [2, 12, 24, 36, 48, counts.max() + 1]
    bin_edges = [(bins[i], bins[i + 1] - 1) for i in range(len(bins) - 1)]
    bin_sizes, _ = np.histogram(counts, bins=bins)

    return bin_edges, bin_sizes


def _make_plot(company_name: str, bin_edges: List[Tuple[int, int]], bin_sizes: ndarray, max_bin_size: int,
               tag: Literal["raw", "selected", "continuous"],
               ):
    fig, ax = plt.subplots(figsize=(7 * 0.8, 8 * 0.8), dpi=150)

    ax.bar(range(len(bin_sizes)), bin_sizes, width=0.75)
    ax.bar_label(
        ax.containers[0],
        labels=[f"{bin_size / 1000:.0f}K" if bin_size > 1e4 else bin_size for bin_size in bin_sizes],
        label_type="edge",
    )

    ax.set_xticks([i for i in range(len(bin_sizes))])
    ax.set_xticklabels([
        f"{bin_edge[0]} - {bin_edge[1]}" if bin_edge[0] != bin_edge[1] else f"{bin_edge[0]}"
        for bin_edge in bin_edges
    ])

    ax.set_ylim([0, max_bin_size * 1.1])
    ax.set_yticks([])

    ax.set_title(
        f"Frequency distribution of{f' {tag}' if tag != 'raw' else ''} SKU series lengths\n"
        f"Company {company_name} "
    )
    ax.set_xlabel("Series Length Brackets")
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    plt.savefig(join("data", company_name, f"freq_distr_series_length_{tag}_{company_name}.png"))
    plt.close()


def _plot_distr(company_names: List[str], tag: Literal["raw", "selected", "continuous"], ):
    dist_per_company_name = {
        company_name: _get_distr(company_name=company_name, tag=tag)
        for company_name in company_names
    }

    max_bin_size = np.concatenate([bin_sizes for _, bin_sizes in dist_per_company_name.values()]).max()
    print(f"max_bin_size = {max_bin_size}")

    for company_name, (bin_edges, bin_sizes) in dist_per_company_name.items():
        _make_plot(
            company_name=company_name,
            bin_edges=bin_edges, bin_sizes=bin_sizes, max_bin_size=max_bin_size,
            tag=tag,
        )


def _main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("tag", type=str, choices=["raw", "selected", "continuous"])
    arg_parser.add_argument("company_names", type=str, nargs="+", help="company names")

    args = arg_parser.parse_args()

    _plot_distr(
        company_names=args.company_names,
        tag=args.tag,
    )


if __name__ == '__main__':
    _main()
