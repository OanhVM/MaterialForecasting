from os.path import join
from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from common import read_source_csv


def _get_raw_dist(company_name: str, delimiter: str) -> Tuple[List[Tuple[int, int]], ndarray]:
    data = read_source_csv(company_name=company_name, delimiter=delimiter)

    counts = data.groupby(["MaterialNo", "MaterialGroupNo"])["EpochM"].count()
    bins = [2, 12, 24, 36, 48, counts.max() + 1]

    bin_edges = [(bins[i], bins[i + 1] - 1) for i in range(len(bins) - 1)]

    bin_sizes, _ = np.histogram(counts, bins=bins)

    return bin_edges, bin_sizes


def get_dist_per_company_name(company_name_delimiter_pairs: List[Tuple[str, str]]):
    return {
        _get_raw_dist(company_name=company_name, delimiter=delimiter)
        for company_name, delimiter in company_name_delimiter_pairs
    }


def _plot(company_name: str, bin_edges: List[Tuple[int, int]], bin_sizes: ndarray, max_bin_size: int):
    fig, ax = plt.subplots(figsize=(6 * 0.8, 8 * 0.8), dpi=150)

    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(bin_sizes)), bin_sizes, width=0.8)
    # ax.bar_label(ax.containers[0], label_type="edge")

    ax.set_xticks([i for i in range(len(bin_sizes))])
    ax.set_xticklabels([f"{bin_edge[0]} - {bin_edge[1]}" for bin_edge in bin_edges])

    ax.set_ylim([0, max_bin_size * 1.1])
    y_ticks = range(0, int(max_bin_size * 1.1), 25000)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y // 1000:4}K" if y > 0 else f"{y:4}" for y in y_ticks])

    ax.set_title(
        f"Frequency distribution of SKU series lengths\n"
        f"Company {company_name} "
    )
    ax.set_xlabel("Series Length Brackets")
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    plt.savefig(join("data", company_name, f"raw_freq_dist_series_length_{company_name}.png"))
    plt.close()


def main(company_name_delimiter_pairs: List[Tuple[str, str]]):
    dist_per_company_name = {
        company_name: _get_raw_dist(company_name=company_name, delimiter=delimiter)
        for company_name, delimiter in company_name_delimiter_pairs
    }

    max_bin_size = np.concatenate([bin_sizes for _, bin_sizes in dist_per_company_name.values()]).max()
    print(f"max_bin_size = {max_bin_size}")

    for company_name, (bin_edges, bin_sizes) in dist_per_company_name.items():
        _plot(
            company_name=company_name,
            bin_edges=bin_edges, bin_sizes=bin_sizes, max_bin_size=max_bin_size,
        )


if __name__ == '__main__':
    main([
        ("A", ","),
        ("B", ";"),
        ("C", ";"),
    ])
