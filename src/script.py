from os import DirEntry, scandir
from os.path import join, isfile
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_example():
    company_name = "B"
    dir_path = join("data", company_name, "cont_seqs")

    # continuous_seqs_lens = []

    files: List[DirEntry] = [_f for _f in scandir(dir_path) if isfile(_f.path)]

    for file in files:
        if file.name.endswith(".csv"):
            results = pd.read_csv(file.path)
            # continuous_seqs_lens.append(len(results))

            if len(results) > 25:
                print(file.name)


def discrete_data_histogram(company_name, delimiter):
    data = pd.read_csv(
        join("data", company_name, f"{company_name}.csv"),
        delimiter=delimiter,
        dtype={
            "MaterialNo": object,
            "MaterialGroupNo": object,
            "Y": int,
            "M": int,
            "Spend": float,
            "Quantity": float,
        },
    )
    data["EpochM"] = data["Y"] * 12 + data["M"]

    discrete_data_length = []
    for i, group_rows in data.groupby(["MaterialNo", "MaterialGroupNo"]):
        group_rows = group_rows.sort_values(by=["EpochM"])
        discrete_data_length.append(len(group_rows))

    discrete_data_length = np.asarray(discrete_data_length)

    bins = [2, 12, 24, 36, 48, discrete_data_length.max()]

    hist, bin_edges = np.histogram(discrete_data_length, bins=bins)

    fig, ax = plt.subplots(figsize=(6, 9), dpi=100)

    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(hist)), hist, width=0.8)

    # Set the ticks to the middle of the bars
    ax.set_xticks([i for i in range(len(hist))])
    # Set the xticklabels to a string that tells us what the bin edges were
    ax.set_xticklabels([f"{bins[i]} - {bins[i + 1] - 1}" for i, j in enumerate(hist)])

    ax.set_yticklabels([f"{y}K" for y in ax.get_yticks() // 1000])

    ax.set_title(
        f"Frequency distribution of series lengths by materials\n"
        f"Company {company_name}"
    )
    ax.set_xlabel("Length brackets")
    ax.set_ylabel("Frequency")

    plt.savefig(join("data", company_name, "freq_dist_series_length_by_materials.png"))
    plt.close()


if __name__ == '__main__':
    discrete_data_histogram("A", ",")
    discrete_data_histogram("B", ";")
    discrete_data_histogram("C", ";")
