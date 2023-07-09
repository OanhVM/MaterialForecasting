from os.path import join
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv


def main(company_name):
    dir_path = join("data", company_name, "cont_seqs")

    continuous_seqs_lens = []

    for file in os.listdir(dir_path):
        if file.endswith(".csv"):
            results = pd.read_csv(join(dir_path, file))
            continuous_seqs_lens.append(len(results))

    continuous_seqs_lens = np.asarray(continuous_seqs_lens)

    bins = [2, 12, 24, 36, 48, continuous_seqs_lens.max()]

    hist, bin_edges = np.histogram(continuous_seqs_lens, bins=bins)

    fig, ax = plt.subplots()

    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(hist)), hist, width=0.8)

    # Set the ticks to the middle of the bars
    ax.set_xticks([i for i in range(len(hist))])
    # Set the xticklabels to a string that tells us what the bin edges were
    ax.set_xticklabels([f"{bins[i]} - {bins[i + 1] - 1}" for i, j in enumerate(hist)])

    ax.set_yticklabels([f"{y}K" for y in ax.get_yticks() // 1000])

    ax.set_title(f"Frequency distribution of continuous series lengths - Company {company_name} ")
    ax.set_xlabel("Length brackets")
    ax.set_ylabel("Frequency")

    plt.savefig(join("data", company_name, f"freq_dist_cont_series_length_{company_name}.png"))
    plt.close()


if __name__ == '__main__':
    main(company_name="A")
    main(company_name="B")
    main(company_name="C")
