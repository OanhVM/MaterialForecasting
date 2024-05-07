import math
from argparse import ArgumentParser
from os.path import join
from typing import List

from matplotlib import pyplot as plt
from numpy import ndarray
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.utils import to_time_series_dataset

from common import get_cluster_results_base_name, read_cont_seqs_cluster_csv


def _plot_cont_seq_clusters(cont_seqs_clusters: List[List[ndarray]], n_total_cont_seq: int,
                            company_name: str, selected_cont_length: int,
                            n_cluster: int, n_dim: int,
                            data_dir_path: str,
                            fig_scale: int = 3, fig_dpi: int = 100,
                            ):
    n_col = min([
        div for div in range(n_cluster, int(math.ceil(n_cluster ** 0.5)), -1)
        if n_cluster % div == 0
    ])
    n_row = n_cluster // n_col

    fig, axes = plt.subplots(
        ncols=n_col, nrows=n_row,
        figsize=(n_col * fig_scale, n_row * fig_scale),
        dpi=fig_dpi,
        layout="constrained",
    )

    for cluster_idx, cont_seqs in enumerate(cont_seqs_clusters):
        print(f"Plotting company_name = {company_name}; cluster_idx = {cluster_idx}; len(cont_seqs) = {len(cont_seqs)}")
        axis = axes[cluster_idx // n_col, cluster_idx % n_col]

        for cont_seq in cont_seqs:
            axis.plot(cont_seq, c="gray", alpha=max(1e3 / n_total_cont_seq, 1 / 255.), linewidth=0.5)

        if len(cont_seqs) > 0:
            axis.plot(dtw_barycenter_averaging(to_time_series_dataset(cont_seqs)), c="red")

        axis.set_title(f"Cluster {cluster_idx}")

    cluster_results_base_name = get_cluster_results_base_name(
        company_name=company_name, selected_cont_length=selected_cont_length,
        n_cluster=n_cluster, n_dim=n_dim,
    )
    fig_file_path = join(data_dir_path, company_name, cluster_results_base_name, f"{cluster_results_base_name}.jpg")
    print(f"Writing to {fig_file_path} ... ")
    fig.savefig(fig_file_path)
    print(f"Writing to {fig_file_path} ... DONE!")
    plt.close()


def _plot_cont_seq_clusters_dist(cont_seqs_clusters: List[List[ndarray]],
                                 company_name: str, selected_cont_length: int,
                                 n_cluster: int, n_dim: int,
                                 data_dir_path: str,
                                 fig_scale: int = 8, fig_dpi: int = 100,
                                 ):
    fig, ax = plt.subplots(
        figsize=(2 * fig_scale, 1 * fig_scale),
        dpi=fig_dpi,
        layout="constrained",
    )
    ax.bar(
        [f"{cluster_idx}" for cluster_idx in range(n_cluster)],
        [len(cont_seqs) for cont_seqs in cont_seqs_clusters],
        width=0.75,
    )
    ax.set_xlabel("Cluster indices")

    cluster_results_base_name = get_cluster_results_base_name(
        company_name=company_name, selected_cont_length=selected_cont_length,
        n_cluster=n_cluster, n_dim=n_dim,
    )
    fig_file_path = join(
        data_dir_path, company_name, cluster_results_base_name, f"{cluster_results_base_name}_dist.jpg",
    )
    print(f"Writing to {fig_file_path} ... ")
    fig.savefig(fig_file_path)
    print(f"Writing to {fig_file_path} ... DONE!")
    plt.close()


def plot_cont_seq_clusters(company_name: str, col_name: str, selected_cont_length: int, n_cluster: int, n_dim: int,
                           data_dir_path: str = "data",
                           ):
    cont_seqs_clusters = [
        read_cont_seqs_cluster_csv(
            company_name=company_name, col_name=col_name,
            selected_cont_length=selected_cont_length, n_cluster=n_cluster, n_dim=n_dim,
            cluster_idx=cluster_idx,
            data_dir_path=data_dir_path,
        )
        for cluster_idx in range(n_cluster)
    ]
    n_total_cont_seq = sum([len(cont_seqs) for cont_seqs in cont_seqs_clusters])

    _plot_cont_seq_clusters(
        cont_seqs_clusters=cont_seqs_clusters, n_total_cont_seq=n_total_cont_seq,
        company_name=company_name, selected_cont_length=selected_cont_length,
        n_cluster=n_cluster, n_dim=n_dim,
        data_dir_path=data_dir_path,
    )

    _plot_cont_seq_clusters_dist(
        cont_seqs_clusters=cont_seqs_clusters,
        company_name=company_name, selected_cont_length=selected_cont_length,
        n_cluster=n_cluster, n_dim=n_dim,
        data_dir_path=data_dir_path,
    )


def _main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("company_names", type=str, nargs="+", help="company names")
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--selected-cont-length", "-l", metavar="", type=int, default=24)
    arg_parser.add_argument("--n-cluster", "-c", metavar="", type=int, default=32)
    arg_parser.add_argument("--n-dim", "-d", metavar="", type=int, default=None)

    args = arg_parser.parse_args()

    assert args.n_dim is None or 2 <= args.n_dim <= args.selected_cont_length

    for company_name in args.company_names:
        plot_cont_seq_clusters(
            company_name=company_name,
            col_name=args.col_name,
            selected_cont_length=args.selected_cont_length,
            n_cluster=args.n_cluster,
            n_dim=args.n_dim if args.n_dim else args.selected_cont_length,
        )


if __name__ == '__main__':
    _main()
