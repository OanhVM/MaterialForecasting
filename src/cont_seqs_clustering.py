from argparse import ArgumentParser
from os import makedirs
from os.path import join, basename
from shutil import rmtree
from typing import Union, List

import pandas as pd
from numpy import ndarray
from pandas import Series
from tslearn.clustering import TimeSeriesKMeans

from common import read_cont_seqs_csv, get_cluster_results_base_name


def _save_seq_clusters_csvs(seq_clusters: List[List[Union[Series, ndarray]]], dst_dir_path: str):
    for cluster_idx, seq_cluster in enumerate(seq_clusters):
        seq_cluster_dfs = [
            # Create rows of e.g. ("NormSpend", "SeqIdx") for each sequence in the cluster
            pd.concat((
                seq.reset_index(drop=True),
                pd.Series([seq_idx for _ in range(len(seq))], name="SeqIdx", dtype=int)
            ), axis=1)
            for seq_idx, seq in enumerate(seq_cluster)
        ]

        # Vertically stack rows of sequences in the cluster and save to file
        pd.concat(seq_cluster_dfs).to_csv(
            join(dst_dir_path, f"{basename(dst_dir_path)}_{cluster_idx}.csv")
        )


def cluster_cont_seqs(company_name: str, min_cont_length: int, selected_cont_length: int, col_name: str,
                      n_cluster: int, n_dim: int,
                      data_dir_path: str = "data"):
    cont_seqs = read_cont_seqs_csv(company_name=company_name, min_cont_length=min_cont_length, col_name=col_name)

    selected_cont_seqs = [
        cont_seq for cont_seq in cont_seqs
        if len(cont_seq) >= selected_cont_length
    ]

    train_cont_seqs = []
    for selected_cont_seq in selected_cont_seqs:
        train_cont_seqs.extend([
            selected_cont_seq[i: i + selected_cont_length]
            for i in range(len(selected_cont_seq) - selected_cont_length)
        ])

    print(f"company_name =            {company_name}")
    print(f"selected_cont_length =    {selected_cont_length}")
    print(f"n_cluster =               {n_cluster}")
    print(f"n_dim =                   {n_dim}")
    print(f"len(cont_seqs) =          {len(cont_seqs)}")
    print(f"len(selected_cont_seqs) = {len(selected_cont_seqs)}")
    print(f"len(train_cont_seqs) =    {len(train_cont_seqs)}")
    print()

    cluster_labels = TimeSeriesKMeans(n_clusters=n_cluster, metric="dtw").fit_predict(train_cont_seqs)
    train_cont_seq_clusters = [
        [
            train_cont_seqs[seq_idx]
            for seq_idx in range(len(train_cont_seqs))
            if cluster_labels[seq_idx] == cluster_label
        ]
        for cluster_label in set(cluster_labels)
    ]
    print(f"cluster_n_seqs = {[len(c) for c in train_cont_seq_clusters]}")
    print()

    cluster_results_base_name = get_cluster_results_base_name(
        company_name=company_name, selected_cont_length=selected_cont_length, n_cluster=n_cluster,
    )
    dst_dir_path = join(data_dir_path, company_name, cluster_results_base_name)
    rmtree(dst_dir_path, ignore_errors=True)
    makedirs(dst_dir_path, exist_ok=True)

    _save_seq_clusters_csvs(seq_clusters=train_cont_seq_clusters, dst_dir_path=dst_dir_path)


def _main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("company_names", type=str, nargs="+", help="company names")
    arg_parser.add_argument("--min-cont-length", metavar="", type=int, default=2)
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--selected-cont-length", metavar="", type=int, default=24)
    arg_parser.add_argument("--n-cluster", metavar="", type=int, default=32)
    arg_parser.add_argument("--n-dim", metavar="", type=int, default=None)

    args = arg_parser.parse_args()

    assert args.n_dim is None or 2 <= args.n_dim <= args.selected_cont_length

    for company_name in args.company_names:
        cluster_cont_seqs(
            company_name=company_name,
            min_cont_length=args.min_cont_length,
            col_name=args.col_name,
            selected_cont_length=args.selected_cont_length,
            n_cluster=args.n_cluster,
            n_dim=args.n_dim if args.n_dim is None else args.selected_cont_length,
        )


if __name__ == '__main__':
    _main()
