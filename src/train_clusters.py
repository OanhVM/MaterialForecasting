from argparse import ArgumentParser
from typing import Optional, Literal, List

import numpy as np
from numpy import ndarray

from common import read_cont_seqs_cluster
from models.lstm_stateful import build_and_train_lstm_stateful
from models.naive import build_and_train_naive


def build_and_train_model(model_type: Literal["naive", "lstm_stateful"], n_epoch: int, cont_seqs: List[ndarray]):
    return {
        "naive": build_and_train_naive,
        "lstm_stateful": build_and_train_lstm_stateful
    }[model_type](cont_seqs=cont_seqs, n_epoch=n_epoch)


def _train_cluster(model_type: Literal["naive", "lstm_stateful"], n_epoch: int,
                   company_name: str, col_name: str, selected_cont_length: int, n_cluster: int, n_dim: int,
                   data_dir_path: str, cluster_idx: int,
                   do_diff: bool,
                   ):
    cont_seqs = read_cont_seqs_cluster(
        company_name=company_name, col_name=col_name, selected_cont_length=selected_cont_length,
        n_cluster=n_cluster, n_dim=n_dim,
        cluster_idx=cluster_idx,
        data_dir_path=data_dir_path,
    )

    print(f"len(cont_seqs) = {len(cont_seqs)}")

    cont_seqs = [np.diff(seq) for seq in cont_seqs] if do_diff else cont_seqs

    model = build_and_train_model(model_type=model_type, n_epoch=n_epoch, cont_seqs=cont_seqs)

    # TODO
    # model.save


def _train_clusters(model_type: Literal["naive", "lstm_stateful"], n_epoch: int,
                    company_name: str, col_name: str, selected_cont_length: int, n_cluster: int, n_dim: int,
                    selected_cluster_idx: Optional[int],
                    do_diff: bool,
                    data_dir_path: str = "data",
                    ):
    cluster_inds = list(range(n_cluster)) if selected_cluster_idx is None else [selected_cluster_idx]

    for cluster_idx in cluster_inds:
        _train_cluster(
            model_type=model_type, n_epoch=n_epoch,
            company_name=company_name,
            col_name=col_name,
            selected_cont_length=selected_cont_length,
            n_cluster=n_cluster, n_dim=n_dim,
            data_dir_path=data_dir_path,
            cluster_idx=cluster_idx,
            do_diff=do_diff,
        )


def _main():
    arg_parser = ArgumentParser()

    # TODO: add more `model_type`
    arg_parser.add_argument("model_type", type=str, choices=["naive", "lstm_stateful"])
    arg_parser.add_argument("n_epoch", type=int)
    arg_parser.add_argument("company_names", type=str, nargs="+")
    arg_parser.add_argument("--col-name", metavar="", type=str, default="NormSpend")
    arg_parser.add_argument("--selected-cont-length", metavar="", type=int, default=24)
    arg_parser.add_argument("--n-cluster", metavar="", type=int, default=32)
    arg_parser.add_argument("--n-dim", metavar="", type=int, default=None)
    arg_parser.add_argument("--cluster-idx", metavar="", type=int, default=None)
    arg_parser.add_argument("--do-diff", action="store_true")

    args = arg_parser.parse_args()

    assert args.n_dim is None or 2 <= args.n_dim <= args.selected_cont_length

    for company_name in args.company_names:
        _train_clusters(
            model_type=args.model_type,
            n_epoch=args.n_epoch,
            company_name=company_name,
            col_name=args.col_name,
            selected_cont_length=args.selected_cont_length,
            n_cluster=args.n_cluster,
            n_dim=args.n_dim if args.n_dim is None else args.selected_cont_length,
            selected_cluster_idx=args.cluster_idx,
            do_diff=args.do_diff,
        )


if __name__ == "__main__":
    _main()
