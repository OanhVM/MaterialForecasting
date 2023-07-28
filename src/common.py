from os.path import join
from typing import Optional, Tuple, List

import pandas as pd
from numpy import ndarray
from parse import parse

CONT_SEQ_FILE_NAME_FORMAT = "{company_name}_cont_seqs_min_{min_cont_length:d}.csv"
# TODO: see if `_cont_seq` prefix will still apply to filled sequences (to be implemented)
CLUSTER_RESULTS_BASE_NAME_FORMAT = (
    "{company_name}_cont_seq_clusters_l{selected_cont_length:d}_k{n_cluster:d}_d{n_dim:d}"
)


def get_cluster_results_base_name(company_name: str, selected_cont_length: int, n_cluster: int,
                                  n_dim: Optional[int] = None,
                                  ) -> str:
    return CLUSTER_RESULTS_BASE_NAME_FORMAT.format(
        company_name=company_name,
        selected_cont_length=selected_cont_length,
        n_cluster=n_cluster,
        n_dim=n_dim if n_dim else selected_cont_length
    )


def parse_cluster_results_base_name(cluster_results_base_name: str) -> Tuple[str, int, int, int]:
    result = parse(CLUSTER_RESULTS_BASE_NAME_FORMAT, cluster_results_base_name)
    try:
        company_name = result["company_name"]
        selected_cont_length = result["selected_cont_length"]
        n_cluster = result["n_cluster"]
        n_dim = result["n_dim"]
    except (ValueError, KeyError, TypeError):
        raise ValueError(f"Invalid cluster results base name: {cluster_results_base_name}")

    return company_name, selected_cont_length, n_cluster, n_dim


def get_selected_data_csv_file_path(company_name: str, data_dir_path: str = "data") -> str:
    return join(data_dir_path, company_name, f"{company_name}_selected.csv")


def get_cont_seqs_cluster_csv_file_path(
        company_name: str, selected_cont_length: int, n_cluster: int, n_dim: int, cluster_idx: int,
        data_dir_path: str = "data",
):
    cluster_results_base_name = get_cluster_results_base_name(
        company_name=company_name, selected_cont_length=selected_cont_length, n_cluster=n_cluster, n_dim=n_dim,
    )

    return join(
        data_dir_path, company_name, cluster_results_base_name, f"{cluster_results_base_name}_{cluster_idx}.csv",
    )


def read_source_csv(company_name: str, delimiter: str, data_dir_path: str = "data"):
    data = pd.read_csv(
        join(data_dir_path, company_name, f"{company_name}.csv"),
        delimiter=delimiter,
        dtype={
            "MaterialNo": object,
            "MaterialGroupNo": object,
            "Y": int,
            "M": int,
            "Spend": float,
            "Quantity": float,
        },
    ).dropna()

    data["EpochM"] = data["Y"] * 12 + data["M"]

    return data[["MaterialNo", "MaterialGroupNo", "Y", "M", "EpochM", "Spend", "Quantity"]]


def read_selected_data_csv(company_name: str, data_dir_path: str = "data"):
    return pd.read_csv(
        join(data_dir_path, company_name, f"{company_name}_selected.csv"),
        dtype={
            "MaterialNo": object,
            "MaterialGroupNo": object,
            "Y": int,
            "M": int,
            "EpochM": int,
            "Spend": float,
            "Quantity": float,
        },
    )


def read_cont_seqs_csv(company_name: str, min_cont_length: int, data_dir_path: str = "data"):
    data = pd.read_csv(
        join(
            data_dir_path, company_name,
            CONT_SEQ_FILE_NAME_FORMAT.format(company_name=company_name, min_cont_length=min_cont_length),
        ),
        dtype={
            "MaterialNo": object,
            "MaterialGroupNo": object,
            "SequenceName": object,
            "Y": int,
            "M": int,
            "Spend": float,
            "Quantity": float,
            "EpochM": int
        },
    )

    return data


def read_cont_seqs_cluster(company_name: str, col_name: str,
                           selected_cont_length: int, n_cluster: int, n_dim: int, cluster_idx: int,
                           data_dir_path: str = "data",
                           ) -> List[ndarray]:
    data = pd.read_csv(
        get_cont_seqs_cluster_csv_file_path(
            company_name=company_name, selected_cont_length=selected_cont_length, n_cluster=n_cluster, n_dim=n_dim,
            cluster_idx=cluster_idx,
            data_dir_path=data_dir_path,
        ),
        dtype={col_name: float, "SeqIdx": int},
    )

    return [cont_seq[col_name].values for _, cont_seq in data.groupby("SeqIdx")]
