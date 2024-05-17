from os import makedirs
from os.path import join, abspath, pardir, isfile, dirname
from typing import Optional, Tuple, List, Union

import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from parse import parse

CONT_SEQS_FILE_NAME_FORMAT = "{company_name}_cont_seqs_min_{min_cont_length:d}.csv"
EVAL_OUTPUTS_FILE_NAME_FORMAT = "{result_type}_{company_name}_{col_name}_l{min_cont_length:d}_lw{label_width:d}.csv"

# TODO: see if `_cont_seq` prefix will still apply to filled sequences (to be implemented)
CLUSTER_RESULTS_BASE_NAME_FORMAT = (
    "{company_name}_cont_seq_clusters_l{selected_cont_length:d}_k{n_cluster:d}_d{n_dim:d}"
)
DELIMITER_PER_COMPANY_NAME = {
    "A": ",",
    "B": ";",
    "C": ";",
}

PROJECT_ROOT_DIR_PATH = abspath(join(abspath(__file__), pardir, pardir))

RESULTS_DIR_PATH = join(PROJECT_ROOT_DIR_PATH, "results")
GLOBAL_RESULTS_CSV_FILE_PATH = join(RESULTS_DIR_PATH, "global_results.csv")


def update_global_results_csv_file_name(
        model_name: str,
        company_name: str, col_name: str,
        min_cont_length: int,
        horizons: List[int], rmses: List[float],
):
    print(f"Writing to {GLOBAL_RESULTS_CSV_FILE_PATH}...")

    index_record = {
        "model_name": model_name,
        "company_name": company_name,
        "col_name": col_name,
        "min_cont_length": min_cont_length,
    }

    if isfile(GLOBAL_RESULTS_CSV_FILE_PATH):
        results_df = pd.read_csv(GLOBAL_RESULTS_CSV_FILE_PATH, index_col=tuple(index_record.keys()))
        results_df.loc[tuple(index_record.values())] = rmses
    else:
        results_df = DataFrame.from_records([{
            **index_record,
            **{f"rmse_{horizon}": rmse for horizon, rmse in zip(horizons, rmses)},
        }]).set_index(tuple(index_record.keys()))

    makedirs(dirname(GLOBAL_RESULTS_CSV_FILE_PATH), exist_ok=True)
    results_df.to_csv(GLOBAL_RESULTS_CSV_FILE_PATH, float_format="%.6f")

    print(f"Writing to {GLOBAL_RESULTS_CSV_FILE_PATH}... DONE!")


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


def read_source_csv(company_name: str, data_dir_path: str = "data"):
    data = pd.read_csv(
        join(data_dir_path, company_name, f"{company_name}.csv"),
        delimiter=DELIMITER_PER_COMPANY_NAME[company_name],
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


def read_selected_data_csv(company_name: str):
    return pd.read_csv(
        get_selected_data_csv_file_path(company_name=company_name),
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


def get_eval_results_file_path(
        result_type: str,
        company_name: str, col_name: str, min_cont_length: int, label_width: int,
        data_dir_path: str = "data",
) -> str:
    assert (result_type in ("inputs", "labels") or result_type.startswith("preds_"))

    return join(
        data_dir_path, company_name,
        EVAL_OUTPUTS_FILE_NAME_FORMAT.format(
            result_type=result_type,
            company_name=company_name, col_name=col_name,
            min_cont_length=min_cont_length, label_width=label_width,
        ),
    )


def save_eval_results(eval_results: List[ndarray], eval_results_file_path: str):
    print(f"Writing to {eval_results_file_path}...")
    DataFrame(eval_results).to_csv(eval_results_file_path, header=False, index=False)
    print(f"Writing to {eval_results_file_path}... DONE!")


def read_eval_results(eval_results_file_path: str) -> List[ndarray]:
    print(f"Reading from {eval_results_file_path}...")
    eval_results = [
        r.dropna().values for _, r in pd.read_csv(
            eval_results_file_path, header=None, index_col=None,
        ).iterrows()
    ]
    print(f"Reading from {eval_results_file_path}... DONE!")
    return eval_results


def read_cont_seqs_csv(
        company_name: str, col_name: Optional[str] = None, min_cont_length: int = 2,
        data_dir_path: str = "data",
) -> List[Union[ndarray, DataFrame]]:
    csv_file_path = join(
        data_dir_path, company_name,
        CONT_SEQS_FILE_NAME_FORMAT.format(company_name=company_name, min_cont_length=2),
    )

    data = pd.read_csv(
        csv_file_path,
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

    cont_seqs = [
        cont_seq[col_name].values if col_name else cont_seq
        for _, cont_seq in data.groupby("SequenceName")
        if len(cont_seq) >= min_cont_length
    ]
    print(f"Found {len(cont_seqs)} cont_seqs of min length {min_cont_length} in {csv_file_path}.")

    return cont_seqs


def read_cont_seqs_cluster_csv(company_name: str, col_name: str,
                               selected_cont_length: int, n_cluster: int, n_dim: int, cluster_idx: int,
                               data_dir_path: str = "data",
                               ) -> List[ndarray]:
    csv_file_path = get_cont_seqs_cluster_csv_file_path(
        company_name=company_name, selected_cont_length=selected_cont_length, n_cluster=n_cluster, n_dim=n_dim,
        cluster_idx=cluster_idx,
        data_dir_path=data_dir_path,
    )

    data = pd.read_csv(
        csv_file_path,
        dtype={col_name: float, "SeqIdx": int},
    )

    cont_seqs = [cont_seq[col_name].values for _, cont_seq in data.groupby("SeqIdx")]
    print(f"Found {len(cont_seqs)} cont_seqs in {csv_file_path}.")

    return cont_seqs
