from os import makedirs
from os.path import join, abspath, pardir, isfile, dirname
from typing import Optional, Tuple, List, Union, Dict

import pandas as pd
from keras.models import Model, load_model
from numpy import ndarray
from pandas import DataFrame
from parse import parse

CONT_SEQS_FILE_NAME_FORMAT = "{company_name}_cont_seqs_min_{min_cont_length:d}.csv"
FORECAST_DATA_FILE_NAME_FORMAT = (
    "{forecast_data_type}_{company_name}_{col_name}_l{min_cont_length:d}_lw{label_width:d}.csv"
)
MODEL_FILE_NAME_FORMAT = "{model_name}_{company_name}_{col_name}_l{min_cont_length:d}_lw{label_width:d}"

# TODO: see if `_cont_seq` prefix will still apply to filled sequences (to be implemented)
DELIMITER_PER_COMPANY_NAME = {
    "A": ",",
    "B": ";",
    "C": ";",
}

PROJECT_ROOT_DIR_PATH = abspath(join(abspath(__file__), pardir, pardir))

RESULTS_DIR_PATH = join(PROJECT_ROOT_DIR_PATH, "results")
GLOBAL_RESULTS_CSV_FILE_PATH = join(RESULTS_DIR_PATH, "global_results.csv")

_RESULTS_CSV_INDEX_COLS = ["company_name", "col_name", "min_cont_length", "horizon", "model_name"]


def update_global_results_csv_file_name(
        model_name: str,
        company_name: str, col_name: str,
        min_cont_length: int,
        metric_result_and_metric_name_per_horizon: Dict[int, Dict[str, float]],
):
    print(f"Writing to {GLOBAL_RESULTS_CSV_FILE_PATH}...")

    if isfile(GLOBAL_RESULTS_CSV_FILE_PATH):
        results_df = pd.read_csv(
            GLOBAL_RESULTS_CSV_FILE_PATH,
            index_col=_RESULTS_CSV_INDEX_COLS,
        ).sort_index()

        for horizon, metric_result_and_metric_name in metric_result_and_metric_name_per_horizon.items():
            results_df.loc[
                (company_name, col_name, min_cont_length, horizon, model_name),
                metric_result_and_metric_name.keys(),
            ] = metric_result_and_metric_name.values()

    else:
        results_df = DataFrame.from_records([
            {
                **dict(zip(_RESULTS_CSV_INDEX_COLS, (company_name, col_name, min_cont_length, horizon, model_name))),
                **metric_result_and_metric_name,
            }
            for horizon, metric_result_and_metric_name in metric_result_and_metric_name_per_horizon.items()
        ]).set_index(_RESULTS_CSV_INDEX_COLS)

    makedirs(dirname(GLOBAL_RESULTS_CSV_FILE_PATH), exist_ok=True)
    results_df.to_csv(GLOBAL_RESULTS_CSV_FILE_PATH, float_format="%.6f")

    print(f"Writing to {GLOBAL_RESULTS_CSV_FILE_PATH}... DONE!")


def read_global_results_csv_file_name(
        company_name: str, col_name: str, min_cont_length: int,
) -> DataFrame:
    print(f"Reading from {GLOBAL_RESULTS_CSV_FILE_PATH}...")

    results_df = pd.read_csv(
        GLOBAL_RESULTS_CSV_FILE_PATH,
        index_col=_RESULTS_CSV_INDEX_COLS,
    ).sort_index().loc[
        (company_name, col_name, min_cont_length)
    ]

    print(f"Reading from {GLOBAL_RESULTS_CSV_FILE_PATH}... DONE!")

    return results_df


def get_selected_data_csv_file_path(company_name: str, data_dir_path: str = "data") -> str:
    return join(data_dir_path, company_name, f"{company_name}_selected.csv")


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


def get_forecast_data_file_path(
        forecast_data_type: str,
        company_name: str, col_name: str, min_cont_length: int, label_width: int,
        data_dir_path: str = "data",
) -> str:
    assert (forecast_data_type in ("inputs", "labels") or forecast_data_type.startswith("preds_"))

    return join(
        data_dir_path, company_name,
        FORECAST_DATA_FILE_NAME_FORMAT.format(
            forecast_data_type=forecast_data_type,
            company_name=company_name, col_name=col_name,
            min_cont_length=min_cont_length, label_width=label_width,
        ),
    )


def save_forecast_data(forecast_data: List[ndarray], forecast_data_file_path: str):
    print(f"Writing to {forecast_data_file_path}...")
    DataFrame(forecast_data).to_csv(forecast_data_file_path, header=False, index=False)
    print(f"Writing to {forecast_data_file_path}... DONE!")


def read_forecast_data(forecast_data_file_path: str) -> List[ndarray]:
    print(f"Reading from {forecast_data_file_path}...")
    forecast_data = [
        r.dropna().values for _, r in pd.read_csv(
            forecast_data_file_path, header=None, index_col=None,
        ).iterrows()
    ]
    print(f"Reading from {forecast_data_file_path}... DONE!")
    return forecast_data


def get_model_file_name(
        model_name: str,
        company_name: str, col_name: str, min_cont_length: int, label_width: int,
) -> str:
    return MODEL_FILE_NAME_FORMAT.format(
        model_name=model_name,
        company_name=company_name, col_name=col_name,
        min_cont_length=min_cont_length, label_width=label_width,
    )


def parse_model_file_name(model_file_name: str) -> Tuple[str, str, str, int, int]:
    result = parse(MODEL_FILE_NAME_FORMAT, model_file_name)

    try:
        model_name = result["model_name"]
        company_name = result["company_name"]
        col_name = result["col_name"]
        min_cont_length = result["min_cont_length"]
        label_width = result["label_width"]
    except (ValueError, KeyError, TypeError):
        raise ValueError(f"Invalid model file name: {model_file_name}")

    return model_name, company_name, col_name, min_cont_length, label_width


def get_model_file_path(
        model_name: str,
        company_name: str, col_name: str, min_cont_length: int, label_width: int,
        models_dir_path: str = "models",
) -> str:
    return join(
        models_dir_path, company_name,
        get_model_file_name(
            model_name=model_name,
            company_name=company_name, col_name=col_name,
            min_cont_length=min_cont_length, label_width=label_width,
        ),
    )


def save_model(model: Model, model_file_path: str):
    print(f"Saving model to {model_file_path}...")
    model.save(filepath=model_file_path)
    print(f"Saving model to {model_file_path}... DONE!")


def read_model(model_file_path: str) -> Model:
    print(f"Loading model from {model_file_path}...")
    model = load_model(model_file_path)
    print(f"Loading model {model_file_path}... DONE!")
    return model


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
