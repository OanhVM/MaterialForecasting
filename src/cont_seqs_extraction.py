from os import makedirs
from os.path import join, dirname

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from common import CONT_SEQS_FILE_NAME_FORMAT, read_selected_data_csv


def _is_continuous(group_rows: DataFrame, row_idx: int):
    return group_rows["EpochM"].iloc[row_idx] - group_rows["EpochM"].iloc[row_idx - 1] == 1


def _get_norm_cont_seqs(mat_rows: DataFrame, min_cont_length: int):
    # Normalise data
    mat_rows[["NormQuantity", "NormSpend"]] = MinMaxScaler().fit_transform(mat_rows[["Quantity", "Spend"]])

    mat_rows = mat_rows.sort_values(by=["EpochM"]).reset_index(drop=True)

    continuous_seqs = [[mat_rows.iloc[0]]]
    for row_idx in range(1, len(mat_rows)):
        if _is_continuous(mat_rows, row_idx):
            continuous_seqs[-1].append(mat_rows.iloc[row_idx])
        else:
            continuous_seqs.append([mat_rows.iloc[row_idx]])

    continuous_seqs = [
        pd.DataFrame.from_records(seq)
        for seq in continuous_seqs if len(seq) >= min_cont_length
    ]

    for idx, seq in enumerate(continuous_seqs):
        seq["SequenceName"] = f"{seq['MaterialGroupNo'].iloc[0]}_{seq['MaterialNo'].iloc[0]}_{idx}"

    return continuous_seqs


def main(company_name: str, min_cont_length: int = 2):
    data = read_selected_data_csv(company_name=company_name)

    grouped_data = data.groupby(["MaterialNo", "MaterialGroupNo"])

    continuous_seqs = []
    for group_idx, (_, mat_rows) in enumerate(grouped_data):
        print(f"company_name = {company_name}; group_idx = {group_idx + 1} / {len(grouped_data)}")

        _continuous_seqs = _get_norm_cont_seqs(mat_rows=mat_rows, min_cont_length=min_cont_length)
        continuous_seqs.extend(_continuous_seqs)

    dst_csv_path = join(
        "data", company_name,
        CONT_SEQS_FILE_NAME_FORMAT.format(company_name=company_name, min_cont_length=min_cont_length),
    )
    makedirs(dirname(dst_csv_path), exist_ok=True)
    pd.concat(continuous_seqs).to_csv(dst_csv_path)


if __name__ == '__main__':
    # TODO: add argparse
    main(company_name="A")
    main(company_name="B")
    main(company_name="C")
