from os.path import join

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from common import read_cont_seqs_csv
from train_data_prep import _difference


def std_calculation_threaded(company_name: str, min_cont_length: int):
    data = read_cont_seqs_csv(company_name=company_name, min_cont_length=min_cont_length)

    grouped_cont_seq_rows = data.groupby("SequenceName")

    records = []
    for cont_seq_idx, (_, cont_seq_rows) in enumerate(grouped_cont_seq_rows):
        print(f"company_name = {company_name}; cont_seq_idx = {cont_seq_idx + 1} / {len(grouped_cont_seq_rows)}")

        spends = np.array(data["Spend"])
        spend_diffs = _difference(spends)
        scaled_spend_diffs = MinMaxScaler(feature_range=(-1, 1)).fit_transform(spend_diffs.reshape(-1, 1))
        std = np.std(scaled_spend_diffs)

        records.append({
            "MaterialNo": data["MaterialNo"][0],
            "MaterialGroupNo": data["MaterialGroupNo"][0],
            "SequenceName": data["SequenceName"][0],
            "NoOfTransactions": len(spends),
            "STD": std,
        })

    pd.DataFrame.from_records(records).to_csv(
        join("data", company_name, f"{company_name}_cont_seqs_min_{min_cont_length}_std.csv")
    )


if __name__ == '__main__':
    std_calculation_threaded("A", min_cont_length=2)
    std_calculation_threaded("B", min_cont_length=2)
    std_calculation_threaded("C", min_cont_length=2)
