from os import scandir
from os.path import join

import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

from train_data_prep import _difference


def main():
    data_dir_path = join("data", "cont_seqs")

    mg_files_per_mg_name = {}
    # 'C30210000': ['C30210000_99400004390.csv']
    for dir_entry in scandir(data_dir_path):
        # TODO: add checks: isfile(dir_entry) and is correctly named .csv
        resource_group_name = dir_entry.name.split("_")[0]

        if resource_group_name not in mg_files_per_mg_name:
            mg_files_per_mg_name[resource_group_name] = []

        mg_files_per_mg_name[resource_group_name].append(dir_entry.name)

    model_version = "naive"
    rmse_per_mg_name = {}

    for idx, (mg_name, mg_files) in enumerate(list(mg_files_per_mg_name.items())):
        print(f"Material group name: {mg_name}")
        rmse = sub_main(mg_files=mg_files)

        rmse_per_mg_name[mg_name] = rmse

    with open(join("model_" + model_version, f"{model_version}_rmse.csv"), 'w') as f:
        for mg_name, rmse in rmse_per_mg_name.items():
            f.write(str(mg_name) + ";" + str(rmse) + "\n")


def sub_main(mg_files):
    train = []
    predict = []
    scalers = []

    for file_name in mg_files:
        df = read_csv(join("data", "cont_seqs", file_name),
                      header=0, index_col=0, squeeze=True)
        df = df["Spend"]
        raw_values = df.to_numpy()

        # transform data to be stationary
        diffs = _difference(raw_values, interval=1)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(diffs.reshape(-1, 1))
        scalers.append(scaler)

        train_diff = scaler.transform(diffs.reshape(-1, 1))

        train.append(train_diff[1:])
        predict.append(train_diff[:-1])

    print(f"train:{train}")
    print(f"predict:{predict}")

    squared_errs = []
    for y_truths, y_preds in zip(train, predict):
        _squared_errs = (y_truths - y_preds) ** 2
        squared_errs.extend(_squared_errs)

    rmse = float(np.mean(squared_errs) ** 0.5)
    print(rmse)

    return rmse


if __name__ == '__main__':
    main()
