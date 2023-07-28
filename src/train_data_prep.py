from numpy import ndarray
from pandas import read_csv, datetime
import numpy as np
from os.path import join, dirname, splitext
from sklearn.preprocessing import MinMaxScaler


def generate_train_data(train_list):
    concat_x_list = []
    concat_y_list = []
    for train in train_list:
        x, y = train[:, :-1], train[:, -1]
        # # rows, number of features, lookback
        # x = x.reshape(x.shape[0], 1, x.shape[1])
        # rows, lookback, n_features
        x = x.reshape(x.shape[0], x.shape[1], 1)
        concat_x_list.append(x)
        concat_y_list.append(y)

    return concat_x_list, concat_y_list


# create a differenced series
def _difference(dataset, interval: int = 1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.asarray(diff)


def data_preparation(company_name: str, feature_name: str, mg_files, lookback,
                     min_cont_seq_len: int,
                     diff_interval: int = 1):
    # load dataset
    train_short_seqs_list = []
    test_short_seqs_list = []
    train_raw_values_list = []
    test_raw_values_list = []
    for file_name in mg_files:
        raw_df = read_csv(
            join("data", company_name, "cont_seqs", file_name),
            header=0, index_col=0,
        ).squeeze()
        raw_df = raw_df["Spend"]
        raw_values = raw_df[feature_name].to_numpy()

        if len(raw_values) >= min_cont_seq_len:
            # transform data to be stationary
            diffs = _difference(raw_values, interval=diff_interval)
            # [2644.304837 - 4356.390857  4694.394835 - 1515.221035  2812.878756
            #  - 4596.021522  1123.5084    1666.193773 - 632.881926 - 714.549297
            #  2540.982422 - 3423.51571   3098.853775 - 744.839023 - 2926.783483
            #  1013.884438 - 871.394561 - 84.769743  2465.987851 - 3932.740527
            #  3919.562771  2144.843217 - 5388.995517 - 304.445692  1891.127202
            #  - 373.606475]

            # transform data to short sequences
            short_seqs = _timeseries_to_short_seqs(diffs, lookback=lookback)
            # [[2644.304837 - 4356.390857  4694.394835 - 1515.221035]
            #  [-4356.390857  4694.394835 - 1515.221035  2812.878756]]

            # split data into train_short_seqs and test_short_seqs-sets
            train_short_seqs, test_short_seqs = short_seqs[0:-12], short_seqs[-12:]
            train_raw_values, test_raw_values = raw_values[lookback + diff_interval + 1:-12], raw_values[-12:]

            print(f"file_name = {file_name}")
            print(f"len(train_short_seqs) = {len(train_short_seqs)}")
            print(f"len(test_short_seqs) = {len(test_short_seqs)}")
            print(f"len(train_raw_values) = {len(train_raw_values)}")
            print(f"len(test_raw_values) = {len(test_raw_values)}")

            # train_raw_values, test_raw_values = raw_values[0:-12], raw_values[-12:]

            train_short_seqs_list.append(train_short_seqs)
            test_short_seqs_list.append(test_short_seqs)

            train_raw_values_list.append(train_raw_values)
            test_raw_values_list.append(test_raw_values)

    print(f"len(train_short_seqs_list) = {len(train_short_seqs_list)}")
    print(f"len(test_short_seqs_list) = {len(test_short_seqs_list)}")

    # one element of scaler corresponding to one short_seqs (1 file)
    scalers, scaled_train_short_seqs_list, scaled_test_short_seqs_list = _scale(
        train_short_seqs_list, test_short_seqs_list,
    )

    print(f"len(scaled_train_short_seqs_list) = {len(scaled_train_short_seqs_list)}")
    print(f"len(scaled_test_short_seqs_list) = {len(scaled_test_short_seqs_list)}")

    return (
        scalers,
        scaled_train_short_seqs_list,
        scaled_test_short_seqs_list,
        train_raw_values_list,
        test_raw_values_list,
    )


# frame a sequence as a supervised learning problem
def _timeseries_to_short_seqs(series: ndarray, lookback: int = 1) -> ndarray:
    return np.asarray([
        series[idx: idx + lookback + 1]
        for idx in range(len(series) - (lookback + 1))
    ])


# scale train and test data to [-1, 1]
def _scale(train_short_seqs_list, test_short_seqs_list):
    scaled_train_short_seqs_list = []
    scaled_test_short_seqs_list = []
    scalers = []

    for i in range(len(train_short_seqs_list)):
        train_short_seqs = train_short_seqs_list[i]
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit([*train_short_seqs, *test_short_seqs_list[i]])
        scalers.append(scaler)

        scaled_train_short_seqs = scaler.transform(train_short_seqs)
        scaled_train_short_seqs_list.append(scaled_train_short_seqs)

        scaled_test_short_seqs = scaler.transform(test_short_seqs_list[i])
        scaled_test_short_seqs_list.append(scaled_test_short_seqs)

    return scalers, scaled_train_short_seqs_list, scaled_test_short_seqs_list
