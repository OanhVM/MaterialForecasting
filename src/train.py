import gc
from os import scandir, makedirs
from os.path import join, dirname, isfile

import numpy
import numpy as np
from keras.backend import clear_session
from keras.layers import Dense, LSTM, Flatten
from keras.models import Sequential
from keras.models import load_model
from matplotlib import pyplot as plt

# date-time parsing function for loading the dataset
from train_data_prep import data_preparation, generate_train_data


# invert differenced value
def _inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# inverse scaling for a forecasted value
def _invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def fit_model(model, mg_x, mg_y, n_epochs: int, batch_size: int, is_lstm: bool, is_dense: bool):
    """
    mg_x = 1 material group
    # [[[0.55866564],
    #   [-0.79518701],
    #   [1.]],
    #
    #  [[-0.94841336],
    #   [1.],
    #   [-0.23165238]]]
    """

    if is_lstm:
        for _ in range(n_epochs):
            for m_x, m_y in zip(mg_x, mg_y):
                model.fit(m_x, m_y, epochs=1, verbose=0, batch_size=batch_size, shuffle=False)
                # # Reset state after each material
                model.reset_states()

    elif is_dense:
        model.fit(mg_x, mg_y, epochs=n_epochs, batch_size=batch_size)


def _build_lstm(n_neurons: int, batch_size: int, n_features: int, lookback: int):
    model = Sequential([
        # LSTM(n_neurons),
        # TODO: stateful LSTM batch_size = 1
        LSTM(n_neurons, batch_input_shape=(batch_size, lookback, n_features), stateful=True),
        Dense(1),
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


def _build_dense(n_neurons: int):
    model = Sequential([
        Flatten(),
        Dense(units=n_neurons, activation='relu'),
        Dense(units=n_neurons, activation='relu'),
        Dense(units=1)
    ])

    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


# # make a one-step forecast
# def _forecast_lstm(model, short_seqs, n_features: int = 1, batch_size: int = 1):
#     preds = []
#     for short_seq in short_seqs:
#         short_seq = short_seq.reshape(1, n_features, len(short_seq))
#         pred = model.predict(short_seq, batch_size=batch_size)[0]
#
#         preds.append(pred)
#
#     return preds


def _prediction_plot_and_evaluation(model, lookback, scaled_short_seqs_list, mg_name, model_version, company_name):
    plot_size = 4
    fig, axs = plt.subplots(plot_size, plot_size, figsize=(plot_size * 2, plot_size * 2), dpi=300)

    y_min = np.asarray(scaled_short_seqs_list).min() * 0.9
    y_max = np.asarray(scaled_short_seqs_list).max() * 1.1

    squared_errs = []
    for file_idx, scaled_test_short_seqs in enumerate(scaled_short_seqs_list[:plot_size * plot_size]):
        # y_preds = _forecast_lstm(model=model, short_seqs=scaled_test_short_seqs[:, :lookback])

        y_preds = model.predict(scaled_test_short_seqs[:, :lookback, np.newaxis], batch_size=1)
        y_truths = scaled_test_short_seqs[:, -1:]

        _squared_errs = (y_truths - y_preds) ** 2
        squared_errs.extend(_squared_errs)

        row_no = int(file_idx / plot_size)
        col_no = file_idx % plot_size

        ax = axs[row_no, col_no]
        ax.plot(y_truths, color='blue')
        ax.plot(y_preds, color='orange')
        ax.set_ylim([y_min, y_max])

    plt.savefig(join("models", company_name, model_version, mg_name, model_version + ".png"))
    plt.close()

    rmse = float(np.mean(squared_errs) ** 0.5)

    return rmse


def sub_main(ds_name, files, lookback, do_train, model_version, company_name, feature_name,
             min_cont_seq_len: int,
             is_lstm: bool, is_dense: bool,
             batch_size, n_epochs, n_neurons, n_features,
             ):
    # one element of scaler corresponding to one short_seqs (1 file)
    (
        scalers,
        scaled_train_short_seqs_list,
        scaled_test_short_seqs_list,
        train_raw_values_list,
        test_raw_values_list,
    ) = data_preparation(
        company_name=company_name, feature_name=feature_name,
        mg_files=files, lookback=lookback,
        min_cont_seq_len=min_cont_seq_len,
    )

    if (
            len(scaled_train_short_seqs_list) ==
            len(scaled_test_short_seqs_list) ==
            len(train_raw_values_list) ==
            len(train_raw_values_list) == 0
    ):
        return None

    if do_train:
        x_train, y_train = generate_train_data(scaled_train_short_seqs_list)

        if is_dense:
            model = _build_dense(n_neurons=n_neurons)
            fit_model(model=model, mg_x=x_train, mg_y=y_train, n_epochs=n_epochs, batch_size=batch_size,
                      is_lstm=False,
                      is_dense=True)
        elif is_lstm:
            model = _build_lstm(n_neurons=n_neurons, batch_size=batch_size, n_features=n_features, lookback=lookback)
            fit_model(model=model, mg_x=x_train, mg_y=y_train, n_epochs=n_epochs, batch_size=batch_size,
                      is_lstm=True,
                      is_dense=False)
        else:
            raise ValueError()

        # model summary
        model_summary_file_path = join("models", company_name, model_version, ds_name, model_version + ".txt")
        makedirs(dirname(model_summary_file_path), exist_ok=True)
        with open(model_summary_file_path, 'w') as f:
            f.write(f"batch_size: {batch_size} \n")
            f.write(f"n_epochs:   {n_epochs} \n")
            f.write(f"n_neurons:  {n_neurons} \n")
            model.summary(print_fn=lambda _: f.write(_ + '\n'))

        model.save(join("models", company_name, model_version, ds_name, f"{model_version}.h5"))

    else:
        # load model
        model = load_model(join("models", company_name, model_version, ds_name, f"{model_version}.h5"))
        model.summary()

    rmse = _prediction_plot_and_evaluation(
        model, lookback, scaled_test_short_seqs_list, ds_name,
        model_version, company_name,
    )

    return rmse


def main():
    company_name = "A"
    feature_name = "Spend"
    model_version = f"{feature_name.lower()}_v1"

    data_dir_path = join("data", company_name, "cont_seqs")

    mg_files_per_mg_name = {}
    # 'C30210000': ['C30210000_99400004390.csv']

    # Get sorted file names
    all_mg_files = [d for d in scandir(data_dir_path) if isfile(d)]
    all_mg_files.sort(key=lambda f: f.name)

    for dir_entry in all_mg_files:
        # TODO: add checks: isfile(dir_entry) and is correctly named .csv
        resource_group_name = dir_entry.name.split("_")[0]

        if resource_group_name not in mg_files_per_mg_name:
            mg_files_per_mg_name[resource_group_name] = []

        mg_files_per_mg_name[resource_group_name].append(dir_entry.name)

    rmse_per_mg_name = {}

    selected_mg_names = [
        # "10201001",
    ]

    for idx, (mg_name, mg_files) in enumerate(list(mg_files_per_mg_name.items())):
        # for idx, (mg_name, mg_files) in enumerate(list(mg_files_per_mg_name.items())[:20]):
        if not len(selected_mg_names) or mg_name in selected_mg_names:
            print(
                f"\n"
                f"company_name = {company_name}; "
                f"mg_name = {mg_name} "
                f"({idx + 1}/{len(list(mg_files_per_mg_name.keys()))})"
                f"\n"
            )
            rmse = sub_main(
                ds_name=mg_name,
                files=mg_files,
                lookback=3,
                do_train=True,
                model_version=model_version,
                company_name=company_name,
                feature_name=feature_name,
                min_cont_seq_len=24,
                # batch size must be 1 for stateful
                batch_size=1,
                # batch_size=64,
                n_epochs=100,
                n_neurons=32,
                n_features=1,
                # TODO: replace with ModelType(Enum)
                is_dense=False,
                is_lstm=True,
            )

            rmse_per_mg_name[mg_name] = rmse

            clear_session()
            gc.collect()

    with open(join("models", company_name, model_version, f"{model_version}_rmse.csv"), 'w') as rmse_file:
        for mg_name, rmse in rmse_per_mg_name.items():
            if rmse is not None:
                rmse_file.write(str(mg_name) + ";" + str(rmse) + "\n")


# def main_singles():
#     data_dir_path = join("../data", "cont_seqs")
#
#     mno_file_per_mat_id = {}
#     for dir_entry in scandir(data_dir_path):
#         # TODO: add checks: isfile(dir_entry) and is correctly named .csv
#
#         # TODO: add proper parsing function for .csv file
#         mg_no, mat_no = splitext(dir_entry.name)[0].split("_")
#
#         mno_file_per_mat_id[f"{mg_no}_{mat_no}"] = dir_entry
#
#     for mat_id, mat_file in mno_file_per_mat_id.items():
#         sub_main(ds_name=mat_id, files=[mat_file], lookback=3, do_train=True,
#                  model_version="v1",
#                  batch_size=1,
#                  n_epochs=1000,
#                  n_neurons=32,
#                  n_features=1)


if __name__ == '__main__':
    # TODO: fill gaps
    # TODO: histogram to visualise series "noisiness" i.e. standard deviation >> data balancing
    # TODO: explore more complicated LSTM models
    # TODO: Tensorflow time series forcasting tutorial https://www.tensorflow.org/tutorials/structured_data/time_series
    main()
