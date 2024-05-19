from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from keras import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, LSTM, Reshape
from keras.metrics import RootMeanSquaredError
from keras.src.optimizers import Adam
from numpy import ndarray
from numpy.random import default_rng

from common import save_model


def get_balanced_inputs_and_labels(
        input_label_pairs_per_input_width: Dict[int, List[Tuple[ndarray, ndarray]]],
) -> Dict[int, List[Tuple[ndarray, ndarray]]]:
    bin_sizes = np.asarray([
        len(input_label_pairs)
        for input_label_pairs in input_label_pairs_per_input_width.values()
    ])

    balanced_coeffs = bin_sizes ** 0.25
    balanced_coeffs /= balanced_coeffs.sum()

    balanced_bin_sizes = np.round(balanced_coeffs * bin_sizes.sum()).astype(int)

    balanced_input_label_pairs_per_input_width = {
        input_width: [
            input_label_pairs[_idx] for _idx in default_rng(seed=37).choice(
                list(range(len(input_label_pairs))),
                size=balanced_bin_sizes[bin_idx],
            )
        ]
        for bin_idx, (input_width, (input_label_pairs)) in enumerate(input_label_pairs_per_input_width.items())
    }

    return balanced_input_label_pairs_per_input_width


def _make_dataset(input_label_pairs: List[Tuple[ndarray, ndarray]]) -> tf.data.Dataset:
    inputs, labels = [
        np.asarray(arr).astype(np.float32)[..., np.newaxis]
        for arr in zip(*input_label_pairs)
    ]

    return tf.data.Dataset.from_tensor_slices(
        (inputs, labels)
    ).shuffle(
        buffer_size=len(input_label_pairs),
    ).batch(
        batch_size=len(input_label_pairs),
    )


def get_datasets(
        inputs: List[ndarray], labels: List[ndarray], do_balance: bool,
        val_ratio: float = 0.2,
) -> Tuple[
    tf.data.Dataset, tf.data.Dataset,
]:
    input_label_pairs_per_input_width = defaultdict(list)
    for _input, label in zip(inputs, labels):
        input_label_pairs_per_input_width[len(_input)].append((_input, label))

    if do_balance:
        input_label_pairs_per_input_width = get_balanced_inputs_and_labels(
            input_label_pairs_per_input_width=input_label_pairs_per_input_width,
        )

    train_ds, val_ds = None, None
    for input_label_pairs in input_label_pairs_per_input_width.values():
        if len(input_label_pairs) > 1:
            val_size = max(int(round(len(input_label_pairs) * val_ratio)), 1)

            _train_ds = _make_dataset(input_label_pairs[:-val_size])
            _val_ds = _make_dataset(input_label_pairs[-val_size:])

            train_ds = _train_ds if train_ds is None else train_ds.concatenate(_train_ds)
            val_ds = _val_ds if val_ds is None else val_ds.concatenate(_val_ds)

    return train_ds, val_ds


def build_and_train_lstm(
        train_ds: tf.data.Dataset, val_ds: tf.data.Dataset,
        label_width: int, n_neuron: int, n_epoch: int,
        model_file_path: str,
        loss_name: str = "mean_squared_error", learning_rate: float = 1e-3,
) -> Model:
    model = Sequential([
        LSTM(n_neuron),
        Dense(label_width, kernel_initializer=tf.initializers.zeros()),
        Reshape([label_width, 1]),
    ])
    model.compile(loss=loss_name, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    model_ckpt_file_path = f"{model_file_path}.ckpt"
    model.fit(
        train_ds, validation_data=val_ds, epochs=n_epoch,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=int(round(0.025 * n_epoch)), mode="min"),
            ModelCheckpoint(filepath=model_ckpt_file_path, save_best_only=True, save_weights_only=True),
        ],
        verbose=2,
    )

    model.load_weights(model_ckpt_file_path).expect_partial()

    return model


def model_forecast(model: Model, inputs: List[ndarray]) -> List[ndarray]:
    preds = []
    for idx, _input in enumerate(inputs):
        if idx % 100 == 0 or idx == len(inputs) - 1:
            print(f"model_forecast: idx = {idx:4d}/{len(inputs)}")

        preds.append(
            model(_input.reshape((1, -1, 1)))[0, :, 0].numpy()
        )

    return preds


def train_and_forecast_lstm(
        inputs: List[ndarray], labels: List[ndarray], label_width: int, model_file_path: str,
        n_neuron: int, do_balance: bool,
        n_epoch: int = 500,
) -> List[ndarray]:
    train_ds, val_ds = get_datasets(inputs=inputs, labels=labels, do_balance=do_balance)

    # TODO: make federated
    model = build_and_train_lstm(
        train_ds=train_ds, val_ds=val_ds,
        label_width=label_width, n_neuron=n_neuron, n_epoch=n_epoch,
        model_file_path=model_file_path,
    )

    preds = model_forecast(model=model, inputs=inputs)

    save_model(model=model, model_file_path=model_file_path)

    return preds
