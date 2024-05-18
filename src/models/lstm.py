from collections import defaultdict
from random import Random
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Reshape
from keras.metrics import RootMeanSquaredError
from keras.src.optimizers import Adam
from numpy import ndarray


def get_balanced_inputs_and_labels(
        input_label_pairs_per_input_width: Dict[int, List[Tuple[ndarray, ndarray]]],
) -> Dict[int, List[Tuple[ndarray, ndarray]]]:
    bin_sizes = np.asarray(input_label_pairs_per_input_width.keys())

    balanced_coeffs = bin_sizes ** 0.25
    balanced_coeffs /= balanced_coeffs.sum()

    balanced_bin_sizes = np.round(balanced_coeffs * bin_sizes.sum()).astype(int)

    balanced_input_label_pairs_per_input_width = {
        width: Random(37).sample(cont_seqs, k=balanced_bin_sizes[idx])
        for idx, (width, cont_seqs) in enumerate(input_label_pairs_per_input_width.items())
    }

    return balanced_input_label_pairs_per_input_width


def get_datasets(
        inputs: List[ndarray], labels: List[ndarray], do_balance: bool,
        val_ratio: float = 0.2,
) -> Tuple[
    tf.data.Dataset, tf.data.Dataset,
]:
    def _make_dataset(_input_label_pairs: List[Tuple[ndarray, ndarray]]) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensors(
            _input_label_pairs
        ).shuffle(
            buffer_size=len(_input_label_pairs),
        ).batch(
            batch_size=len(_input_label_pairs),
        ).cache()

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
        n_neuron: int, label_width: int, n_epoch: int,
        loss_name: str = "mean_squared_error", learning_rate: float = 0.5e-3,
) -> Model:
    model = Sequential([
        LSTM(n_neuron),
        Dense(label_width, kernel_initializer=tf.initializers.zeros()),
        Reshape([label_width, 1]),
    ])
    model.compile(loss=loss_name, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    model.fit(
        train_ds, validation_data=val_ds, epochs=n_epoch,
        callbacks=[EarlyStopping(monitor="val_loss", patience=int(round(0.05 * n_epoch)), mode="min")],
        verbose=2,
    )

    return model


def train_and_eval_lstm(
        inputs: List[ndarray], labels: List[ndarray], label_width: int,
        n_neuron: int, do_balance: bool,
        n_epoch: int = 200,
) -> List[ndarray]:
    train_ds, val_ds = get_datasets(inputs=inputs, labels=labels, do_balance=do_balance)

    # TODO: make federated
    model = build_and_train_lstm(
        train_ds=train_ds, val_ds=val_ds,
        n_neuron=n_neuron, label_width=label_width, n_epoch=n_epoch,
    )

    preds = []
    for _inputs, _ in train_ds.concatenate(val_ds):
        preds.extend(model.predict(_inputs))

    return preds
