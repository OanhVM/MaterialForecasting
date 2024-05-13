from collections import defaultdict
from random import Random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Reshape
from keras.metrics import RootMeanSquaredError
from keras.src.optimizers import Adam
from numpy import ndarray


def get_datasets(
        cont_seqs: List[ndarray], out_n_step: int, val_ratio: float = 0.2,
) -> Tuple[
    tf.data.Dataset, tf.data.Dataset,
]:
    def _make_dataset(__cont_seqs: List[ndarray]) -> tf.data.Dataset:
        assert len(__cont_seqs) > 0

        __cont_seqs = np.asarray(__cont_seqs).astype(np.float32)[..., np.newaxis]

        inputs = __cont_seqs[:, :-out_n_step, :]
        labels = __cont_seqs[:, -out_n_step:, :]

        return tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(len(__cont_seqs))

    Random(42).shuffle(cont_seqs)

    cont_seqs_per_width = defaultdict(list)
    for cont_seq in cont_seqs:
        cont_seqs_per_width[len(cont_seq)].append(cont_seq)

    train_ds, val_ds = None, None
    for _cont_seqs in cont_seqs_per_width.values():
        if len(_cont_seqs) > 1:
            val_size = max(int(round(len(_cont_seqs) * val_ratio)), 1)

            _train_ds = _make_dataset(_cont_seqs[:-val_size])
            _val_ds = _make_dataset(_cont_seqs[-val_size:])

            train_ds = _train_ds if train_ds is None else train_ds.concatenate(_train_ds)
            val_ds = _val_ds if val_ds is None else val_ds.concatenate(_val_ds)

    return train_ds, val_ds


def eval_model(
        model: Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, horizons: List[int],
) -> Tuple[
    List[float], Tuple[tf.data.Dataset, tf.data.Dataset],
]:
    labels, preds = [], []
    for _inputs, _labels in train_ds.concatenate(val_ds):
        _inputs = tf.convert_to_tensor(_inputs)
        _labels = tf.convert_to_tensor(_labels)
        _preds = model(_inputs)

        labels.extend(_labels)
        preds.extend(_preds)

    labels = tf.convert_to_tensor(labels)
    preds = tf.convert_to_tensor(preds)

    squared_errs = (labels - preds) ** 2
    rmses = [
        float(tf.reduce_mean(squared_errs[:, :horizon]) ** 0.5)
        for horizon in horizons
    ]
    return rmses, (labels, preds)


def build_and_train_lstm(
        train_ds: tf.data.Dataset, val_ds: tf.data.Dataset,
        n_neuron: int, out_n_step: int, n_epoch: int,
        loss_name: str = "mean_squared_error", learning_rate: float = 0.5e-3,
) -> Model:
    model = Sequential([
        LSTM(n_neuron),
        Dense(out_n_step, kernel_initializer=tf.initializers.zeros()),
        Reshape([out_n_step, 1]),
    ])
    model.compile(loss=loss_name, optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    model.fit(
        train_ds, validation_data=val_ds, epochs=n_epoch,
        callbacks=[EarlyStopping(monitor="val_loss", patience=int(round(0.05 * n_epoch)), mode="min")],
        verbose=2,
    )

    return model


def train_and_eval_lstm(
        cont_seqs: List[ndarray], horizons: List[int],
        n_neuron: int, n_epoch: int = 200,
) -> List[float]:
    out_n_step = max(horizons)

    # TODO: make optionally rebalanced
    train_ds, val_ds = get_datasets(cont_seqs=cont_seqs, out_n_step=out_n_step)

    # TODO: make federated
    model = build_and_train_lstm(
        train_ds=train_ds, val_ds=val_ds,
        n_neuron=n_neuron, out_n_step=out_n_step, n_epoch=n_epoch,
    )

    rmses, _ = eval_model(model=model, train_ds=train_ds, val_ds=val_ds, horizons=horizons)

    return rmses
