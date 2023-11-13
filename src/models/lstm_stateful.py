from time import time
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Dense, LSTM
from keras.metrics import RootMeanSquaredError
from keras.models import Sequential
from numpy import ndarray

from models.base_exp import BaseExpModel


def _build(n_neuron: int, n_feature: int, lookback: int, batch_size: int,
           loss_name: str = "mean_squared_error",
           optimizer_name: str = "adam",
           ):
    model = Sequential([
        # TODO: more layers
        LSTM(n_neuron, batch_input_shape=(batch_size, lookback, n_feature), stateful=True),
        # LSTM(n_neuron, batch_input_shape=(1, lookback, n_feature), stateful=True),
        Dense(1),
    ])
    model.compile(loss=loss_name, optimizer=optimizer_name, metrics=[RootMeanSquaredError()])

    return model


def preprocess_lstm_stateful(cont_seqs: List[ndarray], lookback: int) -> Tuple[List[ndarray], List[ndarray], int]:
    Xs, Ys = [], []
    for cont_seq in cont_seqs:
        # Make overlapping short sequences with length `lookback + 1`
        # e.g. from [11 .. 42] and `lookback = 3` gives [11, 12, 13, 14], [12, 13, 14, 15], ... [39, 40, 41, 42]
        short_seqs = np.asarray([
            cont_seq[idx: idx + lookback + 1]
            for idx in range(len(cont_seq) - (lookback + 1))
        ])

        if short_seqs.ndim == 2:
            # Reshape to (n_sample, lookback + 1, n_feature)
            short_seqs = short_seqs[..., np.newaxis]

        x, y = short_seqs[:, :-1], short_seqs[:, -1:]

        assert x.shape == (len(cont_seq) - lookback - 1, lookback, 1)
        assert y.shape == (len(cont_seq) - lookback - 1, 1, 1)

        Xs.append(x)
        Ys.append(y)

    Xs = tf.convert_to_tensor(Xs)
    Ys = tf.convert_to_tensor(Ys)
    print(f"len(Xs) = {len(Xs)}")
    print(f"len(Ys) = {len(Ys)}")

    # Verify all training sample has the same shape (i.e. all cont_seqs has the same length)
    assert len({x.shape for x in Xs}) == 1
    assert len({y.shape for y in Ys}) == 1
    assert len({*[x.shape[0] for x in Xs], *[y.shape[0] for y in Ys]}) == 1

    batch_size = Xs[0].shape[0]

    return tf.convert_to_tensor(Xs), Ys, batch_size


def _train(model: Model, Xs: List[ndarray], Ys: List[ndarray], n_epoch: int):
    for epoch in range(n_epoch):
        epoch_start_time = time()

        _epoch_losses = []
        _epoch_rmses = []
        for x, y in zip(Xs, Ys):
            loss, rmse = model.train_on_batch(x, y)
            model.reset_states()

            _epoch_losses.append(loss)
            _epoch_rmses.append(rmse)

        _avg_epoch_loss = np.average(_epoch_losses, axis=0)
        _avg_epoch_rmse = np.average(_epoch_rmses, axis=0)

        epoch_time = time() - epoch_start_time

        print(
            f"Epoch: {epoch + 1:3d}/{n_epoch}: "
            f"_avg_epoch_loss = {_avg_epoch_loss:.4f}; _avg_epoch_rmse = {_avg_epoch_rmse:.4f} "
            f"({epoch_time:.2f} s)"
        )


class StatefulLSTM(BaseExpModel):

    def __init__(self, lookback: int = 3, n_neuron: int = 32):
        super(BaseExpModel, self).__init__()
        self.lookback: int = lookback
        self.n_neuron: int = n_neuron

    def _build_and_train(self, cont_seqs: List[ndarray], n_epoch: int, *args, **kwargs):
        Xs, Ys, batch_size = preprocess_lstm_stateful(cont_seqs=cont_seqs, lookback=self.lookback)

        model = _build(n_neuron=self.n_neuron, n_feature=Xs[0].shape[2], lookback=self.lookback, batch_size=batch_size)
        _train(model=model, Xs=Xs, Ys=Ys, n_epoch=n_epoch)

        return model

    def _eval(self, cont_seqs: List[ndarray], *args, **kwargs):
        Xs, Ys, batch_size = preprocess_lstm_stateful(cont_seqs=cont_seqs, lookback=self.model.input_shape[-1])

        scores = []
        for x, y in zip(Xs, Ys):
            _scores = self.model.test_on_batch(x, y, return_dict=True)
            scores.append(_scores)
            self.model.reset_states()

        avg_loss, avg_rmse = np.average(scores, axis=0).astype(float)
        return avg_loss, avg_rmse
