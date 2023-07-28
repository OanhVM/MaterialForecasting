from time import time
from typing import List, Tuple

import numpy as np
from keras import Model
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.metrics import RootMeanSquaredError
from numpy import ndarray


def _build(n_neuron: int, n_feature: int, lookback: int,
           loss_name: str = "mean_squared_error",
           optimizer_name: str = "adam",
           ):
    model = Sequential([
        # LSTM(n_neurons),
        # TODO: stateful LSTM batch_size = 1
        # LSTM(n_neuron, batch_input_shape=(batch_size, lookback, n_feature), stateful=True),
        LSTM(n_neuron, batch_input_shape=(1, lookback, n_feature), stateful=True),
        Dense(1),
    ])
    model.compile(loss=loss_name, optimizer=optimizer_name, metrics=[RootMeanSquaredError()])

    return model


def preprocess_lstm_stateful(cont_seqs: List[ndarray], lookback: int) -> Tuple[List[ndarray], List[ndarray]]:
    Xs, Ys = [], []
    for cont_seq in cont_seqs:
        # Make overlapping short sequences with length `lookback + 1`
        # e.g. from [11 .. 42] and `lookback = 3` gives [11, 12, 13, 14], [12, 13, 14, 15], ... [39, 40, 41, 42]
        short_seqs = np.asarray([
            cont_seq[idx: idx + lookback + 1]
            for idx in range(len(cont_seq) - (lookback + 1))
        ])

        if short_seqs.ndim == 2:
            # reshape to (n_sample, lookback + 1, n_feature)
            short_seqs = short_seqs[..., np.newaxis]

        x, y = short_seqs[:, :-1], short_seqs[:, -1:]

        assert x.shape == (len(cont_seq) - lookback - 1, lookback, 1)
        assert y.shape == (len(cont_seq) - lookback - 1, 1, 1)

        Xs.append(x)
        Ys.append(y)

    return Xs, Ys


def _train(model: Model, Xs: List[ndarray], Ys: List[ndarray], n_epoch: int):
    for epoch in range(n_epoch):
        print(f"Epoch: {epoch + 1}/{n_epoch} ...")
        epoch_start_time = time()

        for x, y in zip(Xs, Ys):
            model.fit(x, y, epochs=1, verbose=0, batch_size=1, shuffle=False)
            model.reset_states()

        epoch_time = time() - epoch_start_time
        print(f"Epoch: {epoch + 1}/{n_epoch} ... DONE! ({epoch_time:.2f} s)")


def _eval(model: Model, Xs: List[ndarray], Ys: List[ndarray]):
    scores = [
        model.evaluate(x, y, verbose=0, batch_size=1)
        for x, y in zip(Xs, Ys)
    ]

    avg_scores = np.average(scores, axis=0)
    print(f"avg_scores = {avg_scores}")


def build_and_train_lstm_stateful(cont_seqs: List[ndarray], n_epoch: int,
                                  lookback: int = 3,
                                  n_neuron: int = 32,
                                  ):
    Xs, Ys = preprocess_lstm_stateful(cont_seqs=cont_seqs, lookback=lookback)

    model = _build(n_neuron=n_neuron, n_feature=Xs[0].shape[2], lookback=lookback)
    _train(model=model, Xs=Xs, Ys=Ys, n_epoch=n_epoch)
    _eval(model=model, Xs=Xs, Ys=Ys)

    return model
