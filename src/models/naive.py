from typing import List, Tuple

import numpy as np
from keras import Model
from keras.metrics import RootMeanSquaredError
from numpy import ndarray


class NaiveModel(Model):

    def call(self, inputs, training=None, mask=None):
        return inputs


def _build(loss_name: str = "mean_squared_error"):
    model = NaiveModel()
    model.compile(loss=loss_name, metrics=[RootMeanSquaredError()])

    return model


def preprocess_naive(cont_seqs: List[ndarray]) -> Tuple[List[ndarray], List[ndarray]]:
    Xs, Ys = [], []
    for cont_seq in cont_seqs:
        x, y = cont_seq[:-1], cont_seq[1:]

        Xs.append(x)
        Ys.append(y)

    return Xs, Ys


def _eval(model: Model, Xs: List[ndarray], Ys: List[ndarray]):
    scores = [
        model.evaluate(x, y, verbose=0, batch_size=32)
        for x, y in zip(Xs, Ys)
    ]

    avg_scores = np.average(scores, axis=0)
    print(f"avg_scores = {avg_scores}")


def build_and_train_naive(cont_seqs: List[ndarray], *args, **kwargs):
    Xs, Ys = preprocess_naive(cont_seqs=cont_seqs)

    model = _build()
    _eval(model=model, Xs=Xs, Ys=Ys)

    return model
