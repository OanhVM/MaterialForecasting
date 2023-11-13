from typing import List, Tuple

import numpy as np
from keras import Model
from keras.metrics import RootMeanSquaredError
from numpy import ndarray

from models.base_exp import BaseExpModel


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


def evaluate_naive(cont_seqs: List[ndarray]) -> Tuple[float, float]:
    Xs, Ys = preprocess_naive(cont_seqs=cont_seqs)

    model = _build()

    scores = []
    for x, y in zip(Xs, Ys):
        _scores = model.test_on_batch(x, y)
        scores.append(_scores)

    avg_loss, avg_rmse = np.average(scores, axis=0).astype(float)
    return avg_loss, avg_rmse


class Naive(BaseExpModel):

    def __init__(self):
        super(BaseExpModel, self).__init__()

    def _build_and_train(self, cont_seqs: List[ndarray], n_epoch: int, *args, **kwargs):
        return self

    def _eval(self, cont_seqs: List[ndarray], *args, **kwargs):
        return evaluate_naive(cont_seqs=cont_seqs)
