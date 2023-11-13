from enum import Enum

from models.lstm_stateful import StatefulLSTM
from models.naive import Naive


class ModelType(Enum):
    NAIVE = Naive
    LSTM_STATEFUL = StatefulLSTM
