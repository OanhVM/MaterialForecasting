from abc import abstractmethod, ABC
from os import makedirs
from os.path import dirname, join
from typing import List, Optional, Tuple

from keras import Model
from keras.models import load_model
from numpy import ndarray


class BaseExpModel(ABC):
    def __init__(self):
        self._model: Optional[Model] = None

    @property
    def model(self) -> Optional[Model]:
        return self._model

    @abstractmethod
    def _build_and_train(self, cont_seqs: List[ndarray], n_epoch: int, *args, **kwargs) -> Optional[Model]:
        pass

    def build_and_train(self, cont_seqs: List[ndarray], n_epoch: int, *args, **kwargs):
        self._model = self._build_and_train(cont_seqs, n_epoch, *args, **kwargs)

    def save(self, model_sub_path: str, model_dir_path: str = "models"):
        model_file_path = join(model_dir_path, model_sub_path)

        if self._model is not None:
            makedirs(dirname(model_file_path), exist_ok=True)
            return self._model.save(model_file_path)

        raise RuntimeError("Model must be either built and trained or loaded beforehand.")

    def load(self, model_file_path: str):
        self._model = load_model(model_file_path)

    @abstractmethod
    def _eval(self, cont_seqs: List[ndarray], *args, **kwargs) -> Tuple[float, float]:
        pass

    def eval(self, cont_seqs: List[ndarray], *args, **kwargs) -> Tuple[float, float]:
        if self._model is not None:
            return self._eval(self._model, cont_seqs, *args, **kwargs)

        raise RuntimeError("Model must be either built and trained or loaded beforehand.")
