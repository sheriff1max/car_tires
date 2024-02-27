import torch
import numpy as np


class Result:
    """Результат предсказания."""
    def __init__(
        self,
        path: str,
        label: str,
        label_idx: int,
        prob: float,
        image: torch.Tensor,
    ):
        self._path = path
        self._label = label
        self._label_idx = label_idx
        self._prob = prob
        self._image = self.__tensor_to_numpy(image)

    def __tensor_to_numpy(self, image: torch.Tensor) -> np.ndarray:
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        return image

    def get_path(self) -> str:
        return self._path

    def get_label(self) -> str:
        return self._label

    def get_label_idx(self) -> int:
        return self._label_idx

    def get_prob(self) -> float:
        return round(self._prob, 3) * 100

    def get_image(self) -> np.ndarray:
        return self._image
