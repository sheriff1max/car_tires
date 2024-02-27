import torch
from .utils import (
    load_model,
    predict_image,
)
from .schemas import Result


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Api:
    def __init__(self, path_model: str, classes: list[str]):
        self._model = load_model(path_model)
        self._classes = classes

    def predict(self, paths: list[str]) -> list[Result]:
        list_results = []
        for path in paths:
            result = predict_image(
                self._model,
                path,
                self._classes,
            )
            list_results.append(result)
        return list_results


if __name__ == '__main__':
    api = Api('F:/study/car_tires/src/ml/weights/resnet18_09439.pt', [0, 1])
