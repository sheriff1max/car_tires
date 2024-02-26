import torch
from utils import (
    load_model,
    load_folder,
    load_file,
    predict_list_image,
    predict_image,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Api:
    def __init__(self, path_model: str, classes: list[str]):
        self._model = load_model(path_model)
        self._classes = classes

    def predict_folder(self, path: str) -> list[str]:
        images = load_folder(path)
        pred_images = predict_list_image(
            self._model,
            images,
            self._classes,
        )
        return pred_images

    def predict_file(self, path: str) -> str:
        image = load_file(path)
        pred = predict_image(
            self._model,
            image,
            self._classes,
        )
        return pred


if __name__ == '__main__':
    api = Api('F:\study\car_tires\src\ml\weights\resnet18_09439.pt')
