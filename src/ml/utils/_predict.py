import torch
from ..schemas import Result
from ._load_data import load_file


softmax = torch.nn.Softmax()

def predict_image(
    model,
    path: str,
    classes: list[str],
) -> Result:
    image = load_file(path)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probs = softmax(output)[0]
        _, label_idx = torch.max(output, 1)
        label_idx = label_idx.item()
        label = classes[label_idx]
        prob = probs[label_idx].item()

    result = Result(path, label, label_idx, prob, image)
    return result
