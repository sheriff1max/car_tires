import torch


def predict_image(
    model,
    image: torch.Tensor,
    classes: list[str],
) -> str:
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        pred = classes[classes]
    return pred


def predict_list_image(
    model,
    images: list[torch.Tensor],
    classes: list[str],
) -> list[str]:
    pred_images = []
    for image in images:
        pred = predict_image(model, image, classes)
        pred_images.append(pred)
    return pred_images
