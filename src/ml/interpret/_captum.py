from captum.attr import IntegratedGradients, GuidedGradCam
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import numpy as np
import torch


default_cmap = LinearSegmentedColormap.from_list(
    'custom blue', 
    [(0, '#ffffff'), (0.25, '#000000'), (1, '#000000')],
    N=256
)


def interpret_integrated_gradients(
    model,
    image: torch.Tensor,
    image_show: np.ndarray,
    label_idx: int,
):
    image = image.unsqueeze(0)
    integrated_gradients = IntegratedGradients(model)
    attributions = integrated_gradients.attribute(image, target=label_idx)

    tuple_figures = __visualize_image_attr_multiple(attributions, image_show)
    image = __figure_to_numpy(tuple_figures[0])
    return image


def interpret_grad_cam(
    model,
    image: torch.Tensor,
    image_show: np.ndarray,
    label_idx: int,
):
    target_layer = __find_last_conv_layer(model)

    image = image.unsqueeze(0)
    guidedGradCam = GuidedGradCam(model, target_layer)
    attributions = guidedGradCam.attribute(image, target=label_idx)

    tuple_figures = __visualize_image_attr_multiple(attributions, image_show)
    image = __figure_to_numpy(tuple_figures[0])
    return image


def __figure_to_numpy(figure: Figure) -> np.ndarray:
    canvas = figure.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(height, width, 3)
    return image


def __visualize_image_attr_multiple(
    attributions: torch.Tensor,
    image_show: np.ndarray
) -> tuple[Figure]:
    tuple_figures = viz.visualize_image_attr_multiple(
        np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        image_show,
        methods=["original_image", "heat_map"],
        signs=['all', 'positive'],
        cmap=default_cmap,
        fig_size=(8, 3),
        show_colorbar=True
    )
    return tuple_figures


def __find_last_conv_layer(model: torch.nn.Module) -> tuple:
    blocks = []
    good_block_idx = None
    for _, layer in model.named_modules():

        if len(list(layer.children())) > 0:
            blocks.append(layer)

        elif isinstance(layer, torch.nn.Conv2d):
            good_block_idx = len(blocks) - 1
    return blocks[good_block_idx]
