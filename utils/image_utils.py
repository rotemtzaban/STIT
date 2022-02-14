import numpy as np
import torch
from PIL import Image


def concat_images_horizontally(*images: Image.Image):
    assert all(image.height == images[0].height for image in images)
    total_width = sum(image.width for image in images)

    new_im = Image.new(images[0].mode, (total_width, images[0].height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width

    return new_im


def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    x = tensor.squeeze(0).permute(1, 2, 0).add(1).mul(255).div(2).squeeze()
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)
