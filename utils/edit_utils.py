import argparse
import math
import os
import pickle
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import configs.paths_config
from configs import paths_config
from training.networks import SynthesisBlock


def add_texts_to_image_vertical(texts, pivot_images):
    images_height = pivot_images.height
    images_width = pivot_images.width

    text_height = 256 + 16 - images_height % 16
    num_images = len(texts)
    image_width = images_width // num_images
    text_image = Image.new('RGB', (images_width, text_height), (255, 255, 255))
    draw = ImageDraw.Draw(text_image)
    font_size = int(math.ceil(24 * image_width / 256))

    try:
        font = ImageFont.truetype("truetype/freefont/FreeSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for i, text in enumerate(texts):
        draw.text((image_width // 2 + i * image_width, text_height // 2), text, fill='black', anchor='ms', font=font)

    out_image = Image.new('RGB', (pivot_images.width, pivot_images.height + text_image.height))
    out_image.paste(text_image, (0, 0))
    out_image.paste(pivot_images, (0, text_image.height))
    return out_image


def get_affine_layers(synthesis):
    blocks: List[SynthesisBlock] = [getattr(synthesis, f'b{res}') for res in synthesis.block_resolutions]
    affine_layers = []
    for block in blocks:
        if hasattr(block, 'conv0'):
            affine_layers.append((block.conv0.affine, True))
        affine_layers.append((block.conv1.affine, True))
        affine_layers.append((block.torgb.affine, False))
    return affine_layers


def load_stylespace_std():
    with open(paths_config.stylespace_mean_std, 'rb') as f:
        _, s_std = pickle.load(f)
        s_std = [torch.from_numpy(s).cuda() for s in s_std]

    return s_std


def to_styles(edit: torch.Tensor, affine_layers):
    idx = 0
    styles = []
    for layer, is_conv in affine_layers:
        layer_dim = layer.weight.shape[0]
        if is_conv:
            styles.append(edit[idx:idx + layer_dim].clone())
            idx += layer_dim
        else:
            styles.append(torch.zeros(layer_dim, device=edit.device, dtype=edit.dtype))

    return styles


def w_to_styles(w, affine_layers):
    w_idx = 0
    styles = []
    for affine, is_conv in affine_layers:
        styles.append(affine(w[:, w_idx]))
        if is_conv:
            w_idx += 1

    return styles


def paste_image_mask(inverse_transform, image, dst_image, mask, radius=0, sigma=0.0):
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    if radius != 0:
        mask_np = np.array(mask)
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    projected = image_masked.transform(dst_image.size, Image.PERSPECTIVE, inverse_transform,
                                       Image.BILINEAR)
    pasted_image.alpha_composite(projected)
    return pasted_image


def paste_image(inverse_transform, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, inverse_transform, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image
