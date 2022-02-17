import json

from tqdm import tqdm

from utils.models_utils import save_tuned_G

from datasets.image_list_dataset import ImageListDataset
from training.coaches.coach import Coach
from utils.data_utils import make_dataset
import os

import click
import numpy as np
import torch
import wandb
from PIL import Image
from torchvision import transforms

from configs import paths_config, global_config, hyperparameters
from utils.alignment import crop_faces, calc_alignment_coefficients


def save_image(image: Image.Image, output_folder, image_name, image_index, ext='jpg'):
    if ext == 'jpeg' or ext == 'jpg':
        image = image.convert('RGB')
    folder = os.path.join(output_folder, image_name)
    os.makedirs(folder, exist_ok=True)
    image.save(os.path.join(folder, f'{image_index}.{ext}'))


def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image


def to_pil_image(tensor: torch.Tensor) -> Image.Image:
    x = (tensor[0].permute(1, 2, 0) + 1) * 255 / 2
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


@click.command()
@click.option('-i', '--input_folder', type=str, help='Path to (unaligned) images folder', required=True)
@click.option('-o', '--output_folder', type=str, help='Path to output folder', required=True)
@click.option('--start_frame', type=int, default=0)
@click.option('--end_frame', type=int, default=None)
@click.option('-r', '--run_name', type=str, required=True)
@click.option('--use_fa/--use_dlib', default=False, type=bool)
@click.option('--scale', default=1.0, type=float)
@click.option('--num_pti_steps', default=300, type=int)
@click.option('--l2_lambda', type=float, default=10.0)
@click.option('--center_sigma', type=float, default=1.0)
@click.option('--xy_sigma', type=float, default=3.0)
@click.option('--pti_learning_rate', type=float, default=3e-5)
@click.option('--use_locality_reg/--no_locality_reg', type=bool, default=False)
@click.option('--use_wandb/--no_wandb', type=bool, default=False)
@click.option('--pti_adam_beta1', type=float, default=0.9)
def main(**config):
    _main(**config, config=config)


def _main(input_folder, output_folder, start_frame, end_frame, run_name,
          scale, num_pti_steps, l2_lambda, center_sigma, xy_sigma,
          use_fa, use_locality_reg, use_wandb, config, pti_learning_rate, pti_adam_beta1):
    global_config.run_name = run_name
    hyperparameters.max_pti_steps = num_pti_steps
    hyperparameters.pt_l2_lambda = l2_lambda
    hyperparameters.use_locality_regularization = use_locality_reg
    hyperparameters.pti_learning_rate = pti_learning_rate
    hyperparameters.pti_adam_beta1 = pti_adam_beta1
    if use_wandb:
        wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name, config=config)
    files = make_dataset(input_folder)
    files = files[start_frame:end_frame]
    print(f'Number of images: {len(files)}')
    image_size = 1024
    print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, files, scale,
                                           center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    print('Aligning completed')


    ds = ImageListDataset(crops, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    coach = Coach(ds, use_wandb)

    ws = coach.train()

    save_tuned_G(coach.G, ws, quads, global_config.run_name)

    inverse_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads]

    gen = coach.G.requires_grad_(False).eval()

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'opts.json'), 'w') as f:
        json.dump(config, f)

    for i, (coeffs, crop, orig_image, w) in tqdm(
            enumerate(zip(inverse_transforms, crops, orig_images, ws)), total=len(ws)):
        w = w[None]
        pasted_image = paste_image(coeffs, crop, orig_image)

        save_image(pasted_image, output_folder, 'projected', i)
        with torch.no_grad():
            inversion = gen.synthesis(w, noise_mode='const', force_fp32=True)
            pivot = coach.original_G.synthesis(w, noise_mode='const', force_fp32=True)
            inversion = to_pil_image(inversion)
            pivot = to_pil_image(pivot)

        save_image(pivot, output_folder, 'pivot', i)
        save_image(inversion, output_folder, 'inversion', i)
        save_image(paste_image(coeffs, pivot, orig_image), output_folder, 'pivot_projected', i)
        save_image(paste_image(coeffs, inversion, orig_image), output_folder, 'inversion_projected', i)


if __name__ == '__main__':
    main()
