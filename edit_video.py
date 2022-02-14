import json
import os
from collections import defaultdict

import click
import imageio
import torch
import torchvision.transforms.functional
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

import models.seg_model_2
from configs import paths_config
from editings.latent_editor import LatentEditor
from utils.alignment import crop_faces_by_quads, calc_alignment_coefficients
from utils.data_utils import make_dataset
from utils.edit_utils import add_texts_to_image_vertical, paste_image_mask, paste_image
from utils.image_utils import concat_images_horizontally, tensor2pil
from utils.models_utils import load_generators


def calc_mask(inversion, segmentation_model):
    background_classes = [0, 18, 16]
    inversion_resized = torch.cat([F.interpolate(inversion, (512, 512), mode='nearest')])
    inversion_normalized = transforms.functional.normalize(inversion_resized.clip(-1, 1).add(1).div(2),
                                                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    segmentation = segmentation_model(inversion_normalized)[0].argmax(dim=1, keepdim=True)
    is_foreground = torch.stack([segmentation != cls for cls in background_classes], dim=0).all(dim=0)
    foreground_mask = F.interpolate(is_foreground.float(), (1024, 1024), mode='bilinear', align_corners=True)
    return foreground_mask


@click.command()
@click.option('-i', '--input_folder', type=str, help='Path to (unaligned) images folder', required=True)
@click.option('-o', '--output_folder', type=str, help='Path to output folder', required=True)
@click.option('-r', '--run_name', type=str, required=True)
@click.option('--use_mask/--no_mask', type=bool, default=True)
@click.option('--start_frame', type=int, default=0)
@click.option('--end_frame', type=int, default=None)
@click.option('-et', '--edit_type',
              type=click.Choice(['interfacegan', 'styleclip_global'], case_sensitive=False),
              default='interfacegan')
@click.option('--beta', default=0.2, type=float)
@click.option('--neutral_class', default='face', type=str)
@click.option('--target_class', default=None, type=str)
@click.option('-en', '--edit_name', type=str, default=None, multiple=True)
@click.option('-er', '--edit_range', type=(float, float, int), nargs=3, default=(2, 20, 10))
@click.option('--freeze_fine_layers', type=int, default=None)
@click.option('--output_frames/--videos_only', type=bool, default=False)
@click.option('--feathering_radius', type=int, default=0)
def main(**config):
    _main(**config, config=config)


def _main(input_folder, output_folder, start_frame, end_frame, run_name,
          edit_range, edit_type, edit_name, use_mask, freeze_fine_layers, neutral_class, target_class, beta,
          output_frames, feathering_radius, config):
    orig_files = make_dataset(input_folder)
    orig_files = orig_files[start_frame:end_frame]

    image_size = 1024

    segmentation_model = models.seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
    segmentation_model.load_state_dict(torch.load(paths_config.segmentation_model_path))

    gen, orig_gen, pivots, quads = load_generators(run_name)

    crops, orig_images = crop_faces_by_quads(image_size, orig_files, quads)

    inverse_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads]

    if freeze_fine_layers is not None:
        pivots_mean = torch.mean(pivots, dim=0, keepdim=True).expand_as(pivots)
        pivots = torch.cat([pivots[:, :freeze_fine_layers], pivots_mean[:, freeze_fine_layers:]], dim=1)

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'opts.json'), 'w') as f:
        json.dump(config, f)

    latent_editor = LatentEditor()
    if edit_type == 'styleclip_global':
        edits, is_style_input = latent_editor.get_styleclip_global_edits(
            pivots, neutral_class, target_class, beta, edit_range, gen, edit_name
        )
    else:
        edits, is_style_input = latent_editor.get_interfacegan_edits(pivots, edit_name, edit_range)

    for edits_list, direction, factor in edits:
        video_frames = defaultdict(list)
        for i, (orig_image, crop, quad, inverse_transform) in \
                tqdm(enumerate(zip(orig_images, crops, quads, inverse_transforms)), total=len(orig_images)):
            w_pivot = pivots[i][None]
            if is_style_input:
                w_edit = [style[i][None] for style in edits_list]
            else:
                w_edit = edits_list[i][None]

            edited_tensor = gen.synthesis.forward(w_edit, noise_mode='const', force_fp32=False,
                                                  style_input=is_style_input)
            mask = None
            if use_mask:
                crop_tensor = to_tensor(crop).mul(2).sub(1)[None].cuda()
                inversion = gen.synthesis(w_pivot, noise_mode='const', force_fp32=False)
                mask = calc_mask(crop_tensor, segmentation_model)
                mask = tensor2pil(mask.mul(2).sub(1))
            else:
                inversion = gen.synthesis(w_pivot, noise_mode='const', force_fp32=False)

            inversion = tensor2pil(inversion)
            edited_image = tensor2pil(edited_tensor)
            if mask is not None:
                inversion_projection = paste_image_mask(inverse_transform, inversion, orig_image, mask,
                                                        radius=feathering_radius)
                edit_projection = paste_image_mask(inverse_transform, edited_image, orig_image, mask,
                                                   radius=feathering_radius)
            else:
                inversion_projection = paste_image(inverse_transform, inversion, orig_image)
                edit_projection = paste_image(inverse_transform, edited_image, orig_image)
            folder_name = f'{direction}/{factor}'
            if output_frames:
                frames_dir = os.path.join(output_folder, 'frames', folder_name)
                os.makedirs(frames_dir, exist_ok=True)
                save_image(inversion_projection, os.path.join(frames_dir, f'inversion_{i:04d}.jpeg'))
                save_image(orig_image, os.path.join(frames_dir, f'source_{i:04d}.jpeg'))
                save_image(edit_projection, os.path.join(frames_dir, f'edit_{i:04d}.jpeg'))
            video_frame = concat_images_horizontally(orig_image, inversion_projection, edit_projection)
            video_frame = add_texts_to_image_vertical(['original', 'inversion', 'edit'], video_frame)
            video_frames[folder_name].append(video_frame)

        for folder_name, frames in video_frames.items():
            folder_path = os.path.join(output_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            imageio.mimwrite(os.path.join(folder_path, 'out.mp4'), frames, fps=18, output_params=['-vf', 'fps=25'])


def save_image(image, file):
    image = image.convert('RGB')
    image.save(file, quality=95)


if __name__ == '__main__':
    main()
