import copy
import io
import json
import os
from collections import defaultdict

import click
import imageio
import torch
import torchvision.transforms.functional
from PIL import Image, ImageChops
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm, trange

import models.seg_model_2
from configs import hyperparameters, paths_config
from edit_video import save_image
from editings.latent_editor import LatentEditor
from utils.alignment import crop_faces_by_quads, calc_alignment_coefficients
from utils.data_utils import make_dataset
from utils.edit_utils import add_texts_to_image_vertical, paste_image, paste_image_mask
from utils.image_utils import concat_images_horizontally, tensor2pil
from utils.models_utils import load_generators
from utils.morphology import dilation

debug = False


def create_masks(border_pixels, mask, inner_dilation=0, outer_dilation=0, whole_image_border=False):
    image_size = mask.shape[2]
    grid = torch.cartesian_prod(torch.arange(image_size), torch.arange(image_size)).view(image_size, image_size,
                                                                                         2).cuda()
    image_border_mask = logical_or_reduce(
        grid[:, :, 0] < border_pixels,
        grid[:, :, 1] < border_pixels,
        grid[:, :, 0] >= image_size - border_pixels,
        grid[:, :, 1] >= image_size - border_pixels
    )[None, None].expand_as(mask)

    temp = mask
    if inner_dilation != 0:
        temp = dilation(temp, torch.ones(2 * inner_dilation + 1, 2 * inner_dilation + 1, device=mask.device),
                        engine='convolution')

    border_mask = torch.min(image_border_mask, temp)
    full_mask = dilation(temp, torch.ones(2 * outer_dilation + 1, 2 * outer_dilation + 1, device=mask.device),
                         engine='convolution')
    if whole_image_border:
        border_mask_2 = 1 - temp
    else:
        border_mask_2 = full_mask - temp
    border_mask = torch.maximum(border_mask, border_mask_2)

    border_mask = border_mask.clip(0, 1)
    content_mask = (mask - border_mask).clip(0, 1)
    return content_mask, border_mask, full_mask


def calc_masks(inversion, segmentation_model, border_pixels, inner_mask_dilation, outer_mask_dilation,
               whole_image_border):
    background_classes = [0, 18, 16]
    inversion_resized = torch.cat([F.interpolate(inversion, (512, 512), mode='nearest')])
    inversion_normalized = transforms.functional.normalize(inversion_resized.clip(-1, 1).add(1).div(2),
                                                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    segmentation = segmentation_model(inversion_normalized)[0].argmax(dim=1, keepdim=True)
    is_foreground = logical_and_reduce(*[segmentation != cls for cls in background_classes])
    foreground_mask = is_foreground.float()
    content_mask, border_mask, full_mask = create_masks(border_pixels // 2, foreground_mask, inner_mask_dilation // 2,
                                                        outer_mask_dilation // 2, whole_image_border)
    content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=True)
    border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=True)
    full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=True)
    return content_mask, border_mask, full_mask


@click.command()
@click.option('-i', '--input_folder', type=str, help='Path to (unaligned) images folder', required=True)
@click.option('-o', '--output_folder', type=str, help='Path to output folder', required=True)
@click.option('-r', '--run_name', type=str, required=True)
@click.option('--start_frame', type=int, default=0)
@click.option('--end_frame', type=int, default=None)
@click.option('--inner_mask_dilation', type=int, default=0)
@click.option('--outer_mask_dilation', type=int, default=50)
@click.option('-et', '--edit_type',
              type=click.Choice(['styleclip_global', 'interfacegan'], case_sensitive=False),
              default='interfacegan')
@click.option('--whole_image_border', is_flag=True, type=bool)
@click.option('--beta', default=0.2, type=float)
@click.option('--neutral_class', default='face', type=str)
@click.option('--target_class', default=None, type=str)
@click.option('-en', '--edit_name', type=str, default=None, multiple=True)
@click.option('-er', '--edit_range', type=(float, float, int), nargs=3, default=(2, 20, 10))
@click.option('--freeze_fine_layers', type=int, default=None)
@click.option('--l2/--l1', type=bool, default=True)
@click.option('--output_frames', type=bool, is_flag=True, default=False)
@click.option('--num_steps', type=int, default=100)
@click.option('--content_loss_lambda', type=float, default=0.01)
@click.option('--border_loss_threshold', type=float, default=0.0)
def main(**config):
    _main(**config, config=config)


def _main(input_folder, output_folder, start_frame, end_frame, run_name,
          edit_range, edit_type, edit_name, inner_mask_dilation,
          outer_mask_dilation, whole_image_border,
          freeze_fine_layers, l2, output_frames, num_steps, neutral_class, target_class,
          beta, config, content_loss_lambda, border_loss_threshold):
    orig_files = make_dataset(input_folder)
    orig_files = orig_files[start_frame:end_frame]
    segmentation_model = models.seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
    segmentation_model.load_state_dict(torch.load(paths_config.segmentation_model_path))

    gen, orig_gen, pivots, quads = load_generators(run_name)
    image_size = 1024

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
            w_interp = pivots[i][None]
            if is_style_input:
                w_edit_interp = [style[i][None] for style in edits_list]
            else:
                w_edit_interp = edits_list[i][None]

            edited_tensor = gen.synthesis.forward(w_edit_interp, style_input=is_style_input, noise_mode='const',
                                                  force_fp32=True)

            inversion = gen.synthesis(w_interp, noise_mode='const', force_fp32=True)
            border_pixels = outer_mask_dilation

            crop_tensor = to_tensor(crop)[None].mul(2).sub(1).cuda()
            content_mask, border_mask, full_mask = calc_masks(crop_tensor, segmentation_model, border_pixels,
                                                              inner_mask_dilation, outer_mask_dilation,
                                                              whole_image_border)
            inversion = tensor2pil(inversion)
            inversion_projection = paste_image(inverse_transform, inversion, orig_image)

            optimized_tensor = optimize_border(gen, crop_tensor, edited_tensor,
                                               w_edit_interp, border_mask=border_mask, content_mask=content_mask,
                                               optimize_generator=True, num_steps=num_steps, loss_l2=l2,
                                               is_style_input=is_style_input, content_loss_lambda=content_loss_lambda,
                                               border_loss_threshold=border_loss_threshold)

            video_frames[f'optimized_edits/{direction}/{factor}'].append(
                tensor2pil(optimized_tensor)
            )

            optimized_image = tensor2pil(optimized_tensor)
            edited_image = tensor2pil(edited_tensor)

            full_mask_image = tensor2pil(full_mask.mul(2).sub(1))

            edit_projection = paste_image_mask(inverse_transform, edited_image, orig_image, full_mask_image, radius=0)
            optimized_projection = paste_image_mask(inverse_transform, optimized_image, orig_image, full_mask_image,
                                                    radius=0)

            optimized_projection_feathered = paste_image_mask(inverse_transform, optimized_image, orig_image,
                                                              full_mask_image,
                                                              radius=outer_mask_dilation // 2)

            folder_name = f'{direction}/{factor}'

            video_frame = concat_images_horizontally(orig_image, edit_projection, optimized_projection)
            video_frame = add_texts_to_image_vertical(['original', 'mask', 'stitching tuning'], video_frame)

            video_frames[folder_name].append(video_frame)

            video_frame = concat_images_horizontally(orig_image, edit_projection, optimized_projection_feathered)
            video_frame = add_texts_to_image_vertical(['original', 'mask', 'stitching tuning'], video_frame)

            video_frames[f'{folder_name}/feathering'].append(video_frame)

            if output_frames:
                frames_dir = os.path.join(output_folder, 'frames', folder_name)
                os.makedirs(frames_dir, exist_ok=True)
                save_image(inversion_projection, os.path.join(frames_dir, f'pti_{i:04d}.jpeg'))
                save_image(orig_image, os.path.join(frames_dir, f'source_{i:04d}.jpeg'))
                save_image(edit_projection, os.path.join(frames_dir, f'edit_{i:04d}.jpeg'))
                save_image(optimized_projection, os.path.join(frames_dir, f'optimized_{i:04d}.jpeg'))
                save_image(optimized_projection_feathered,
                           os.path.join(frames_dir, f'optimized_feathering_{i:04d}.jpeg'))

            if debug:
                border_mask_image = tensor2pil(border_mask.mul(2).sub(1))
                inner_mask_image = tensor2pil(content_mask.mul(2).sub(1))

                video_frames[f'masks/{direction}/{factor}'].append(
                    concat_images_horizontally(
                        border_mask_image,
                        inner_mask_image,
                        full_mask_image
                    ))

                inner_image = optimized_projection.copy()
                outer_mask_image = ImageChops.invert(inner_mask_image)

                full_mask_projection = full_mask_image.transform(orig_image.size, Image.PERSPECTIVE, inverse_transform,
                                                                 Image.BILINEAR)
                outer_mask_projection = outer_mask_image.transform(orig_image.size, Image.PERSPECTIVE,
                                                                   inverse_transform,
                                                                   Image.BILINEAR)
                inner_image.putalpha(full_mask_projection)
                outer_image = optimized_projection.copy()
                outer_image.putalpha(outer_mask_projection)

                masked = concat_images_horizontally(inner_image, outer_image)
                video_frames[f'masked/{folder_name}'].append(masked)

                frame_data = create_dump_file(border_mask_image, crop, full_mask_image, inner_mask_image,
                                              inverse_transform, optimized_image, orig_image, quad, edited_image)
                os.makedirs(os.path.join(output_folder, 'dumps', folder_name), exist_ok=True)
                torch.save(frame_data, os.path.join(output_folder, 'dumps', folder_name, f'{i}.pt'))

        for folder_name, frames in video_frames.items():
            folder_path = os.path.join(output_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            imageio.mimwrite(os.path.join(folder_path, 'out.mp4'), frames, fps=25, output_params=['-vf', 'fps=25'])


def create_dump_file(border_mask_image, crop, full_mask_image, inner_mask_image, inverse_transform, optimized_image,
                     orig_image, quad, edited_image):
    def compress_image(img: Image.Image):
        output = io.BytesIO()
        img.save(output, format='png', optimize=False)
        return output.getvalue()

    frame_data = {'inverse_transform': inverse_transform, 'orig_image': compress_image(orig_image), 'quad': quad,
                  'optimized_image': compress_image(optimized_image), 'crop': compress_image(crop),
                  'inner_mask_image': compress_image(inner_mask_image),
                  'full_mask_image': compress_image(full_mask_image),
                  'border_mask_image': compress_image(border_mask_image),
                  'edited_image': compress_image(edited_image)}
    return frame_data


def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)


def logical_and_reduce(*tensors):
    return torch.stack(tensors, dim=0).all(dim=0)


def optimize_border(G: torch.nn.Module, border_image, content_image, w: torch.Tensor, border_mask, content_mask,
                    optimize_generator=False, optimize_wplus=False, num_steps=100, loss_l2=True, is_style_input=False,
                    content_loss_lambda=0.01, border_loss_threshold=0.0):
    assert optimize_generator or optimize_wplus

    G = copy.deepcopy(G).train(optimize_generator).requires_grad_(optimize_generator).float()
    if not is_style_input:
        latent = torch.nn.Parameter(w, requires_grad=optimize_wplus)
    else:
        latent = w
        assert not optimize_wplus
    parameters = []
    if optimize_generator:
        parameters += list(G.parameters())
    if optimize_wplus:
        parameters += [latent]

    optimizer = torch.optim.Adam(parameters, hyperparameters.stitching_tuning_lr)
    for step in trange(num_steps, leave=False):
        generated_image = G.synthesis(latent, style_input=is_style_input, noise_mode='const', force_fp32=True)

        border_loss = masked_l2(generated_image, border_image, border_mask, loss_l2)
        loss = border_loss + content_loss_lambda * masked_l2(generated_image, content_image, content_mask, loss_l2)
        if border_loss < border_loss_threshold:
            break
        optimizer.zero_grad()
        # wandb.log({f'border_loss_{frame_id}': border_loss.item()})
        loss.backward()
        optimizer.step()

    generated_image = G.synthesis(latent, style_input=is_style_input, noise_mode='const', force_fp32=True)
    del G
    return generated_image.detach()


def masked_l2(input, target, mask, loss_l2):
    loss = torch.nn.MSELoss if loss_l2 else torch.nn.L1Loss
    criterion = loss(reduction='none')
    masked_input = input * mask
    masked_target = target * mask
    error = criterion(masked_input, masked_target)
    return error.sum() / mask.sum()


if __name__ == '__main__':
    main()
