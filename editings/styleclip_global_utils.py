import numpy as np
import os
import click
import torch
from tqdm import tqdm

from configs import paths_config
try:
    import clip
except ImportError:
    print('Warning: clip is not installed, styleclip edits will not work')
    pass

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def get_direction(neutral_class, target_class, beta, model=None, di=None):
    if di is None:
        di = torch.from_numpy(np.load(paths_config.styleclip_fs3)).cuda()

    if model is None:
        model, _ = clip.load("ViT-B/32")

    class_names = [neutral_class, target_class]
    class_weights = zeroshot_classifier(model, class_names, imagenet_templates)

    dt = class_weights[:, 1] - class_weights[:, 0]
    dt = dt / dt.norm()
    relevance = di @ dt
    mask = relevance.abs() > beta
    direction = relevance * mask
    direction_max = direction.abs().max()
    if direction_max > 0:
        direction = direction / direction_max
    else:
        raise ValueError(f'Beta value {beta} is too high for mapping from {neutral_class} to {target_class},'
                         f' try setting it to a lower value')
    return direction


@click.command()
@click.option('-n', '--neutral_class', type=str, required=True)
@click.option('-t', '--target_class', type=str, required=True)
@click.option('-o', '--output_path', type=str, required=True)
def main(neutral_class, target_class, output_path):
    di = torch.from_numpy(np.load(paths_config.styleclip_fs3)).cuda()
    model, preprocess = clip.load("ViT-B/32")
    os.makedirs(output_path, exist_ok=True)
    for beta in [0.01, 0.03, 0.05, 0.1, 0.2]:
        try:
            direction = get_direction(neutral_class, target_class, beta, model, di).cpu().numpy()
            np.save(os.path.join(output_path, f'{beta}.npy'), direction)
        except ValueError:
            pass


if __name__ == '__main__':
    main()
