# STIT - Stitch it in Time

[![arXiv](https://img.shields.io/badge/arXiv-2201.08361-b31b1b.svg)](https://arxiv.org/abs/2201.08361)
[![CGP](https://img.shields.io/badge/CGP-Paper%20Summary-blueviolet)](https://www.casualganpapers.com/hiqh_quality_video_editing_stylegan_inversion/Stitch-It-In-Time-explained.html) [![WAI](https://img.shields.io/badge/WhatsAI-Paper%20Summary-blueviolet)](https://www.louisbouchard.ai/stitch-it-in-time/)



[[Project Page](https://stitch-time.github.io/)]
> **Stitch it in Time: GAN-Based Facial Editing of Real Videos**<br>
> Rotem Tzaban, Ron Mokady, Rinon Gal, Amit Bermano, Daniel Cohen-Or <br>

>**Abstract**: <br>
> The ability of Generative Adversarial Networks to encode rich semantics within their latent space has been widely adopted for facial image editing. However, replicating their success with videos has proven challenging. Sets of high-quality facial videos are lacking, and working with videos introduces a fundamental barrier to overcome - temporal coherency. We propose that this barrier is largely artificial. The source video is already temporally coherent, and deviations from this state arise in part due to careless treatment of individual components in the editing pipeline. We leverage the natural alignment of StyleGAN and the tendency of neural networks to learn low frequency functions, and demonstrate that they provide a strongly consistent prior. We draw on these insights and propose a framework for semantic editing of faces in videos, demonstrating significant improvements over the current state-of-the-art. Our method produces meaningful face manipulations, maintains a higher degree of temporal consistency, and can be applied to challenging, high quality, talking head videos which current methods struggle with.

## Requirements

- Pytorch(tested with 1.10, should work with 1.8/1.9 as well) + torchvision, Follow <https://pytorch.org/> 
  for pytorch installation instructions.
- CUDA toolkit 11.0 or later. Make sure you have the requirements listed here: 
  <https://github.com/NVlabs/stylegan2-ada-pytorch#requirements>
- For the rest of the requirements, use: ```pip install -r requirements.txt```
- To perform StyleCLIP edits, install clip with:

```
pip install git+https://github.com/openai/CLIP.git
```

### Pretrained models

In order to use this project you need to download pretrained models from the following 
[Link](https://drive.google.com/file/d/1cDvUHPTZQAEWvfiK9C0nDuI9C3Qdgbbp/view?usp=sharing).

Unzip it inside the project's main directory.

You can use the download_models.sh script (requires installing gdown with `pip install gdown`)

Alternatively, you can unzip the models to a location of your choice and update `configs/path_config.py` accordingly.


## Splitting videos into frames
Our code expects videos in the form of a directory with individual frame images.
To produce such a directory from an existing video, we recommend using ffmpeg:
```
ffmpeg -i "video.mp4" "video_frames/out%04d.png"
```

## Example Videos
The videos used to produce our results can be downloaded from the following 
[Link](https://drive.google.com/file/d/1ZzpUJSq3Z8ZU8MKfvZ8w8MayMDxLBEy1/view?usp=sharing).


## Inversion
To invert a video run:

```
python train.py --input_folder /path/to/images_dir \
 --output_folder /path/to/experiment_dir \
 --run_name RUN_NAME \
 --num_pti_steps NUM_STEPS
```
This includes aligning, cropping, e4e encoding and PTI

For example:

```
python train.py --input_folder /data/obama \
 --output_folder training_results/obama \
 --run_name obama \
 --num_pti_steps 80
```

Weights and biases logging is disabled by default. to enable, add --use_wandb

## Naive Editing 
To run edits without stitching tuning:
```
python edit_video.py --input_folder /path/to/images_dir \
 --output_folder /path/to/experiment_dir \
 --run_name RUN_NAME \
 --edit_name EDIT_NAME \
 --edit_range EDIT_RANGE
```

edit_range determines the strength of the edits applied.
It should be in the format RANGE_START RANGE_END RANGE_STEPS.   
for example, if we use `--edit_range 1 5 2`,
we will apply edits with strength 1, 3 and 5.


For young Obama use:

```
python edit_video.py --input_folder /data/obama \
 --output_folder edits/obama/ \
 --run_name obama \
 --edit_name age \
 --edit_range -8 -8 1 \
```

## Editing + Stitching Tuning

To run edits with stitching tuning:
```
python edit_video_stitching_tuning.py --input_folder /path/to/images_dir \
 --output_folder /path/to/experiment_dir \
 --run_name RUN_NAME \
 --edit_name EDIT_NAME \
 --edit_range EDIT_RANGE \
 --outer_mask_dilation MASK_DILATION
```

We support early breaking the stitching tuning process, when the loss reaches a specified threshold.  
This enables us to perform more iterations for difficult frames while maintaining a reasonable running time.  
To use this feature, add ```--border_loss_threshold THRESHOLD``` to the command(Shown in the Jim and Kamala Harris examples below).  
For videos with a simple background to reconstruct (e.g Obama, Jim, Emma Watson, Kamala Harris), we use ```THRESHOLD=0.005```.  
For videos where a more exact reconstruction of the background is required (e.g Michael Scott), we use ```THRESHOLD=0.002```.  
Early breaking is disabled by default.

For young Obama use:

```
python edit_video_stitching_tuning.py --input_folder /data/obama \
 --output_folder edits/obama/ \
 --run_name obama \
 --edit_name age \
 --edit_range -8 -8 1 \
 --outer_mask_dilation 50
```

For gender editing on Obama use:

```
python edit_video_stitching_tuning.py --input_folder /data/obama \
 --output_folder edits/obama/ \
 --run_name obama \
 --edit_name gender \
 --edit_range -6 -6 1 \
 --outer_mask_dilation 50
```

For young Emma Watson use:

```
python edit_video_stitching_tuning.py --input_folder /data/emma_watson \
 --output_folder edits/emma_watson/ \
 --run_name emma_watson \
 --edit_name age \
 --edit_range -8 -8 1 \
 --outer_mask_dilation 50
```
For smile removal on Emma Watson use:
```
python edit_video_stitching_tuning.py --input_folder /data/emma_watson \
 --output_folder edits/emma_watson/ \
 --run_name emma_watson \
 --edit_name smile \
 --edit_range -3 -3 1 \
 --outer_mask_dilation 50
```

For Emma Watson lipstick editing use: (done with styleclip global direction)

```
python edit_video_stitching_tuning.py --input_folder /data/emma_watson \
 --output_folder edits/emma_watson/ \
 --run_name emma_watson \
 --edit_type styleclip_global \
 --edit_name lipstick \
 --neutral_class "Face" \
 --target_class "Face with lipstick" \
 --beta 0.2 \
 --edit_range 10 10 1 \
 --outer_mask_dilation 50
```

For Old + Young Jim use (with early breaking):

```
python edit_video_stitching_tuning.py --input_folder datasets/jim/ \
 --output_folder edits/jim \
 --run_name jim \
 --edit_name age \
 --edit_range -8 8 2 \
 --outer_mask_dilation 50 \
 --border_loss_threshold 0.005
 ```

For smiling Kamala Harris:
```
python edit_video_stitching_tuning.py \
 --input_folder datasets/kamala/ \
 --output_folder edits/kamala \
 --run_name kamala \
 --edit_name smile \
 --edit_range 2 2 1 \
 --outer_mask_dilation 50 \
 --border_loss_threshold 0.005
```

## Example Results


With stitching tuning:
<video src="https://user-images.githubusercontent.com/24721699/153860260-a431379e-ebab-4777-844d-4900a448cf85.mp4" controls width=512></video>

Without stitching tuning:
<video src="https://user-images.githubusercontent.com/24721699/153860400-ba792b37-f8fc-431e-93c1-8751d7b2ea0e.mp4" controls width=512></video>

Gender editing:
<video src="https://user-images.githubusercontent.com/24721699/153861476-08e9ff12-0fb7-4f4f-9703-4611e5b78fca.mp4" controls width=512></video>

Young Emma Watson:

<video src="https://user-images.githubusercontent.com/24721699/153863426-1b2485f6-83ba-404b-8d74-f34675e02036.mp4" controls width=512></video>

Emma Watson with lipstick:

<video src="https://user-images.githubusercontent.com/24721699/153876897-745dcb21-cfcd-44fc-8644-7302d2b41b81.mp4" controls width=512></video>

Emma Watson smile removal:
<video src="https://user-images.githubusercontent.com/24721699/153864167-d5d428c8-706e-4925-a26a-8791812de94f.mp4" controls width=512></video>

Old Jim:

<video src="https://user-images.githubusercontent.com/24721699/153885229-a17561fb-c6d3-4cff-9f8b-c7bad40a5654.mp4" controls width=512></video>

Young Jim:

<video src="https://user-images.githubusercontent.com/24721699/153885371-0c47febf-7e58-4d6f-bf72-a0e1b1a47a60.mp4" controls width=512></video>

Smiling Kamala Harris:

<video src="https://user-images.githubusercontent.com/24721699/153889707-ed6c872a-941c-4e43-b31b-b934546d6b24.mp4" controls width=512></video>


## Out of domain video editing (Animations)

For editing out of domain videos, Some different parameters are required while training.
First, dlib's face detector doesn't detect all animated faces, so we use a different face detector provided by the [face_alignment package](https://github.com/1adrianb/face-alignment).
Second, we reduce the smoothing of the alignment parameters with ```--center_sigma 0.0```
Third, OOD videos require more training steps, as they are more difficult to invert.

To train, we use:

```
python train.py --input_folder datasets/ood_spiderverse_gwen/ \
 --output_folder training_results/ood \
 --run_name ood \
 --num_pti_steps 240 \
 --use_fa \
 --center_sigma 0.0
```

Afterwards, editing is performed the same way:

```
python edit_video.py --input_folder datasets/ood_spiderverse_gwen/ \
 --output_folder edits/ood \
 --run_name ood \
 --edit_name smile \
 --edit_range 2 2 1
```

<video src="https://user-images.githubusercontent.com/24721699/153874953-1b840a07-4b25-4866-b0bf-e19fabffa989.mp4" controls width=512></video>

```
python edit_video.py --input_folder datasets/ood_spiderverse_gwen/ \
 --output_folder edits/ood \
 --run_name ood \
 --edit_type styleclip_global \
 --edit_range 10 10 1 \
 --edit_name lipstick \
 --target_class 'Face with lipstick'
```

<video src="https://user-images.githubusercontent.com/24721699/153875508-7eecfa4b-3529-40ef-83ae-80a02f3b5ec4.mp4" controls width=512></video>

## Credits:
**StyleGAN2-ada model and implementation:**  
https://github.com/NVlabs/stylegan2-ada-pytorch
Copyright © 2021, NVIDIA Corporation.  
Nvidia Source Code License https://nvlabs.github.io/stylegan2-ada-pytorch/license.html

**PTI implementation**:   
https://github.com/danielroich/PTI  
Copyright (c) 2021 Daniel Roich  
License (MIT) https://github.com/danielroich/PTI/blob/main/LICENSE  

**LPIPS model and implementation:**  
https://github.com/richzhang/PerceptualSimilarity  
Copyright (c) 2020, Sou Uchida  
License (BSD 2-Clause) https://github.com/richzhang/PerceptualSimilarity/blob/master/LICENSE

**e4e model and implementation:**   
https://github.com/omertov/encoder4editing
Copyright (c) 2021 omertov  
License (MIT) https://github.com/omertov/encoder4editing/blob/main/LICENSE

**StyleCLIP model and implementation:**   
https://github.com/orpatashnik/StyleCLIP
Copyright (c) 2021 orpatashnik  
License (MIT) https://github.com/orpatashnik/StyleCLIP/blob/main/LICENSE

**StyleGAN2 Distillation for Feed-forward Image Manipulation - for editing directions:**  
https://github.com/EvgenyKashin/stylegan2-distillation  
Copyright (c) 2019, Yandex LLC  
License (Creative Commons NonCommercial) https://github.com/EvgenyKashin/stylegan2-distillation/blob/master/LICENSE  

**face-alignment Library:**  
https://github.com/1adrianb/face-alignment  
Copyright (c) 2017, Adrian Bulat  
License (BSD 3-Clause License) https://github.com/1adrianb/face-alignment/blob/master/LICENSE  

**face-parsing.PyTorch:**  
https://github.com/zllrunning/face-parsing.PyTorch  
Copyright (c) 2019 zll  
License (MIT) https://github.com/zllrunning/face-parsing.PyTorch/blob/master/LICENSE  


## Citation

If you make use of our work, please cite our paper:

```
@misc{tzaban2022stitch,
      title={Stitch it in Time: GAN-Based Facial Editing of Real Videos},
      author={Rotem Tzaban and Ron Mokady and Rinon Gal and Amit H. Bermano and Daniel Cohen-Or},
      year={2022},
      eprint={2201.08361},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

