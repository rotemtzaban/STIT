import os
import os.path
from collections import defaultdict

import numpy as np
import torch
import wandb
from lpips import LPIPS
from torchvision import transforms
from tqdm import tqdm, trange

from configs import global_config, paths_config, hyperparameters
from criteria import l2_loss
from criteria.localitly_regularizer import SpaceRegularizer
from training.projectors import w_projector
from utils.log_utils import log_image_from_w, log_images_from_w
from utils.models_utils import load_old_G, initialize_e4e_wplus, save_tuned_G


class Coach:
    def __init__(self, dataset, use_wandb):

        self.use_wandb = use_wandb
        self.dataset = dataset

        if hyperparameters.first_inv_type == 'e4e':
            self.e4e_inversion_net = initialize_e4e_wplus()

        self.e4e_image_transform = transforms.Resize((256, 256))

        # Initialize loss
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()

        self.restart_training()

        # Initialize checkpoint dir
        self.checkpoint_dir = paths_config.checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def restart_training(self):

        # Initialize networks
        self.G = load_old_G()
        self.G.requires_grad_(True)

        self.original_G = load_old_G()

        self.space_regularizer = SpaceRegularizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    def get_inversion(self, image_name, image):
        if hyperparameters.first_inv_type == 'e4e':
            w_pivot = self.get_e4e_inversion(image)
        else:
            id_image = torch.squeeze((image.to(global_config.device) + 1) / 2) * 255
            w_pivot = w_projector.project(self.G, id_image, device=torch.device(global_config.device),
                                          w_avg_samples=600,
                                          num_steps=hyperparameters.first_inv_steps, w_name=image_name,
                                          use_wandb=self.use_wandb)

        w_pivot = w_pivot.to(global_config.device)
        return w_pivot

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), betas=(hyperparameters.pti_adam_beta1, 0.999),
                                     lr=hyperparameters.pti_learning_rate)

        return optimizer

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            if self.use_wandb:
                wandb.log({f'losses/MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, commit=False)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'losses/LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, commit=False)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regularizer.space_regularizer_loss(new_G, w_batch, log_name,
                                                                                 use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips

    def forward(self, w):
        generated_images = self.G.synthesis(w, noise_mode='const', force_fp32=True)

        return generated_images

    def get_e4e_inversion(self, image):
        new_image = self.e4e_image_transform(image).to(global_config.device)
        _, w = self.e4e_inversion_net(new_image.unsqueeze(0), randomize_noise=False, return_latents=True, resize=False,
                                      input_code=False)
        if self.use_wandb:
            log_image_from_w(w, self.G, 'First e4e inversion')
        return w

    def train(self):
        self.G.synthesis.train()
        self.G.mapping.train()

        use_ball_holder = True
        w_pivots = []
        images = []

        print('Calculating initial inversions')
        for fname, image in tqdm(self.dataset):
            image_name = fname
            w_pivot = self.get_inversion(image_name, image)
            w_pivots.append(w_pivot)
            images.append((image_name, image))

        self.G = self.G.to(global_config.device)

        print('Fine tuning generator')

        for step in trange(hyperparameters.max_pti_steps):
            step_loss_dict = defaultdict(list)
            t = (step + 1) / hyperparameters.max_pti_steps

            if hyperparameters.use_lr_ramp:
                lr_ramp = min(1.0, (1.0 - t) / hyperparameters.lr_rampdown_length)
                lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
                lr_ramp = lr_ramp * min(1.0, t / hyperparameters.lr_rampup_length)
                lr = hyperparameters.pti_learning_rate * lr_ramp
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            for data, w_pivot in zip(images, w_pivots):
                image_name, image = data
                image = image.unsqueeze(0)

                real_images_batch = image.to(global_config.device)

                generated_images = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)

                step_loss_dict['loss'].append(loss.item())
                step_loss_dict['l2_loss'].append(l2_loss_val.item())
                step_loss_dict['loss_lpips'].append(loss_lpips.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1

            log_dict = {}
            for key, losses in step_loss_dict.items():
                loss_mean = sum(losses) / len(losses)
                loss_max = max(losses)
                log_dict[f'losses_agg/{key}_mean'] = loss_mean
                log_dict[f'losses_agg/{key}_max'] = loss_max

            if self.use_wandb:
                wandb.log(log_dict)

        print('Finished training')
        if self.use_wandb:
            log_images_from_w(w_pivots, self.G, [image[0] for image in images])
        w_pivots = torch.cat(w_pivots)
        return w_pivots
