import torch
import numpy as np
import wandb
from criteria import l2_loss
from configs import hyperparameters
from configs import global_config


class SpaceRegularizer:
    def __init__(self, original_G, lpips_net):
        self.original_G = original_G
        self.morphing_regularizer_alpha = hyperparameters.regularizer_alpha
        self.lpips_loss = lpips_net

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = hyperparameters.regularizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        self.morphing_regularizer_alpha * fixed_w + (1 - self.morphing_regularizer_alpha) * new_w_code

        return result_w

    def get_image_from_ws(self, w_codes, G):
        return torch.cat([G.synthesis(w_code, noise_mode='none', force_fp32=True) for w_code in w_codes])

    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch, log_name, use_wandb=False):
        loss = 0.0

        z_samples = np.random.randn(num_of_sampled_latents, self.original_G.z_dim)
        w_samples = self.original_G.mapping(torch.from_numpy(z_samples).to(global_config.device), None,
                                            truncation_psi=0.5)
        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

        for w_code in territory_indicator_ws:
            new_img = new_G.synthesis(w_code, noise_mode='none', force_fp32=True)
            with torch.no_grad():
                old_img = self.original_G.synthesis(w_code, noise_mode='none', force_fp32=True)

            if hyperparameters.regularizer_l2_lambda > 0:
                l2_loss_val = l2_loss.l2_loss(old_img, new_img)
                if use_wandb:
                    wandb.log({f'losses/space_regularizer_l2_loss_val_{log_name}': l2_loss_val.detach().cpu()}, commit=False)
                loss += l2_loss_val * hyperparameters.regularizer_l2_lambda

            if hyperparameters.regularizer_lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                if use_wandb:
                    wandb.log({f'losses/space_regularizer_lpips_loss_val_{log_name}': loss_lpips.detach().cpu()}, commit=False)
                loss += loss_lpips * hyperparameters.regularizer_lpips_lambda

        return loss / len(territory_indicator_ws)

    def space_regularizer_loss(self, new_G, w_batch, log_name, use_wandb):
        ret_val = self.ball_holder_loss_lazy(new_G, hyperparameters.latent_ball_num_of_samples, w_batch, log_name, use_wandb)
        return ret_val
