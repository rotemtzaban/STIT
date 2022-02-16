## Architecture
lpips_type = 'alex'
first_inv_type = 'e4e'

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = True
regularizer_l2_lambda = 0.1
regularizer_lpips_lambda = 0.1
regularizer_alpha = 30

## Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1

## Steps
max_pti_steps = 50
first_inv_steps = 50

## Optimization
pti_learning_rate = 3e-5
first_inv_lr = 5e-3
stitching_tuning_lr = 3e-4
pti_adam_beta1 = 0.9
lr_rampdown_length = 0.25
lr_rampup_length = 0.05
use_lr_ramp = False
