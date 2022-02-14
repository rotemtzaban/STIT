import torch

l2_criterion = torch.nn.MSELoss(reduction='mean')


def l2_loss(real_images, generated_images):
    loss = l2_criterion(real_images, generated_images)
    return loss
