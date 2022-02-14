import copy
import pickle
from argparse import Namespace

import torch

from configs import paths_config, global_config
from models.e4e.psp import pSp
from training.networks import Generator


def save_tuned_G(generator, pivots, quads, run_id):
    generator = copy.deepcopy(generator).cpu()
    pivots = copy.deepcopy(pivots).cpu()
    torch.save({'generator': generator, 'pivots': pivots, 'quads': quads},
               f'{paths_config.checkpoints_dir}/model_{run_id}.pt')


def load_tuned_G(run_id):
    new_G_path = f'{paths_config.checkpoints_dir}/model_{run_id}.pt'
    with open(new_G_path, 'rb') as f:
        checkpoint = torch.load(f)

    new_G, pivots, quads = checkpoint['generator'], checkpoint['pivots'], checkpoint['quads']
    new_G = new_G.float().to(global_config.device).eval().requires_grad_(False)
    pivots = pivots.float().to(global_config.device)
    return new_G, pivots, quads


def load_old_G():
    return load_g(paths_config.stylegan2_ada_ffhq)


def load_g(file_path):
    with open(file_path, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].to(global_config.device).eval()
        old_G = old_G.float()
    return old_G


def initialize_e4e_wplus():
    ckpt = torch.load(paths_config.e4e, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = paths_config.e4e
    opts = Namespace(**opts)
    e4e_inversion_net = pSp(opts)
    e4e_inversion_net = e4e_inversion_net.eval().to(global_config.device).requires_grad_(False)
    return e4e_inversion_net


def load_from_pkl_model(tuned):
    model_state = {'init_args': tuned.init_args, 'init_kwargs': tuned.init_kwargs
        , 'state_dict': tuned.state_dict()}
    gen = Generator(*model_state['init_args'], **model_state['init_kwargs'])
    gen.load_state_dict(model_state['state_dict'])
    gen = gen.eval().cuda().requires_grad_(False)
    return gen


def load_generators(run_id):
    tuned, pivots, quads = load_tuned_G(run_id=run_id)
    original = load_old_G()
    gen = load_from_pkl_model(tuned)
    orig_gen = load_from_pkl_model(original)
    del tuned, original
    return gen, orig_gen, pivots, quads
