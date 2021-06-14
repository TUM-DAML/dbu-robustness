import torch
import numpy as np
from torch import nn
import eagerpy as ep


from src.foolbox.criteria import MisclassificationBest, Alpha0Best, DiffEntropyBest, DistUncertaintyBest


criterion_dict = {'crossentropy': lambda labels, start_data, threshold: MisclassificationBest(labels),
                  'alpha0': Alpha0Best,
                  'diffE': DiffEntropyBest,
                  'distU': DistUncertaintyBest,
                  }


def tile(a, dim, n_tile):
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    return a


class Gaussian_L2_Noise(nn.Module):

    def __init__(self, model, X, Y, loss_name, epsilons, start_data="in", **kwargs):
        super(Gaussian_L2_Noise, self).__init__()
        self.n_duplicates = 10
        self.model = model
        self.X = X
        self.epsilons = epsilons
        self.criterion = criterion_dict[loss_name](labels=Y.expand(self.n_duplicates), start_data=start_data, threshold=None)

    def forward(self):
        inputs = self.X
        duplicated_inputs = tile(inputs, 0, self.n_duplicates)
        noise_duplicated_inputs = duplicated_inputs + self.epsilons * torch.randn_like(duplicated_inputs)

        model_out = self.model(noise_duplicated_inputs)

        best_noise_input = self.criterion(noise_duplicated_inputs, model_out.to(torch.float64))

        return None, best_noise_input, None
