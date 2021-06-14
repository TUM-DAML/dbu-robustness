import numpy as np
import torch
from torch import nn
from src.models.ensemble_networks.Network import Network


class EnsembleN(nn.Module):
    def __init__(self,
                 input_dims,  # Input dimension. list of ints
                 output_dim,  # Output dimension. int
                 hidden_dims=[64, 64, 64],  # Hidden dimensions. list of ints
                 kernel_dim=None,  # Kernel dimension if conv architecture. int
                 architecture='linear',  # Encoder architecture name. int
                 k_lipschitz=None,  # Lipschitz constant. float or None (if no lipschitz)
                 n_networks=10,  # Number of networks in ensemble. int
                 batch_size=64,  # Batch size. int
                 lr=1e-3,  # Learning rate. float
                 loss='CE',  # Loss name. string
                 seed=123):  # Random seed for init. int
        super().__init__()

        torch.cuda.manual_seed(seed)
        torch.set_default_tensor_type(torch.FloatTensor)

        # Architecture parameters
        self.input_dims, self.output_dim, self.hidden_dims, self.kernel_dim = input_dims, output_dim, hidden_dims, kernel_dim
        self.k_lipschitz = k_lipschitz
        self.n_networks = n_networks
        # Training parameters
        self.batch_size, self.lr = batch_size, lr
        self.loss = loss

        # Ensemble of networks
        self.networks = nn.ModuleList([Network(input_dims=self.input_dims,
                                               output_dim=self.output_dim,
                                               hidden_dims=self.hidden_dims,
                                               kernel_dim=self.kernel_dim,
                                               architecture=architecture,
                                               k_lipschitz=self.k_lipschitz,
                                               lr=self.lr,
                                               loss=self.loss) for n in range(self.n_networks)])

    def forward(self, input, output, return_output='hard', compute_loss=True):
        batch_size = input.size(0)
        perm = torch.cat([torch.arange(self.n_networks) * batch_size + i for i in range(batch_size)], dim=0)
        for n, network in enumerate(self.networks):
            if n == 0:
                pred = network(input, output, return_output=return_output, compute_loss=compute_loss)
            else:
                pred = torch.cat((pred, network(input, output, return_output=return_output, compute_loss=compute_loss)), dim=0)
        pred = pred[perm]
        return pred

    def ensemble_estimate(self, input):
        soft_output_pred = self.forward(input, None, return_output='soft', compute_loss=False).view(-1, self.n_networks, self.output_dim)
        approx_mean = torch.mean(soft_output_pred, -2)
        approx_var = torch.var(soft_output_pred, -2)
        return approx_mean, approx_var
