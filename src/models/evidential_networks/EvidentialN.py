import numpy as np
import torch
from torch import nn
from torch import autograd
from torch import lgamma, digamma
from src.architectures.linear_sequential import linear_sequential
from src.architectures.convolution_linear_sequential import convolution_linear_sequential
from src.architectures.vgg_sequential import vgg16_bn


class EvidentialN(nn.Module):
    def __init__(self,
                 input_dims,  # Input dimension. list of ints
                 output_dim,  # Output dimension. int
                 hidden_dims=[64, 64, 64],  # Hidden dimensions. list of ints
                 kernel_dim=None,  # Kernel dimension if conv architecture. int
                 architecture='linear',  # Encoder architecture name. int
                 k_lipschitz=None,  # Lipschitz constant. float or None (if no lipschitz)
                 batch_size=64,  # Batch size. int
                 lr=1e-3,  # Learning rate. float
                 loss='UMSE',  # Loss name. string
                 seed=123):  # Random seed for init. int
        super().__init__()

        torch.cuda.manual_seed(seed)
        torch.set_default_tensor_type(torch.FloatTensor)

        # Architecture parameters
        self.input_dims, self.output_dim, self.hidden_dims, self.kernel_dim = input_dims, output_dim, hidden_dims, kernel_dim
        self.k_lipschitz = k_lipschitz
        # Training parameters
        self.batch_size, self.lr = batch_size, lr
        self.loss = loss

        # Feature selection
        if architecture == 'linear':
            self.sequential = linear_sequential(input_dims=self.input_dims,
                                                hidden_dims=self.hidden_dims,
                                                output_dim=self.output_dim,
                                                k_lipschitz=self.k_lipschitz)
        elif architecture == 'conv':
            assert len(input_dims) == 3
            self.sequential = convolution_linear_sequential(input_dims=self.input_dims,
                                                            linear_hidden_dims=self.hidden_dims,
                                                            conv_hidden_dims=[64, 64, 64],
                                                            output_dim=self.output_dim,
                                                            kernel_dim=self.kernel_dim,
                                                            k_lipschitz=self.k_lipschitz)
        elif architecture == 'vgg':
            assert len(input_dims) == 3
            self.sequential = vgg16_bn(output_dim=self.output_dim, k_lipschitz=self.k_lipschitz)
        else:
            raise NotImplementedError
        self.softmax = nn.Softmax(dim=-1)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, input, output=None, return_output='alpha', compute_loss=False, epoch=0.):
        assert not (output is None and compute_loss)

        # Forward
        logits = torch.clamp(self.sequential(input), max=70.)

        alpha = 1 + torch.exp(logits)
        soft_output_pred = self.softmax(logits)  # shape: [batch_size, output_dim]
        output_pred = self.predict(soft_output_pred)

        # loss
        if compute_loss:
            soft_output = torch.zeros(output.shape[0], self.output_dim).to(output.device)
            soft_output.scatter_(1, output.unsqueeze(-1), 1)

            if self.loss == 'UMSE':
                regr = np.minimum(1.0, epoch / 10.)
                self.grad_loss = self.UMSE(alpha, soft_output, regr)
            else:
                raise NotImplementedError

        if return_output == 'hard':
            return output_pred
        elif return_output == 'soft':
            return soft_output_pred
        elif return_output == 'alpha':
            return alpha
        else:
            raise AssertionError

    def UMSE(self, alpha, soft_output, regr):
        S = alpha.sum(1, keepdim=True)
        p = alpha / S
        UMSE = torch.sum((soft_output - p)**2 + p*(1-p) / (S + 1))

        alpha_0 = alpha.sum(1)  # shape: [batch_size]
        flat_alpha = torch.ones_like(alpha)
        flat_alpha_0 = flat_alpha.sum(1)
        KL_reg = (lgamma(alpha_0) - lgamma(flat_alpha_0)
                 + torch.sum(lgamma(flat_alpha) - lgamma(alpha), -1)
                 + torch.sum((alpha - flat_alpha) * (digamma(alpha) - digamma(alpha_0).unsqueeze(-1).repeat(1, self.output_dim)), -1)).sum()
        return UMSE + regr * KL_reg

    def step(self):
        self.optimizer.zero_grad()
        self.grad_loss.backward()
        self.optimizer.step()

    def predict(self, p):
        output_pred = torch.max(p, dim=-1)[1]
        return output_pred
