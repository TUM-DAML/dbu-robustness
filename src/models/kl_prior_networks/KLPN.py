import numpy as np
import torch
from torch import nn
from torch import autograd
from torch import lgamma, digamma
from src.architectures.linear_sequential import linear_sequential
from src.architectures.convolution_linear_sequential import convolution_linear_sequential
from src.architectures.vgg_sequential import vgg16_bn


class KLPN(nn.Module):
    def __init__(self,
                 input_dims,  # Input dimension. list of ints
                 output_dim,  # Output dimension. int
                 hidden_dims=[64, 64, 64],  # Hidden dimensions. list of ints
                 kernel_dim=None,  # Kernel dimension if conv architecture. int
                 architecture='linear',  # Encoder architecture name. int
                 k_lipschitz=None,  # Lipschitz constant. float or None (if no lipschitz)
                 batch_size=64,  # Batch size. int
                 lr=1e-3,  # Learning rate. float
                 loss='KL_in_out',  # Loss name. string
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

    def forward(self, input, output=None, ood_input=None, return_output='alpha', compute_loss=False):
        assert not ((output is None or ood_input is None) and compute_loss)

        # Forward
        # print('Input Nan', torch.isnan(input).sum())
        # print('Input Inf', torch.isinf(input).sum())
        logits = torch.clamp(self.sequential(input), max=70.)
        # print('Grad Nan', torch.isnan(logits.grad).sum())
        # logits = 70 * torch.sigmoid(self.sequential(input))

        alpha = torch.exp(logits)
        soft_output_pred = self.softmax(logits)  # shape: [batch_size, output_dim]
        output_pred = self.predict(soft_output_pred)

        # loss
        if compute_loss:
            soft_output = torch.zeros(output.shape[0], self.output_dim).to(output.device)
            soft_output.scatter_(1, output.unsqueeze(-1), 1)

            ood_logits = torch.clamp(self.sequential(ood_input), max=70.)
            ood_alpha = torch.exp(ood_logits)
            ood_soft_output = torch.ones((ood_input.shape[0], self.output_dim)).to(ood_input.device) / self.output_dim
            if self.loss == 'KL_in_out':
                self.grad_loss = self.KL_in_out(alpha, soft_output, ood_alpha, ood_soft_output)
            elif self.loss == 'Reverse_KL_in_out':
                self.grad_loss = self.Reverse_KL_in_out(alpha, soft_output, ood_alpha, ood_soft_output)
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

    def KL_in_out(self, alpha, soft_output, ood_alpha, ood_soft_output):
        alpha_0 = alpha.sum(1)  # shape: [batch_size]
        beta = 1e2
        in_target_alpha = soft_output * (1 + beta) + (1 - soft_output)
        in_target_alpha_0 = in_target_alpha.sum(1)
        KL_in = (lgamma(in_target_alpha_0) - lgamma(alpha_0)
                 + torch.sum(lgamma(alpha) - lgamma(in_target_alpha), -1)
                 + torch.sum((in_target_alpha - alpha) * (digamma(in_target_alpha) - digamma(in_target_alpha_0).unsqueeze(-1).repeat(1, self.output_dim)), -1)).sum()

        ood_alpha_0 = ood_alpha.sum(1)  # shape: [batch_size]
        ood_target_alpha = torch.ones_like(ood_soft_output)
        ood_target_alpha_0 = ood_target_alpha.sum(1)
        KL_out = (lgamma(ood_target_alpha_0) - lgamma(ood_alpha_0)
                  + torch.sum(lgamma(ood_alpha) - lgamma(ood_target_alpha), -1)
                  + torch.sum((ood_target_alpha - ood_alpha) * (digamma(ood_target_alpha) - digamma(ood_target_alpha_0).unsqueeze(-1).repeat(1, self.output_dim)), -1)).sum()

        return KL_in + KL_out

    def Reverse_KL_in_out(self, alpha, soft_output, ood_alpha, ood_soft_output):
        alpha_0 = alpha.sum(1)  # shape: [batch_size]
        beta = 1e2
        in_target_alpha = soft_output * (1 + beta) + (1 - soft_output)
        in_target_alpha_0 = in_target_alpha.sum(1)
        KL_in = (lgamma(alpha_0) - lgamma(in_target_alpha_0)
                 + torch.sum(lgamma(in_target_alpha) - lgamma(alpha), -1)
                 + torch.sum((alpha - in_target_alpha) * (digamma(alpha) - digamma(alpha_0).unsqueeze(-1).repeat(1, self.output_dim)), -1)).sum()

        ood_alpha_0 = ood_alpha.sum(1)  # shape: [batch_size]
        ood_target_alpha = torch.ones_like(ood_soft_output)
        ood_target_alpha_0 = ood_target_alpha.sum(1)
        KL_out = (lgamma(ood_alpha_0) - lgamma(ood_target_alpha_0)
                  + torch.sum(lgamma(ood_target_alpha) - lgamma(ood_alpha), -1)
                  + torch.sum((ood_alpha - ood_target_alpha) * (digamma(ood_alpha) - digamma(ood_alpha_0).unsqueeze(-1).repeat(1, self.output_dim)), -1)).sum()

        return KL_in + KL_out

    def step(self):
        self.optimizer.zero_grad()
        self.grad_loss.backward()
        self.optimizer.step()

    def predict(self, p):
        output_pred = torch.max(p, dim=-1)[1]
        return output_pred
