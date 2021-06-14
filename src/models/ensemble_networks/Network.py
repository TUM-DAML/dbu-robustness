import numpy as np
import torch
from torch import nn
from torch import autograd
from src.architectures.linear_sequential import linear_sequential
from src.architectures.convolution_linear_sequential import convolution_linear_sequential
from src.architectures.vgg_sequential import vgg16_bn


class Network(nn.Module):
    def __init__(self,
                 input_dims,  # Input dimension. list of ints
                 output_dim,  # Output dimension. int
                 hidden_dims=[64, 64, 64],  # Hidden dimensions. list of ints
                 kernel_dim=None,  # Kernel dimension if conv architecture. int
                 architecture='linear',  # Encoder architecture name. int
                 k_lipschitz=None,  # Lipschitz constant. float or None (if no lipschitz)
                 batch_size=64,  # Batch size. int
                 lr=1e-3,  # Learning rate. float
                 loss='CE'):  # Loss name. string

        super(Network, self).__init__()
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

    def forward(self, input, output, return_output='hard', compute_loss=True):
        # Forward
        logits = self.sequential(input)

        soft_output_pred = self.softmax(logits)
        output_pred = self.predict(soft_output_pred)

        # Loss
        if compute_loss:
            if self.loss == 'CE':
                # soft_output = torch.zeros(output.shape[0], self.output_dim).to(output.device)
                # soft_output.scatter_(1, output, 1)
                # self.grad_loss = self.CE_loss(soft_output_pred, soft_output)
                self.grad_loss = self.CE_loss(logits, output)
            else:
                raise NotImplementedError

        if return_output == 'hard':
            return output_pred
        elif return_output == 'soft':
            return soft_output_pred
        else:
            raise AssertionError

    # def CE_loss(self, soft_output_pred, soft_output):
    #     with autograd.detect_anomaly():
    #         CE_loss = - torch.sum(soft_output.squeeze() * torch.log(soft_output_pred))
    #
    #         return CE_loss

    def CE_loss(self, logits, output):
        CE_loss = nn.CrossEntropyLoss(reduction='sum')(logits, output.squeeze())
        return CE_loss

    def step(self):
        self.optimizer.zero_grad()
        self.grad_loss.backward()
        self.optimizer.step()

    def predict(self, p):
        output_pred = torch.max(p, dim=-1)[1]
        return output_pred
