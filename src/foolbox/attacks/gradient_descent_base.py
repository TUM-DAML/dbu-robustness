from typing import Union, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd

from ..foolbox_types import Bounds

from ..models.base import Model

from ..criteria import Misclassification, Alpha0, DiffEntropy, DistUncertainty

from ..distances import l1, l2, linf

from .base import FixedEpsilonAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs

import torch
import numpy as np


class BaseGradientDescent(FixedEpsilonAttack, ABC):
    def __init__(
        self,
        *,
        rel_stepsize: float,
        abs_stepsize: Optional[float] = None,
        steps: int,
        random_start: bool,
        lambda_diffa: float,     # used to weight terms of alphadist loss
        lambda_ce_uncert: float, # used to weight uncertainty-term in crossentropy_x loss (x = alpha0, diffE, distU)
        lambda_alpha0: float,
        lambda_diffE: float,
        lambda_distU: float,
    ):
        self.rel_stepsize = rel_stepsize
        self.abs_stepsize = abs_stepsize
        self.steps = steps
        self.random_start = random_start
        self.lambda_diffa = lambda_diffa
        self.lambda_ce_uncert = lambda_ce_uncert
        self.lambda_alpha0=lambda_alpha0
        self.lambda_diffE=lambda_diffE
        self.lambda_distU=lambda_distU

    # modified 
    def get_loss_fn(
        self, model: Model, labels: ep.Tensor, loss_name: Any, data_type: Any
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        
        # can be overridden by users
        def loss_crossentropy(inputs: ep.Tensor) -> ep.Tensor: # only for in data
            alpha = model(inputs)
            logits = ep.astensor(torch.log(alpha.raw))
            return ep.crossentropy(logits, labels).sum()


        def get_alpha0(inputs: ep.Tensor) -> ep.Tensor:
            alpha = model(inputs)
            alpha0 = alpha.sum(axis=1)
            return alpha0

        
        def get_diff_entropy(inputs: ep.Tensor) -> ep.Tensor:
            alpha = model(inputs).raw
            alpha0 = alpha.sum(axis=1)
            log_term = torch.sum(torch.lgamma(alpha), axis=1) - torch.lgamma(alpha0)
            digamma_term = torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))), axis = 1)
            differential_entropy = log_term - digamma_term
            diffE = ep.astensor(differential_entropy)

            return diffE

        def get_dist_uncertainty(inputs: ep.Tensor) -> ep.Tensor:
            alpha = model(inputs).raw
            alpha0 = alpha.sum(axis=1)
            probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
            total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=1)
            digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) +1.0)
            dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
            exp_data_uncertainty = -1 * torch.sum(dirichlet_mean * digamma_term, dim=1)
            distributional_uncertainty = total_uncertainty - exp_data_uncertainty
            distU = ep.astensor(distributional_uncertainty)

            return distU



        # careful with signs: 
        # attack tries to maximize the loss
        # adversarial update step: adv = x + stepsize * grad (grad points in direction of greatest increase)
        def loss_alpha0(inputs: ep.Tensor) -> ep.Tensor:
            alpha0 = get_alpha0(inputs)
            if data_type == 'in':
                loss_a = -alpha0.sum()
            elif data_type == 'out':
                loss_a = alpha0.sum()
            return loss_a
        
        def loss_diff_entropy(inputs: ep.Tensor) -> ep.Tensor:
            diffE = get_diff_entropy(inputs)
            if data_type == 'in':
                loss_de = diffE.sum()
            elif data_type == 'out':
                loss_de = -diffE.sum()
            return loss_de

        def loss_dist_uncertainty(inputs: ep.Tensor) -> ep.Tensor:
            distU = get_dist_uncertainty(inputs)
            if data_type == 'in':
                loss_du = distU.sum()
            elif data_type == 'out':
                loss_du = -distU.sum()
            return loss_du

        def loss_alpha_dist(inputs: ep.Tensor) -> ep.Tensor: # only for in data
            alpha = model(inputs).raw
            ind_points = np.array(range(len(labels.raw)))
            ind_corr = labels.raw
            ind_max = torch.max(alpha, axis=1)[1]
            ind_2max = torch.topk(alpha, 2, axis=1)[1][:,1]
            ind_alphak = ind_2max
            ind_alphak[ind_max != labels.raw] = ind_max[ind_max != labels.raw]
            alpha_c = alpha[ind_points, ind_corr]
            alpha_k = alpha[ind_points, ind_alphak]

            loss_dist = -ep.astensor(alpha_c - self.lambda_diffa * alpha_k).sum()

            return loss_dist






        # punish high uncertainty 
        # attack: maximizes loss funktion 
        # second goal of the attack: maximizes alpha0, minimizes diffE, minimizes distU
        def loss_ce_alpha0(inputs: ep.Tensor) -> ep.Tensor: # only for in data
            loss_comb = loss_crossentropy(inputs) + self.lambda_ce_uncert * self.lambda_alpha0 * get_alpha0(inputs).sum()
            return loss_comb

        def loss_ce_diffe(inputs: ep.Tensor) -> ep.Tensor: # only for in data
            loss_comb = loss_crossentropy(inputs) - self.lambda_ce_uncert * self.lambda_diffE * get_diff_entropy(inputs).sum()
            return loss_comb
        
        def loss_ce_distu(inputs: ep.Tensor) -> ep.Tensor: # only for in data
            loss_comb = loss_crossentropy(inputs) - self.lambda_ce_uncert * self.lambda_distU * get_dist_uncertainty(inputs).sum()
            return loss_comb


        loss_dict = {'crossentropy': loss_crossentropy, 
                     'alpha0': loss_alpha0,
                     'diffE': loss_diff_entropy,
                     'distU': loss_dist_uncertainty,
                     'alphadist': loss_alpha_dist,
                     'ce_alpha0': loss_ce_alpha0,
                     'ce_diffe': loss_ce_diffe,
                     'ce_distu': loss_ce_distu,
                    }

        return loss_dict[loss_name]






    def value_and_grad(
        # can be overridden by users
        self,
        loss_fn: Callable[[ep.Tensor], ep.Tensor],
        x: ep.Tensor,
    ) -> Tuple[ep.Tensor, ep.Tensor]:
        return ep.value_and_grad(loss_fn, x)

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, T],
        *,
        epsilon: float,
        loss_name: Any,
        data_type: Any,
        punished_measure: Any,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        if not isinstance(criterion_, Misclassification) and not isinstance(criterion_, Alpha0) and not isinstance(criterion_, DiffEntropy) and not isinstance(criterion_, DistUncertainty):
            raise ValueError("unsupported criterion")

        labels = criterion_.labels
        loss_fn = self.get_loss_fn(model, labels, loss_name, data_type)

        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        x = x0

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model.bounds)
        else:
            x = x0

        for _ in range(self.steps):
            _, gradients = self.value_and_grad(loss_fn, x)
            gradients = self.normalize(gradients, x=x, bounds=model.bounds)
            x = x + stepsize * gradients
            x = self.project(x, x0, epsilon)
            x = ep.clip(x, *model.bounds)

        return restore_type(x)

    @abstractmethod
    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        ...

    @abstractmethod
    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        ...

    @abstractmethod
    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        ...


def clip_lp_norms(x: ep.Tensor, *, norm: float, p: float) -> ep.Tensor:
    assert 0 < p < ep.inf
    norms = flatten(x).norms.lp(p=p, axis=-1)
    norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
    factor = ep.minimum(1, norm / norms)  # clipping -> decreasing but not increasing
    factor = atleast_kd(factor, x.ndim)
    return x * factor


def normalize_lp_norms(x: ep.Tensor, *, p: float) -> ep.Tensor:
    assert 0 < p < ep.inf
    norms = flatten(x).norms.lp(p=p, axis=-1)
    norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
    factor = 1 / norms
    factor = atleast_kd(factor, x.ndim)
    return x * factor


def uniform_l1_n_balls(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    # https://mathoverflow.net/a/9188
    u = ep.uniform(dummy, (batch_size, n))
    v = u.sort(axis=-1)
    vp = ep.concatenate([ep.zeros(v, (batch_size, 1)), v[:, : n - 1]], axis=-1)
    assert v.shape == vp.shape
    x = v - vp
    sign = ep.uniform(dummy, (batch_size, n), low=-1.0, high=1.0).sign()
    return sign * x


def uniform_l2_n_spheres(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    x = ep.normal(dummy, (batch_size, n + 1))
    r = x.norms.l2(axis=-1, keepdims=True)
    s = x / r
    return s


def uniform_l2_n_balls(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    """Sampling from the n-ball

    Implementation of the algorithm proposed by Voelker et al. [#Voel17]_

    References:
        .. [#Voel17] Voelker et al., 2017, Efficiently sampling vectors and coordinates
            from the n-sphere and n-ball
            http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """
    s = uniform_l2_n_spheres(dummy, batch_size, n + 1)
    b = s[:, :n]
    return b


class L1BaseGradientDescent(BaseGradientDescent):
    distance = l1

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        batch_size, n = flatten(x0).shape
        r = uniform_l1_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        return normalize_lp_norms(gradients, p=1)

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + clip_lp_norms(x - x0, norm=epsilon, p=1)


class L2BaseGradientDescent(BaseGradientDescent):
    distance = l2

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        batch_size, n = flatten(x0).shape
        r = uniform_l2_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        return normalize_lp_norms(gradients, p=2)

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + clip_lp_norms(x - x0, norm=epsilon, p=2)


class LinfBaseGradientDescent(BaseGradientDescent):
    distance = linf

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.uniform(x0, x0.shape, -epsilon, epsilon)

    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        return gradients.sign()

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.clip(x - x0, -epsilon, epsilon)
