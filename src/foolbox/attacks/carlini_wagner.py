from typing import Union, Tuple, Any, Optional
from functools import partial
import numpy as np
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd

from ..foolbox_types import Bounds

from ..models import Model

from ..distances import l2

from ..criteria import Misclassification
from ..criteria import TargetedMisclassification
from ..criteria import Alpha0, DiffEntropy, DistUncertainty

from .base import MinimizationAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs

import torch
from tqdm import tqdm

class L2CarliniWagnerAttack(MinimizationAttack):
    """Implementation of the Carlini & Wagner L2 Attack. [#Carl16]_

    Args:
        binary_search_steps : Number of steps to perform in the binary search
            over the const c.
        steps : Number of optimization steps within each binary search step.
        stepsize : Stepsize to update the examples.
        confidence : Confidence required for an example to be marked as adversarial.
            Controls the gap between example and decision boundary.
        initial_const : Initial value of the const c with which the binary search starts.
        abort_early : Stop inner search as soons as an adversarial example has been found.
            Does not affect the binary search over the const c.

    References:
        .. [#Carl16] Nicholas Carlini, David Wagner, "Towards evaluating the robustness of
            neural networks. In 2017 ieee symposium on security and privacy"
            https://arxiv.org/abs/1608.04644
    """

    distance = l2

    def __init__(
        self,
        binary_search_steps: int = 9, # foolbox-default: 9, Carlini-Wagner-Paper-default: 20
        steps: int = 10000,            # foolbox-default: 10000, Carlini-Wagner-Paper-default: 10000
        stepsize: float = 1e-2,
        confidence: float = 0,
        initial_const: float = 1e-3,
        abort_early: bool = True,
        lambda_uncert: float = 1.0,   # weights punished uncertainty
        lambda_alpha0: float = 0.01,  # map alpha0 for in data alpha0 is in [10, 100] (in: high, out:low), L2 distance is in [0,1]
        lambda_diffE: float = 0.05,   # map diffE, diffE is in  [-32.45, -12.80] if alpha0=100 (in: low, out: high)
        lambda_distU: float = 3.0,    # map distU, distU is in [0.038, 0.3736] if alpha0=100 (in: low, out: high)
        uncertainty_tolerance: float = 0.2, # percentage of more uncertainty with respect to original input that is allowed for adv
        lambda_diffa: float = 2.0,     # used to weight terms of alphadist loss
        lambda_ce_uncert: float = 1.0, # used to weight uncertainty-term in crossentropy_x loss (x = alpha0, diffE, distU)
    ):
        self.binary_search_steps = binary_search_steps
        self.steps = steps
        self.stepsize = stepsize
        self.confidence = confidence
        self.initial_const = initial_const
        self.abort_early = abort_early
        self.lambda_alpha0 = lambda_uncert * lambda_alpha0
        self.lambda_diffE = lambda_uncert * lambda_diffE
        self.lambda_distU = lambda_uncert * lambda_distU
        self.uncertainty_tolerance = uncertainty_tolerance
        self.lambda_diffa = lambda_diffa
        self.lambda_ce_uncert = lambda_ce_uncert

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: T, #Union[Misclassification, TargetedMisclassification, T],
        *,
        early_stop: Optional[float] = None,
        loss_name: Any,
        data_type: Any,
        punished_measure: Any,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        N = len(x)


        # to speed up we use loss name to identify the criterion
        # carefull: crossentropy will always result in criterion Misclassification, TargetedMisclassification is unsable
        #if isinstance(criterion_, Misclassification):
        if loss_name == 'crossentropy':
            targeted = False
            classes = criterion_.labels
            change_classes_logits = self.confidence
            label_attack = True
            criterion_type = 'misclassification'
        #elif isinstance(criterion_, TargetedMisclassification):
        elif loss_name == 'crossentropy_targeted':
            targeted = True
            classes = criterion_.target_classes
            change_classes_logits = -self.confidence
            label_attack = True
            criterion_type = 'misclassification'
        #elif isinstance(criterion_, Alpha0):
        elif loss_name == 'alpha0':
            targeted = False
            if data_type == 'in':
                classes = ep.astensor(torch.ones_like(criterion_.labels.raw))
            else:
                classes = ep.astensor(torch.zeros_like(criterion_.labels.raw))
            change_classes_logits = 0
            label_attack = False
            criterion_type = 'alpha0'
        #elif isinstance(criterion_, DiffEntropy):
        elif loss_name == 'diffE':
            targeted = False
            if data_type == 'in':
                classes = ep.astensor(torch.ones_like(criterion_.labels.raw))
            else:
                classes = ep.astensor(torch.zeros_like(criterion_.labels.raw))           
            change_classes_logits = 0
            label_attack = False
            criterion_type = 'diffE'
        #elif isinstance(criterion_, DistUncertainty):
        elif loss_name == 'distU':
            targeted = False
            if data_type == 'in':
                classes = ep.astensor(torch.ones_like(criterion_.labels.raw))
            else:
                classes = ep.astensor(torch.zeros_like(criterion_.labels.raw))
            change_classes_logits = 0
            label_attack = False
            criterion_type = 'distU'
        else:
            raise ValueError("unsupported criterion")

        def is_adversarial(perturbed: ep.Tensor, logits: ep.Tensor) -> ep.Tensor:
            if change_classes_logits != 0:
                logits += ep.onehot_like(logits, classes, value=change_classes_logits)
            return criterion_(perturbed, logits)

        if classes.shape != (N,):
            name = "target_classes" if targeted else "labels"
            raise ValueError(
                f"expected {name} to have shape ({N},), got {classes.shape}"
            )

        bounds = model.bounds
        to_attack_space = partial(_to_attack_space, bounds=bounds)
        to_model_space = partial(_to_model_space, bounds=bounds)

        x_attack = to_attack_space(x)
        reconstsructed_x = to_model_space(x_attack)
        #reconstructed_classes = model(reconstsructed_x).argmax(axis=-1)
        reconstructed_alpha = model(reconstsructed_x)
        reconstructed_alpha_raw = reconstructed_alpha.raw
        reconstructed_classes = reconstructed_alpha.argmax(axis=-1)
        reconstructed_uncertainty = self.get_uncertainty_measure(model, reconstsructed_x, reconstructed_alpha_raw, reconstructed_classes, punished_measure)
        classes_raw = classes.raw

        rows = range(N)

        ## precompute
        #alpha = model(x)
        #alpha_raw = alpha.raw
        #logits = ep.astensor(torch.log(alpha_raw))
        ## uncertainty measure w.r.t. original/unperturbed input data
        #uncertM_orig = self.get_uncertainty_measure(model, x, alpha_raw, classes, punished_measure)
        #uncertM_adv = self.get_uncertainty_measure(model, reconstsructed_x, reconstructed_alpha_raw, reconstructed_classes, punished_measure)


        def loss_fun(
            delta: ep.Tensor, consts: ep.Tensor, 
        ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
            assert delta.shape == x_attack.shape
            assert consts.shape == (N,)

            x = to_model_space(x_attack + delta)

            # logits = model(x)
            alpha = model(x)
            alpha_raw = alpha.raw
            logits = ep.astensor(torch.log(alpha_raw))

            # uncertainty measure w.r.t. original/unperturbed input data
            if punished_measure is None:
                uncertainty = None
            else:
                uncertainty = self.get_uncertainty_measure(model, x, alpha_raw, classes, punished_measure)
            #uncertM_adv = self.get_uncertainty_measure(model, reconstsructed_x, reconstructed_alpha_raw, reconstructed_classes, punished_measure)
            
            if punished_measure == 'alpha0' and label_attack:
                uncert_punish = ((1.0-self.uncertainty_tolerance)*reconstructed_uncertainty - uncertainty).maximum(0.0)
                lambda_uncert = self.lambda_alpha0
           
            elif punished_measure == 'diffE' and label_attack:
                uncert_punish = (uncertainty - (1.0+self.uncertainty_tolerance)*reconstructed_uncertainty).maximum(0.0)
                lambda_uncert = self.lambda_diffE
            
            elif punished_measure == 'distU' and label_attack:
                uncert_punish = (uncertainty - (1.0+self.uncertainty_tolerance)*reconstructed_uncertainty).maximum(0.0)
                lambda_uncert = self.lambda_distU
            
            elif punished_measure == None:
                #print('The chosen uncertainty measure is not implemente such that it can be considered during carlini-wagner-attacks.')
                uncert_punish = 0.0
                lambda_uncert = 0.0
            
            else:
                print('The chosen uncertainty measure is not implemente such that it can be considered during carlini-wagner-attacks.')
                uncert_punish = 0.0
                lambda_uncert = 0.0

            if targeted:
                c_minimize = best_other_classes_raw(alpha_raw, classes_raw)
                c_maximize = classes  # target_classes
            else:
                c_minimize = classes  # labels
                c_maximize = best_other_classes_raw(alpha_raw, classes_raw)


            # careful with signs: 
            # attack tries to maximize the loss
            # adversarial update step: adv = x + stepsize * grad (grad points in direction of greatest increase)


            # check isinstance requires time
            #if isinstance(criterion_, Misclassification) or isinstance(criterion_, TargetedMisclassification):
            if criterion_type == 'misclassification':
                # is_adv_loss = logits[:, c_minimize] - logits[:, c_maximize]
                is_adv_loss = ep.astensor(torch.gather(logits.raw, -1, c_minimize.raw.unsqueeze(1)).squeeze(-1) -
                               torch.gather(logits.raw, -1, c_maximize.raw.unsqueeze(1)).squeeze(-1))

            #elif isinstance(criterion_, Alpha0):
            elif criterion_type == 'alpha0':
                is_adv_loss = -self.get_uncertainty_measure(model, x, alpha_raw, classes, 'alpha0')
                if data_type == 'out':
                    is_adv_loss = -is_adv_loss
            
            #elif isinstance(criterion_, DiffEntropy):
            elif criterion_type == 'diffE':
                is_adv_loss = self.get_uncertainty_measure(model, x, alpha_raw, classes, 'diffE')
                if data_type == 'out':
                    is_adv_loss = -is_adv_loss
            
            #elif isinstance(criterion_, DistUncertainty):
            elif criterion_type == 'distU':
                is_adv_loss = self.get_uncertainty_measure(model, x, alpha_raw, classes, 'distU')
                if data_type == 'out':
                    is_adv_loss = -is_adv_loss
            
            else: 
                print('Wrong criterion, using Misclassification')
                is_adv_loss = ep.astensor(torch.gather(logits.raw, -1, c_minimize.raw.unsqueeze(1)).squeeze(-1) -
                               torch.gather(logits.raw, -1, c_maximize.raw.unsqueeze(1)).squeeze(-1))

            #is_adv_loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert is_adv_loss.shape == (N,)

            #is_adv_loss = is_adv_loss + self.confidence
            is_adv_loss = is_adv_loss + self.confidence  - lambda_uncert * uncert_punish # modified to punish worse uncertainty measures
            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts

            squared_norms = flatten(x - reconstsructed_x).square().sum(axis=-1)
            loss = is_adv_loss.sum() + squared_norms.sum()
            return loss, (x, logits)

        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)

        consts = self.initial_const * np.ones((N,))
        lower_bounds = np.zeros((N,))
        upper_bounds = np.inf * np.ones((N,))

        best_advs = ep.zeros_like(x)
        best_advs_norms = ep.full(x, (N,), ep.inf)
        measure_best = ep.full(x, (N,), ep.inf)

        # the binary search searches for the smallest consts that produce adversarials
        for binary_search_step in range(self.binary_search_steps):
            if (
                binary_search_step == self.binary_search_steps - 1
                and self.binary_search_steps >= 10
            ):
                # in the last binary search step, repeat the search once
                consts = np.minimum(upper_bounds, 1e10)

            # create a new optimizer find the delta that minimizes the loss
            delta = ep.zeros_like(x_attack)
            optimizer = AdamOptimizer(delta)

            # tracks whether adv with the current consts was found
            #found_advs = np.full((N,), fill_value=False)
            found_advs = torch.tensor(np.full((N,), fill_value=False)).to(device) # speed up to avoid shifts from gpu to cpu
            loss_at_previous_check = np.inf

            consts_ = ep.from_numpy(x, consts.astype(np.float32))

            for step in range(self.steps):
                loss, (perturbed, logits), gradient = loss_aux_and_grad(delta, consts_)
                delta += optimizer(gradient, self.stepsize)

                if self.abort_early and step % (np.ceil(self.steps / 10)) == 0:
                    # after each tenth of the overall steps, check progress
                    if not (loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has been no progress
                    loss_at_previous_check = loss

                if label_attack:
                    found_advs_iter = is_adversarial(perturbed, logits)
                    #found_advs = np.logical_or(found_advs, found_advs_iter.numpy())
                    found_advs = ep.logical_or(found_advs, found_advs_iter)
                else: 
                    found_advs_iter = ep.astensor(torch.tensor(np.array([True]*len(classes))).to(device)) # we do not want to set a threshold
                    #found_advs = np.logical_or(found_advs, found_advs_iter.numpy())
                    found_advs = ep.logical_or(found_advs, found_advs_iter)

                    # check if uncertainty measure is better (alph0 --> high, diffE/distU --> low)
                    measure_adv = self.get_uncertainty_measure(model, perturbed, None, classes, criterion_type)
                    if data_type == 'in':
                        if criterion_type == 'alpha0':
                            larger_req = False
                        else:
                            larger_req = True
                    else:
                        if criterion_type == 'alpha0':
                            larger_req = True
                        else:
                            larger_req = False
                    if larger_req:
                        uncert = measure_best > measure_adv
                    else:
                        uncert = measure_best < measure_adv


                norms = flatten(perturbed - x).norms.l2(axis=-1)
                closer = norms < best_advs_norms
                if label_attack:
                    new_best = ep.logical_and(closer, found_advs_iter)
                    new_best_ = atleast_kd(new_best, best_advs.ndim)
                    best_advs = ep.where(new_best_, perturbed, best_advs)
                    best_advs_norms = ep.where(new_best, norms, best_advs_norms)
                else:
                    new_best = ep.logical_and(closer, uncert)
                    new_best_ = atleast_kd(new_best, best_advs.ndim)
                    best_advs = ep.where(new_best_, perturbed, best_advs)
                    best_advs_norms = ep.where(new_best, norms, best_advs_norms)
                    measure_best = ep.where(new_best, measure_adv, measure_best)

            # upper_bounds = np.where(found_advs, consts, upper_bounds)
            #lower_bounds = np.where(found_advs, lower_bounds, consts)
            upper_bounds = np.where(found_advs.raw.cpu().numpy(), consts, upper_bounds)
            lower_bounds = np.where(found_advs.raw.cpu().numpy(), lower_bounds, consts)
            consts_exponential_search = consts * 10
            consts_binary_search = (lower_bounds + upper_bounds) / 2
            consts = np.where(
                np.isinf(upper_bounds), consts_exponential_search, consts_binary_search
            )

        return restore_type(best_advs)

    
# add uncertainty measures
    def get_uncertainty_measure(self, 
        model: Model,
        inputs: ep.Tensor, 
        alpha: Any,
        labels: Any, 
        measure: Any) -> ep.Tensor:

        # using a dict is too slow, speed up
        # measure_dict = { 'alpha0': self.get_alpha0(model, inputs, labels), 
        #                  'diffE':  self.get_diff_entropy(model, inputs, labels),
        #                  'distU':  self.get_dist_uncertainty(model, inputs, labels),
        #                  #'alphaD': self.get_alpha_dist(model, inputs, labels),
        #                  None: None,
        # }
        # 
        # return measure_dict[measure]

        if measure is None:
            m_val = None
        elif measure == 'alpha0':
            m_val = self.get_alpha0(model, inputs, alpha, labels)
        elif measure == 'diffE':
            m_val = self.get_diff_entropy(model, inputs, alpha, labels)
        elif measure == 'distU':
            m_val = self.get_dist_uncertainty(model, inputs, alpha, labels)
        #elif measure == 'alphaD':
        #    m_val = self.get_alpha_dist(model, inputs, labels)
        else:
            print('This measure does not exist, returning None.')
            m_val = None

        return m_val



    def get_alpha0(self,
        model: Model,
        inputs: ep.Tensor, 
        alpha: Any,
        labels: Any) -> ep.Tensor:

        if alpha is None:
            alpha = model(inputs)
        alpha0 = alpha.sum(axis=1)
        return alpha0

    
    def get_diff_entropy(self,
        model: Model,
        inputs: ep.Tensor, 
        alpha: Any,
        labels: Any) -> ep.Tensor:

        if alpha is None:
            alpha = model(inputs).raw
        alpha0 = alpha.sum(axis=1)
        log_term = torch.sum(torch.lgamma(alpha), axis=1) - torch.lgamma(alpha0)
        digamma_term = torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))), axis = 1)
        differential_entropy = log_term - digamma_term
        diffE = ep.astensor(differential_entropy)

        return diffE


    def get_dist_uncertainty(self,
        model: Model,
        inputs: ep.Tensor,
        alpha: Any,
        labels: Any) -> ep.Tensor:

        if alpha is None:
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


    # def get_alpha_dist(self,
    #     model: Model,
    #     inputs: ep.Tensor, 
    #     labels: Any) -> ep.Tensor:

    #     alpha = model(inputs).raw
    #     ind_points = np.array(range(len(labels.raw)))
    #     ind_corr = labels.raw
    #     ind_max = torch.max(alpha, axis=1)[1]
    #     ind_2max = torch.topk(alpha, 2, axis=1)[1][:,1]
    #     ind_alphak = ind_2max
    #     ind_alphak[ind_max != labels.raw] = ind_max[ind_max != labels.raw]
    #     alpha_c = alpha[ind_points, ind_corr]
    #     alpha_k = alpha[ind_points, ind_alphak]
    #     alpha_dist = ep.astensor(alpha_c - self.lambda_diffa * alpha_k)

    #     return alpha_dist



class AdamOptimizer:
    def __init__(self, x: ep.Tensor):
        self.m = ep.zeros_like(x)
        self.v = ep.zeros_like(x)
        self.t = 0

    def __call__(
        self,
        gradient: ep.Tensor,
        stepsize: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> ep.Tensor:
        self.t += 1

        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2

        bias_correction_1 = 1 - beta1 ** self.t
        bias_correction_2 = 1 - beta2 ** self.t

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2

        return -stepsize * m_hat / (ep.sqrt(v_hat) + epsilon)


def best_other_classes(logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
    with torch.no_grad():
        other_logits = logits - torch.eye(logits.shape[-1], device=logits._raw.device)[exclude._raw] * 1e30
        # other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
        return other_logits.argmax(axis=-1)


def best_other_classes_raw(logits: Any, exclude: Any) -> ep.Tensor:
    with torch.no_grad():
        other_logits = logits - torch.eye(logits.shape[-1], device=logits.device)[exclude] * 1e30
        return ep.astensor(other_logits.argmax(axis=-1))




def best_other_classes_topk(logits: Any, exclude: Any) -> ep.Tensor:
    nr_points = logits.shape[0]
    _, top2 = torch.topk(logits, 2, dim=1)
    mask_inds = 1*(top2[:,0] == exclude) # 1 if we need second largest (index 1), 0 if we need largest (index 0)
    best2 = top2[range(nr_points),mask_inds]
    best2 = ep.astensor(best2)
    return best2



def _to_attack_space(x: ep.Tensor, *, bounds: Bounds) -> ep.Tensor:
    min_, max_ = bounds
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = (x - a) / b  # map from [min_, max_] to [-1, +1]
    x = x * 0.999999  # from [-1, +1] to approx. (-1, +1)
    x = x.arctanh()  # from (-1, +1) to (-inf, +inf)
    return x


def _to_model_space(x: ep.Tensor, *, bounds: Bounds) -> ep.Tensor:
    min_, max_ = bounds
    x = x.tanh()  # from (-inf, +inf) to (-1, +1)
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = x * b + a  # map from (-1, +1) to (min_, max_)
    return x
