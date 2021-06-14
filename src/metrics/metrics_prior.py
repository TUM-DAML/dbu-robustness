import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.distributions import Dirichlet
from sklearn import metrics

import scipy
import scipy.special
import torch

def accuracy(Y, alpha):
    corrects = (Y.squeeze() == alpha.max(-1)[1]).type(torch.DoubleTensor)
    accuracy = corrects.sum() / corrects.size(0)
    return accuracy.cpu().detach().numpy()

def confidence(Y, alpha, score_type='AUROC', uncertainty_type='aleatoric'):
    corrects = (Y.squeeze() == alpha.max(-1)[1]).cpu().detach().numpy()
    if uncertainty_type == 'epistemic':
        scores = alpha.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'aleatoric':
        p = torch.nn.functional.normalize(alpha, p=1, dim=-1)
        scores = p.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'differential_entropy':
        eps = 1e-6
        alpha = alpha + eps
        alpha0 = alpha.sum(-1)
        log_term = torch.sum(torch.lgamma(alpha), axis=1) - torch.lgamma(alpha0)
        digamma_term = torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))), axis=1)
        differential_entropy = log_term - digamma_term
        scores = - differential_entropy.cpu().detach().numpy()
    elif uncertainty_type == 'distribution_uncertainty':
        eps = 1e-6
        alpha = alpha + eps
        alpha0 = alpha.sum(-1)
        probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=1)
        digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0)
        dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        exp_data_uncertainty = -1 * torch.sum(dirichlet_mean * digamma_term, dim=1)
        distributional_uncertainty = total_uncertainty - exp_data_uncertainty
        scores = - distributional_uncertainty.cpu().detach().numpy()

    if score_type == 'AUROC':
        fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
        return metrics.auc(fpr, tpr)
    elif score_type == 'APR':
        return metrics.average_precision_score(corrects, scores)
    else:
        raise NotImplementedError


def brier_score(Y, alpha):
    batch_size = alpha.size(0)

    p = torch.nn.functional.normalize(alpha, p=1, dim=-1)
    indices = torch.arange(batch_size)
    p[indices, Y.squeeze()] -= 1
    brier_score = p.norm(dim=-1).mean().cpu().detach().numpy()
    return brier_score


# OOD detection metrics
def anomaly_detection(alpha, ood_alpha, score_type='AUROC', uncertainty_type='aleatoric'):
    if uncertainty_type == 'epistemic':
        scores = alpha.sum(-1).cpu().detach().numpy()
    elif uncertainty_type == 'aleatoric':
        p = torch.nn.functional.normalize(alpha, p=1, dim=-1)
        scores = p.max(-1)[0].cpu().detach().numpy()

    if uncertainty_type == 'epistemic':
        ood_scores = ood_alpha.sum(-1).cpu().detach().numpy()
    elif uncertainty_type == 'aleatoric':
        p = torch.nn.functional.normalize(ood_alpha, p=1, dim=-1)
        ood_scores = p.max(-1)[0].cpu().detach().numpy()

    corrects = np.concatenate([np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0)
    scores = np.concatenate([scores, ood_scores], axis=0)

    if score_type == 'AUROC':
        fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
        return metrics.auc(fpr, tpr)
    elif score_type == 'APR':
        return metrics.average_precision_score(corrects, scores)
    else:
        raise NotImplementedError


def entropy(alpha, uncertainty_type, n_bins=10, plot=True):
    entropy = []

    if uncertainty_type == 'aleatoric':
        p = torch.nn.functional.normalize(alpha, p=1, dim=-1)
        entropy.append(Categorical(p).entropy().squeeze().cpu().detach().numpy())
    elif uncertainty_type == 'epistemic':
        entropy.append(Dirichlet(alpha).entropy().squeeze().cpu().detach().numpy())

    if plot:
        plt.hist(entropy, n_bins)
        plt.show()
    return entropy



# additional metric based on diffEentropyUncertainty
def diff_entropy(alpha, ood_alpha, score_type='AUROC'):
    eps = 1e-6
    alpha = alpha + eps
    ood_alpha = ood_alpha + eps
    alpha0 = alpha.sum(-1)
    ood_alpha0 = ood_alpha.sum(-1)

    id_log_term = torch.sum(torch.lgamma(alpha), axis=1) - torch.lgamma(alpha0)
    id_digamma_term = torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))), axis = 1)
    id_differential_entropy = id_log_term - id_digamma_term

    ood_log_term = torch.sum(torch.lgamma(ood_alpha), axis=1) - torch.lgamma(ood_alpha0)
    ood_digamma_term = torch.sum((ood_alpha - 1.0) * (torch.digamma(ood_alpha) - torch.digamma((ood_alpha0.reshape((ood_alpha0.size()[0], 1))).expand_as(ood_alpha))), axis = 1)
    ood_differential_entropy = ood_log_term - ood_digamma_term 

    scores = - id_differential_entropy.cpu().detach().numpy()
    ood_scores = - ood_differential_entropy.cpu().detach().numpy()

    corrects = np.concatenate([np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0)
    scores = np.concatenate([scores, ood_scores], axis=0)

    if score_type == 'AUROC':
        fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
        return metrics.auc(fpr, tpr)
    elif score_type == 'APR':
        return metrics.average_precision_score(corrects, scores)
    else:
        raise NotImplementedError


# additional metric based on  distUncertainty
def dist_uncertainty(alpha, ood_alpha, score_type='AUROC'):
    eps = 1e-6
    alpha = alpha + eps
    ood_alpha = ood_alpha + eps
    alpha0 = alpha.sum(-1)
    ood_alpha0 = ood_alpha.sum(-1)
    probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
    ood_probs = ood_alpha / ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha)

    id_total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=1)
    id_digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) +1.0)
    id_dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
    id_exp_data_uncertainty = -1 * torch.sum(id_dirichlet_mean * id_digamma_term, dim=1)
    id_distributional_uncertainty = id_total_uncertainty - id_exp_data_uncertainty

    ood_total_uncertainty = -1 * torch.sum(ood_probs * torch.log(ood_probs + 0.00001), dim=1)
    ood_digamma_term = torch.digamma(ood_alpha + 1.0) - torch.digamma(ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha) +1.0)
    ood_dirichlet_mean = ood_alpha / ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha)
    ood_exp_data_uncertainty = -1 * torch.sum(ood_dirichlet_mean * ood_digamma_term, dim=1)
    ood_distributional_uncertainty = ood_total_uncertainty - ood_exp_data_uncertainty


    scores = - id_distributional_uncertainty.cpu().detach().numpy()
    ood_scores = - ood_distributional_uncertainty.cpu().detach().numpy()

    corrects = np.concatenate([np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0)
    scores = np.concatenate([scores, ood_scores], axis=0)

    if score_type == 'AUROC':
        fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
        return metrics.auc(fpr, tpr)
    elif score_type == 'APR':
        return metrics.average_precision_score(corrects, scores)
    else:
        raise NotImplementedError
