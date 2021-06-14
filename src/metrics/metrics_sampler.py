import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from sklearn import metrics


def accuracy(Y, mean_p):
    corrects = (Y.squeeze() == mean_p.max(-1)[1]).type(torch.DoubleTensor)
    accuracy = corrects.sum() / corrects.size(0)
    return accuracy.cpu().detach().numpy()


def confidence(Y, mean_p, var_p, score_type='AUROC', uncertainty_type='aleatoric'):
    corrects = (Y.squeeze() == mean_p.max(-1)[1]).cpu().detach().numpy()
    if uncertainty_type == 'epistemic':
        scores = - var_p[torch.arange(var_p.size(0)), mean_p.max(-1)[1]].cpu().detach().numpy()
    elif uncertainty_type == 'aleatoric':
        scores = mean_p.max(-1)[0].cpu().detach().numpy()

    if score_type == 'AUROC':
        fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
        return metrics.auc(fpr, tpr)
    elif score_type == 'APR':
        return metrics.average_precision_score(corrects, scores)
    else:
        raise NotImplementedError


def brier_score(Y, mean_p):
    batch_size = mean_p.size(0)

    p = mean_p.clone()
    indices = torch.arange(batch_size)
    p[indices, Y.squeeze()] -= 1.
    brier_score = p.norm(dim=-1).mean().cpu().detach().numpy()
    return brier_score


# OOD detection metrics
def anomaly_detection(mean_p, var_p, ood_mean_p, ood_var_p, score_type='AUROC', uncertainty_type='aleatoric'):
    if uncertainty_type == 'epistemic':
        scores = - var_p[torch.arange(var_p.size(0)), mean_p.max(-1)[1]].cpu().detach().numpy()
    elif uncertainty_type == 'aleatoric':
        scores = mean_p.max(-1)[0].cpu().detach().numpy()

    if uncertainty_type == 'epistemic':
        ood_scores = - ood_var_p[torch.arange(ood_var_p.size(0)), ood_mean_p.max(-1)[1]].cpu().detach().numpy()
    elif uncertainty_type == 'aleatoric':
        ood_scores = ood_mean_p.max(-1)[0].cpu().detach().numpy()

    corrects = np.concatenate([np.ones(mean_p.size(0)), np.zeros(ood_mean_p.size(0))], axis=0)
    scores = np.concatenate([scores, ood_scores], axis=0)

    if score_type == 'AUROC':
        fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
        return metrics.auc(fpr, tpr)
    elif score_type == 'APR':
        return metrics.average_precision_score(corrects, scores)
    else:
        raise NotImplementedError


def entropy(mean_p, uncertainty_type, n_bins=10, plot=True):
    entropy = []

    if uncertainty_type == 'aleatoric':
        entropy.append(Categorical(mean_p).entropy().squeeze().cpu().detach().numpy())

    if plot:
        plt.hist(entropy, n_bins)
        plt.show()
    return entropy
