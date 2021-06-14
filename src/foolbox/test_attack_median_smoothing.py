import torch
import numpy as np
import eagerpy as ep
import src.foolbox as fb
import src.foolbox.attacks as fa
from src.foolbox.criteria import Misclassification, Alpha0, DiffEntropy, DistUncertainty
from src.smoothing.median_smoothing import SmoothMedianNMS
from src.metrics.metrics_prior import accuracy
from sklearn import metrics as sk_metrics

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

attack_dict = {'FGSM_Linf': lambda kwargs={}: fa.FGSM(**kwargs),
               'FGSM_L2': lambda kwargs={}: fa.FGM(**kwargs),
               'PGD_Linf': lambda kwargs={}:  fa.LinfProjectedGradientDescentAttack(**kwargs),
               'PGD_L2': lambda kwargs={}:  fa.L2ProjectedGradientDescentAttack(**kwargs),
               'Gaussian_L2_Noise': lambda kwargs={}:  fa.Gaussian_L2_Noise(**kwargs),
               }

criterion_dict = {'crossentropy': lambda label, start_data, threshold: Misclassification(label),
                  'alphadist': lambda label, start_data, threshold: Misclassification(label),
                  'alpha0': Alpha0,
                  'diffE': DiffEntropy,
                  'distU': DistUncertainty,
                  }


class MinusAlpha0Predictor(torch.nn.Module):

    def __init__(self, base_detector):
        super(MinusAlpha0Predictor, self).__init__()
        self.base_detector = base_detector

    def forward(self, X):
        return - self.base_detector(X, None, return_output='alpha', compute_loss=False).sum(-1, keepdim=True)


class DiffEPredictor(torch.nn.Module):

    def __init__(self, base_detector):
        super(DiffEPredictor, self).__init__()
        self.base_detector = base_detector

    def forward(self, X):
        eps = 1e-6
        alpha = self.base_detector(X, None, return_output='alpha', compute_loss=False) + eps
        alpha0 = alpha.sum(-1)
        log_term = torch.sum(torch.lgamma(alpha), axis=1) - torch.lgamma(alpha0)
        digamma_term = torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))), axis=1)
        differential_entropy = log_term - digamma_term
        return differential_entropy.unsqueeze(-1)


class DistUPredictor(torch.nn.Module):

    def __init__(self, base_detector):
        super(DistUPredictor, self).__init__()
        self.base_detector = base_detector

    def forward(self, X):
        eps = 1e-6
        alpha = self.base_detector(X, None, return_output='alpha', compute_loss=False) + eps
        alpha0 = alpha.sum(-1)
        probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=1)
        digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0)
        dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        exp_data_uncertainty = -1 * torch.sum(dirichlet_mean * digamma_term, dim=1)
        distributional_uncertainty = total_uncertainty - exp_data_uncertainty
        return distributional_uncertainty.unsqueeze(-1)


UncertaintyPredictor = {'alpha0': lambda model: MinusAlpha0Predictor(model),
                        'diffE': lambda model: DiffEPredictor(model),
                        'distU': lambda model: DistUPredictor(model)}


def compute_X_Y_bounds(model, loader, attack_loss, magnitude):
    if attack_loss == 'crossentropy':
        smoothed_uncertainty_model = SmoothMedianNMS(UncertaintyPredictor['diffE'](model), sigma=max(magnitude, .01), eps=max(magnitude, .01), sample_count=2000)
    else:
        smoothed_uncertainty_model = SmoothMedianNMS(UncertaintyPredictor[attack_loss](model), sigma=max(magnitude, .01), eps=max(magnitude, .01), sample_count=2000)

    for batch_index, (X, Y) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        with torch.no_grad():
            m, u, l = smoothed_uncertainty_model.predict_range(X, batch_size=1000)
        if batch_index == 0:
            l_scores = l.to("cpu")
            m_scores = m.to("cpu")
            u_scores = u.to("cpu")
        else:
            l_scores = torch.cat([l_scores, l.to("cpu")], dim=0)
            m_scores = torch.cat([m_scores, m.to("cpu")], dim=0)
            u_scores = torch.cat([u_scores, u.to("cpu")], dim=0)
    else:
        return l_scores, m_scores, u_scores


def compute_X_Y_attack_bounds(model, loader, attack_name, attack_loss, punished_measure, start_data, magnitude, bounds, compute_alpha=False):
    bounds = [float(bounds[0]), float(bounds[1])]
    fmodel = fb.PyTorchModel(model, bounds=bounds)
    if attack_loss == 'crossentropy':
        smoothed_uncertainty_model = SmoothMedianNMS(UncertaintyPredictor['diffE'](model), sigma=max(magnitude, .01), eps=max(magnitude, .01), sample_count=2000)
    else:
        smoothed_uncertainty_model = SmoothMedianNMS(UncertaintyPredictor[attack_loss](model), sigma=max(magnitude, .01), eps=max(magnitude, .01), sample_count=2000)

    for batch_index, (X, Y) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        X, Y = ep.astensors(X, Y)
        criterion = criterion_dict[attack_loss](Y, start_data, .1)

        if attack_name == 'Gaussian_L2_Noise':
            attack_params = {'model': model, 'X': X.raw, 'Y': Y.raw, 'loss_name': attack_loss, 'epsilons': magnitude, 'start_data': start_data}
            atk_fun = attack_dict[attack_name](attack_params)
            _, pert_X, _ = atk_fun()
            pert_X = ep.astensors(pert_X.unsqueeze(0))
        else:
            kwargs = {}
            attack_params = {'epsilons': [magnitude], 'loss_name': attack_loss, 'punished_measure': punished_measure, 'data_type': start_data}
            atk_fun = attack_dict[attack_name](kwargs)
            _, pert_X, measure = atk_fun(fmodel, X, criterion, **attack_params)

        pert_X = pert_X[0].raw # unpack to get ride of List with one element and PyTorchTensor-type
        Y = Y.raw.to("cpu")
        with torch.no_grad():
            m, u, l = smoothed_uncertainty_model.predict_range(pert_X, batch_size=1000)
            if compute_alpha:
                alpha_pred = model(pert_X)
        if batch_index == 0:
            l_scores = l.to("cpu")
            m_scores = m.to("cpu")
            u_scores = u.to("cpu")
            if compute_alpha:
                Y_all = Y.to("cpu")
                alpha_pred_all = alpha_pred.to("cpu")
        else:
            l_scores = torch.cat([l_scores, l.to("cpu")], dim=0)
            m_scores = torch.cat([m_scores, m.to("cpu")], dim=0)
            u_scores = torch.cat([u_scores, u.to("cpu")], dim=0)
            if compute_alpha:
                Y_all = torch.cat([Y_all, Y.to("cpu")], dim=0)
                alpha_pred_all = torch.cat([alpha_pred_all, alpha_pred.to("cpu")], dim=0)
    if compute_alpha:
        return Y_all, alpha_pred_all, l_scores, m_scores, u_scores
    else:
        return l_scores, m_scores, u_scores


def auroc_apr_scores(binary_class, scores):
    fpr, tpr, thresholds = sk_metrics.roc_curve(binary_class, scores)
    return sk_metrics.auc(fpr, tpr), sk_metrics.average_precision_score(binary_class, scores)


def test_attack_median_smoothing(model, id_test_loader, ood_test_loader, attack_name, attack_loss, punished_measure, start_data, magnitude, bounds, result_path='saved_results'):
    model.to(device)
    model.eval()

    metrics = {}
    if start_data == 'in':
        # Uncertainty scores on clean/perturbed ID/OOD data
        Y, alpha_id_pert, l_uncertainty_score_id_pert, m_uncertainty_score_id_pert, u_uncertainty_score_id_pert = compute_X_Y_attack_bounds(model, id_test_loader, attack_name, attack_loss, punished_measure, start_data, magnitude, bounds, compute_alpha=True)
        l_uncertainty_score_id_clean, m_uncertainty_score_id_clean, u_uncertainty_score_id_clean = compute_X_Y_bounds(model, id_test_loader, attack_loss, magnitude)
        l_uncertainty_score_ood_clean, m_uncertainty_score_ood_clean, u_uncertainty_score_ood_clean = compute_X_Y_bounds(model, ood_test_loader, attack_loss, magnitude)

        # Confidence metric
        metrics['accuracy'] = accuracy(Y=Y, alpha=alpha_id_pert)
        incorrect, correct = (Y.squeeze() == alpha_id_pert.max(-1)[1]).cpu().detach().nonzero(), (Y.squeeze() != alpha_id_pert.max(-1)[1]).cpu().detach().nonzero()
        binary_class = np.concatenate([np.ones(incorrect.size(0)), np.zeros(correct.size(0))], axis=0)
        wc_scores = np.concatenate([l_uncertainty_score_id_pert[incorrect].cpu().detach().numpy(), u_uncertainty_score_id_pert[correct].cpu().detach().numpy()], axis=0)
        metrics[f'wc_confidence_auroc'], metrics[f'wc_confidence_apr'] = auroc_apr_scores(binary_class, wc_scores)
        m_scores = np.concatenate([m_uncertainty_score_id_pert[binary_class.nonzero()].cpu().detach().numpy(), m_uncertainty_score_id_pert[(1 - binary_class).nonzero()].cpu().detach().numpy()], axis=0)
        metrics[f'm_confidence_auroc'], metrics[f'm_confidence_apr'] = auroc_apr_scores(binary_class, m_scores)
        bc_scores = np.concatenate([u_uncertainty_score_id_pert[binary_class.nonzero()].cpu().detach().numpy(), l_uncertainty_score_id_pert[(1 - binary_class).nonzero()].cpu().detach().numpy()], axis=0)
        metrics[f'bc_confidence_auroc'], metrics[f'bc_confidence_apr'] = auroc_apr_scores(binary_class, bc_scores)
        l_scores = np.concatenate([l_uncertainty_score_id_pert[binary_class.nonzero()].cpu().detach().numpy(), l_uncertainty_score_id_pert[(1 - binary_class).nonzero()].cpu().detach().numpy()], axis=0)
        metrics[f'l_confidence_auroc'], metrics[f'l_confidence_apr'] = auroc_apr_scores(binary_class, l_scores)
        u_scores = np.concatenate([u_uncertainty_score_id_pert[binary_class.nonzero()].cpu().detach().numpy(), u_uncertainty_score_id_pert[(1 - binary_class).nonzero()].cpu().detach().numpy()], axis=0)
        metrics[f'u_confidence_auroc'], metrics[f'u_confidence_apr'] = auroc_apr_scores(binary_class, u_scores)

        # Attack detection metric
        binary_class = np.concatenate([np.ones(l_uncertainty_score_id_pert.size(0)), np.zeros(u_uncertainty_score_id_clean.size(0))], axis=0)
        wc_scores = np.concatenate([l_uncertainty_score_id_pert.cpu().detach().numpy(), u_uncertainty_score_id_clean.cpu().detach().numpy()], axis=0)
        metrics[f'wc_attack_detection_auroc'], metrics[f'wc_attack_detection_apr'] = auroc_apr_scores(binary_class, wc_scores)
        m_scores = np.concatenate([m_uncertainty_score_id_pert.cpu().detach().numpy(), m_uncertainty_score_id_clean.cpu().detach().numpy()], axis=0)
        metrics[f'm_attack_detection_auroc'], metrics[f'm_attack_detection_apr'] = auroc_apr_scores(binary_class, m_scores)
        bc_scores = np.concatenate([u_uncertainty_score_id_pert.cpu().detach().numpy(), l_uncertainty_score_id_clean.cpu().detach().numpy()], axis=0)
        metrics[f'bc_attack_detection_auroc'], metrics[f'bc_attack_detection_apr'] = auroc_apr_scores(binary_class, bc_scores)
        l_scores = np.concatenate([l_uncertainty_score_id_pert.cpu().detach().numpy(), l_uncertainty_score_id_clean.cpu().detach().numpy()], axis=0)
        metrics[f'l_attack_detection_auroc'], metrics[f'l_attack_detection_apr'] = auroc_apr_scores(binary_class, l_scores)
        u_scores = np.concatenate([u_uncertainty_score_id_pert.cpu().detach().numpy(), u_uncertainty_score_id_clean.cpu().detach().numpy()], axis=0)
        metrics[f'u_attack_detection_auroc'], metrics[f'u_attack_detection_apr'] = auroc_apr_scores(binary_class, u_scores)

        # OOD metric
        binary_class = np.concatenate([np.ones(l_uncertainty_score_ood_clean.size(0)), np.zeros(u_uncertainty_score_id_pert.size(0))], axis=0)
        wc_scores = np.concatenate([l_uncertainty_score_ood_clean.cpu().detach().numpy(), u_uncertainty_score_id_pert.cpu().detach().numpy()], axis=0)
        metrics[f'wc_ood_auroc'], metrics[f'wc_ood_apr'] = auroc_apr_scores(binary_class, wc_scores)
        m_scores = np.concatenate([m_uncertainty_score_ood_clean.cpu().detach().numpy(), m_uncertainty_score_id_pert.cpu().detach().numpy()], axis=0)
        metrics[f'm_ood_auroc'], metrics[f'm_ood_apr'] = auroc_apr_scores(binary_class, m_scores)
        bc_scores = np.concatenate([u_uncertainty_score_ood_clean.cpu().detach().numpy(), l_uncertainty_score_id_pert.cpu().detach().numpy()], axis=0)
        metrics[f'bc_ood_auroc'], metrics[f'bc_ood_apr'] = auroc_apr_scores(binary_class, bc_scores)
        l_scores = np.concatenate([l_uncertainty_score_ood_clean.cpu().detach().numpy(), l_uncertainty_score_id_pert.cpu().detach().numpy()], axis=0)
        metrics[f'l_ood_auroc'], metrics[f'l_ood_apr'] = auroc_apr_scores(binary_class, l_scores)
        u_scores = np.concatenate([u_uncertainty_score_ood_clean.cpu().detach().numpy(), u_uncertainty_score_id_pert.cpu().detach().numpy()], axis=0)
        metrics[f'u_ood_auroc'], metrics[f'u_ood_apr'] = auroc_apr_scores(binary_class, u_scores)
    elif start_data == 'out':
        # Uncertainty scores on clean/perturbed ID/OOD data
        l_uncertainty_score_id_clean, m_uncertainty_score_id_clean, u_uncertainty_score_id_clean = compute_X_Y_bounds(model, id_test_loader, attack_loss, magnitude)
        l_uncertainty_score_ood_pert, m_uncertainty_score_ood_pert, u_uncertainty_score_ood_pert = compute_X_Y_attack_bounds(model, ood_test_loader, attack_name, attack_loss, punished_measure, start_data, magnitude, bounds)

        # OOD metric
        binary_class = np.concatenate([np.ones(l_uncertainty_score_ood_pert.size(0)), np.zeros(u_uncertainty_score_id_clean.size(0))], axis=0)
        wc_scores = np.concatenate([l_uncertainty_score_ood_pert.cpu().detach().numpy(), u_uncertainty_score_id_clean.cpu().detach().numpy()], axis=0)
        metrics[f'wc_ood_auroc'], metrics[f'wc_ood_apr'] = auroc_apr_scores(binary_class, wc_scores)
        m_scores = np.concatenate([m_uncertainty_score_ood_pert.cpu().detach().numpy(), m_uncertainty_score_id_clean.cpu().detach().numpy()], axis=0)
        metrics[f'm_ood_auroc'], metrics[f'm_ood_apr'] = auroc_apr_scores(binary_class, m_scores)
        bc_scores = np.concatenate([u_uncertainty_score_ood_pert.cpu().detach().numpy(), l_uncertainty_score_id_clean.cpu().detach().numpy()], axis=0)
        metrics[f'bc_ood_auroc'], metrics[f'bc_ood_apr'] = auroc_apr_scores(binary_class, bc_scores)
        l_scores = np.concatenate([l_uncertainty_score_ood_pert.cpu().detach().numpy(), l_uncertainty_score_id_clean.cpu().detach().numpy()], axis=0)
        metrics[f'l_ood_auroc'], metrics[f'l_ood_apr'] = auroc_apr_scores(binary_class, l_scores)
        u_scores = np.concatenate([u_uncertainty_score_ood_pert.cpu().detach().numpy(), u_uncertainty_score_id_clean.cpu().detach().numpy()], axis=0)
        metrics[f'u_ood_auroc'], metrics[f'u_ood_apr'] = auroc_apr_scores(binary_class, u_scores)
    else:
        raise NotImplementedError

    return metrics









