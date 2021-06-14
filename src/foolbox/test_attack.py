import torch
import numpy as np
import pickle
import gzip
import eagerpy as ep
import src.foolbox as fb
import src.foolbox.attacks as fa
from src.foolbox.criteria import Misclassification, Alpha0, DiffEntropy, DistUncertainty
from src.metrics.metrics_prior import accuracy, confidence, brier_score, anomaly_detection, diff_entropy, dist_uncertainty


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


def compute_X_Y_alpha(model, loader, alpha_only=False):
    for batch_index, (X, Y) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        with torch.no_grad():
            alpha_pred = model(X)
        if batch_index == 0:
            X_duplicate_all = X.to("cpu")
            orig_Y_all = Y.to("cpu")
            alpha_pred_all = alpha_pred.to("cpu")
        else:
            X_duplicate_all = torch.cat([X_duplicate_all, X.to("cpu")], dim=0)
            orig_Y_all = torch.cat([orig_Y_all, Y.to("cpu")], dim=0)
            alpha_pred_all = torch.cat([alpha_pred_all, alpha_pred.to("cpu")], dim=0)
    if alpha_only:
        return alpha_pred_all
    else:
        return orig_Y_all, X_duplicate_all, alpha_pred_all




def compute_X_Y_alpha_attack(model, loader, attack_name, attack_loss, punished_measure, start_data, magnitude, bounds, alpha_only=False):
    bounds = [float(bounds[0]), float(bounds[1])]
    fmodel = fb.PyTorchModel(model, bounds=bounds)

    for batch_index, (X, Y) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        X, Y = ep.astensors(X, Y)
        criterion = criterion_dict[attack_loss](Y, start_data, .1)

        torch.cuda.reset_max_memory_allocated()

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
        with torch.no_grad():
            alpha_pred = model(pert_X)
        if batch_index == 0:
            pert_X_duplicate_all = pert_X.to("cpu")
            orig_Y_all = Y.raw.to("cpu")
            alpha_pred_all = alpha_pred.to("cpu")
        else:
            pert_X_duplicate_all = torch.cat([pert_X_duplicate_all, pert_X.to("cpu")], dim=0)
            orig_Y_all = torch.cat([orig_Y_all, Y.raw.to("cpu")], dim=0)
            alpha_pred_all = torch.cat([alpha_pred_all, alpha_pred.to("cpu")], dim=0)
    alpha_pred_all[alpha_pred_all != alpha_pred_all] = 10e15
    if alpha_only:
        return alpha_pred_all
    else:
        return orig_Y_all, pert_X_duplicate_all, alpha_pred_all


def test_attack(model, id_test_loader, ood_test_loader, attack_name, attack_loss, punished_measure, start_data, magnitude, bounds, result_path='saved_results'):
    model.to(device)
    model.eval()

    if start_data == 'in':
        orig_Y_all, pert_X_duplicate_all, id_alpha_pred_all = compute_X_Y_alpha_attack(model, id_test_loader, attack_name, attack_loss, punished_measure, start_data, magnitude, bounds)
        clean_id_alpha_pred_all = compute_X_Y_alpha(model, id_test_loader, alpha_only=True)
        ood_alpha_pred_all = compute_X_Y_alpha(model, ood_test_loader, alpha_only=True)
        pert_alpha_pred_all = id_alpha_pred_all
    elif start_data == 'out':
        id_alpha_pred_all = compute_X_Y_alpha(model, id_test_loader, alpha_only=True)
        orig_Y_all, pert_X_duplicate_all, ood_alpha_pred_all = compute_X_Y_alpha_attack(model, ood_test_loader, attack_name, attack_loss, punished_measure, start_data, magnitude, bounds)
        pert_alpha_pred_all = ood_alpha_pred_all
    else:
        raise NotImplementedError

    # Save adversarials with alphas
    n_test_samples = orig_Y_all.size(0)
    full_results_dict = {'Y': orig_Y_all.cpu().detach().numpy(),
                         'X': pert_X_duplicate_all.view(n_test_samples, -1).cpu().detach().numpy(),
                         'alpha': pert_alpha_pred_all.cpu().detach().numpy()}

    # Save metrics
    metrics = {}
    metrics['alpha_0'] = pert_alpha_pred_all.sum(-1).mean().detach().numpy()
    if start_data == 'in':
        metrics['clean_alpha_0'] = clean_id_alpha_pred_all.sum(-1).mean().detach().numpy()
        metrics['accuracy'] = accuracy(Y=orig_Y_all, alpha=id_alpha_pred_all)
        metrics['confidence_aleatoric'] = confidence(Y=orig_Y_all, alpha=id_alpha_pred_all, score_type='APR', uncertainty_type='aleatoric')
        metrics['confidence_epistemic'] = confidence(Y=orig_Y_all, alpha=id_alpha_pred_all, score_type='APR', uncertainty_type='epistemic')
        metrics['confidence_diff_entropy'] = confidence(Y=orig_Y_all, alpha=id_alpha_pred_all, score_type='APR', uncertainty_type='differential_entropy')
        metrics['confidence_dist_uncertainty'] = confidence(Y=orig_Y_all, alpha=id_alpha_pred_all, score_type='APR', uncertainty_type='distribution_uncertainty')
        metrics['brier_score'] = brier_score(Y=orig_Y_all, alpha=id_alpha_pred_all)
        metrics[f'attack_detection_aleatoric_apr'] = anomaly_detection(alpha=clean_id_alpha_pred_all, ood_alpha=id_alpha_pred_all, score_type='APR', uncertainty_type='aleatoric')
        metrics[f'attack_detection_epistemic_apr'] = anomaly_detection(alpha=clean_id_alpha_pred_all, ood_alpha=id_alpha_pred_all, score_type='APR', uncertainty_type='epistemic')
        metrics[f'attack_detection_diff_entropy_apr'] = diff_entropy(alpha=clean_id_alpha_pred_all, ood_alpha=id_alpha_pred_all, score_type='APR')
        metrics[f'attack_detection_dist_uncertainty_apr'] = dist_uncertainty(alpha=clean_id_alpha_pred_all, ood_alpha=id_alpha_pred_all, score_type='APR')

    metrics[f'ood_aleatoric_apr'] = anomaly_detection(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='APR', uncertainty_type='aleatoric')
    metrics[f'ood_epistemic_apr'] = anomaly_detection(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='APR', uncertainty_type='epistemic')
    metrics['diff_entropy_apr'] = diff_entropy(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='APR')
    metrics['dist_uncertainty_apr'] = dist_uncertainty(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='APR')

    metrics[f'ood_aleatoric_auroc'] = anomaly_detection(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='AUROC', uncertainty_type='aleatoric')
    metrics[f'ood_epistemic_auroc'] = anomaly_detection(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='AUROC', uncertainty_type='epistemic')
    metrics['diff_entropy_auroc'] = diff_entropy(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='AUROC')
    metrics['dist_uncertainty_auroc'] = dist_uncertainty(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all, score_type='AUROC')

    return metrics









