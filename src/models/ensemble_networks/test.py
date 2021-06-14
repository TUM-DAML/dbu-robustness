import numpy as np
import torch
import pickle
import gzip
from src.metrics.metrics_sampler import accuracy, confidence, brier_score, anomaly_detection


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


def compute_X_Y_mean_var(model, loader, mean_var_only=False):
    for batch_index, (X, Y) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        approx_mean, approx_var = model.ensemble_estimate(X)
        if batch_index == 0:
            X_duplicate_all = X.to("cpu")
            orig_Y_all = Y.to("cpu")
            approx_mean_all = approx_mean.to("cpu")
            approx_var_all = approx_var.to("cpu")
        else:
            X_duplicate_all = torch.cat([X_duplicate_all, X.to("cpu")], dim=0)
            orig_Y_all = torch.cat([orig_Y_all, Y.to("cpu")], dim=0)
            approx_mean_all = torch.cat([approx_mean_all, approx_mean.to("cpu")], dim=0)
            approx_var_all = torch.cat([approx_var_all, approx_var.to("cpu")], dim=0)

    if mean_var_only:
        return approx_mean_all, approx_var_all
    else:
        return orig_Y_all, X_duplicate_all, approx_mean_all, approx_var_all


def test(model, test_loader, ood_dataset_loaders, result_path='saved_results'):
    model.to(device)
    model.eval()

    with torch.no_grad():
        orig_Y_all, X_duplicate_all, approx_mean_all, approx_var_all = compute_X_Y_mean_var(model, test_loader)

        # Save each data result
        n_test_samples = orig_Y_all.size(0)
        full_results_dict = {'Y': orig_Y_all.cpu().detach().numpy(),
                             'X': X_duplicate_all.view(n_test_samples, -1).cpu().detach().numpy(),
                             'mean_p': approx_mean_all.cpu().detach().numpy(),
                             'var_p': approx_var_all.cpu().detach().numpy()}
        # with gzip.open(f'{result_path}.pickle', 'wb') as handle:
        #     pickle.dump(full_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metrics
        metrics = {}
        metrics['accuracy'] = accuracy(Y=orig_Y_all, mean_p=approx_mean_all)
        metrics['confidence_aleatoric_apr'] = confidence(Y=orig_Y_all, mean_p=approx_mean_all, var_p=approx_var_all, score_type='APR', uncertainty_type='aleatoric')
        metrics['confidence_epistemic_apr'] = confidence(Y=orig_Y_all, mean_p=approx_mean_all, var_p=approx_var_all, score_type='APR', uncertainty_type='epistemic')
        metrics['confidence_aleatoric_auroc'] = confidence(Y=orig_Y_all, mean_p=approx_mean_all, var_p=approx_var_all, score_type='AUROC', uncertainty_type='aleatoric')
        metrics['confidence_epistemic_auroc'] = confidence(Y=orig_Y_all, mean_p=approx_mean_all, var_p=approx_var_all, score_type='AUROC', uncertainty_type='epistemic')
        metrics['brier_score'] = brier_score(Y=orig_Y_all, mean_p=approx_mean_all)
        for ood_dataset_name, ood_loader in ood_dataset_loaders.items():
            ood_approx_mean_all, ood_approx_var_all = compute_X_Y_mean_var(model, ood_loader, mean_var_only=True)
            metrics[f'anomaly_detection_aleatoric_{ood_dataset_name}_apr'] = anomaly_detection(mean_p=approx_mean_all, var_p=approx_var_all, ood_mean_p=ood_approx_mean_all, ood_var_p=ood_approx_var_all, score_type='APR', uncertainty_type='aleatoric')
            metrics[f'anomaly_detection_epistemic_{ood_dataset_name}_apr'] = anomaly_detection(mean_p=approx_mean_all, var_p=approx_var_all, ood_mean_p=ood_approx_mean_all, ood_var_p=ood_approx_var_all, score_type='APR', uncertainty_type='epistemic')
            metrics[f'anomaly_detection_aleatoric_{ood_dataset_name}_auroc'] = anomaly_detection(mean_p=approx_mean_all, var_p=approx_var_all, ood_mean_p=ood_approx_mean_all, ood_var_p=ood_approx_var_all, score_type='AUROC', uncertainty_type='aleatoric')
            metrics[f'anomaly_detection_epistemic_{ood_dataset_name}_auroc'] = anomaly_detection(mean_p=approx_mean_all, var_p=approx_var_all, ood_mean_p=ood_approx_mean_all, ood_var_p=ood_approx_var_all, score_type='AUROC', uncertainty_type='epistemic')

    return metrics