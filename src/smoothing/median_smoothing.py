import torch
from torch import nn
from scipy import stats
import math


def estimated_qu_ql(eps, sample_count, sigma, conf_thres=.99999):
    theo_perc_u = stats.norm.cdf(eps / sigma)
    theo_perc_l = stats.norm.cdf(-eps / sigma)

    q_u_u = sample_count + 1
    q_u_l = math.ceil(theo_perc_u * sample_count)
    q_l_u = math.floor(theo_perc_l * sample_count)
    q_l_l = 0
    q_u_final = q_u_u
    for q_u in range(q_u_l, q_u_u):
        conf = stats.binom.cdf(q_u - 1, sample_count, theo_perc_u)
        if conf > conf_thres:
            q_u_final = q_u
            break
    if q_u_final >= sample_count:
        raise ValueError("Quantile index upper bound is larger than n_samples. Increase n_samples or sigma; "
                         "or reduce eps.")
    q_l_final = q_l_l
    for q_l in range(q_l_u, q_l_l, -1):
        conf = 1-stats.binom.cdf(q_l-1, sample_count, theo_perc_l)
        if conf > conf_thres:
            q_l_final = q_l
            break

    return q_u_final, q_l_final


class SmoothMedianNMS(nn.Module):

    def __init__(self, base_detector, sigma, eps, sample_count, conf_thres=0.99999):
        super(SmoothMedianNMS, self).__init__()
        self.base_detector = base_detector
        self.sigma = self.eps = self.n = self.alpha = self.q_u = self.q_l = None
        self.compute_qu_ql(sigma=sigma, eps=eps, sample_count=sample_count, conf_thres=conf_thres)

    def compute_qu_ql(self, sigma, eps, sample_count, conf_thres=0.99999):
        self.sigma = sigma
        self.eps = eps
        self.n = sample_count
        self.alpha = conf_thres
        self.q_u, self.q_l = estimated_qu_ql(self.eps, self.n, self.sigma, self.alpha)

    def predict_range(self, x: torch.tensor, batch_size: int, seed: int=None) -> (torch.tensor,
                                                                   torch.tensor,
                                                                   torch.tensor):

        if seed is not None:
            torch.cuda.manual_seed(seed)
        all_predictions = []
        n_remaining = self.n
        with torch.no_grad():
            for i in range(math.ceil(n_remaining/batch_size)):
                this_batch_size = min(batch_size, n_remaining)
                n_remaining -= this_batch_size
                if len(x.shape) == 1 or len(x.shape) == 2:
                    batch = x.repeat((this_batch_size, 1))
                else:
                    batch = x.repeat((this_batch_size, 1, 1, 1))

                prediction = self.base_detector(batch + torch.randn_like(batch) * self.sigma)
                all_predictions.append(prediction)

        all_predictions = torch.cat(all_predictions)
        all_predictions = all_predictions.reshape([-1, all_predictions.shape[-1]])
        all_predictions, _ = all_predictions.sort(0)
        median, _ = all_predictions.median(0)
        lower = all_predictions[self.q_l]
        upper = all_predictions[self.q_u]
        return median, upper, lower
