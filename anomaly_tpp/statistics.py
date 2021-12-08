from typing import List

import numpy as np
import torch


__all__ = [
    "loglike",
    "chi_squared",
    "ks_arrival",
    "ks_interevent",
    "sum_of_squared_spacings",
]


def _concat_times(times_list: List[np.ndarray]) -> np.ndarray:
    """Concatenate multiple SPP sequences into a single SPP sequences.

    See Appendix D - marked sequences.
    """
    # Lambda_k^*(T) for each mark k = 1, ..., K
    Ts = np.cumsum(np.concatenate([[0]] + [[t[-1]] for t in times_list]))  # (K,)
    return np.concatenate(
        [t[:-1] + Ts[k] for (k, t) in enumerate(times_list)] + [[Ts[-1]]]
    )


def loglike(model=None, batch=None, **kwargs):
    """Compute the log-likelihood score for samples in the dataloader."""
    with torch.no_grad():
        return model.nll_loss(batch).neg().detach().cpu().numpy()


def _ks_stat(x: np.ndarray, sqrt_n=True):
    """Compute the Kolmogorov-Smirnov statistic.

    Args:
        x: Sorted samples from Uniform(0, 1).
    """
    num_obs = len(x)
    if num_obs == 0:
        return 1.0
    else:
        d_plus = (np.arange(1.0, num_obs + 1) / num_obs - x).max()
        d_min = (x - np.arange(0.0, num_obs) / num_obs).max()
        score = max(d_plus, d_min)
        if sqrt_n:
            score *= np.sqrt(num_obs)
        return score


def ks_interevent(poisson_times_per_mark: List[List[np.ndarray]], **kwargs):
    """Check if the transformed inter-event times are sampled from Exponential(1)."""

    def ks_inter_single_sequence(t: np.ndarray, sqrt_n=True):
        """Check if the inter-event times are sampled from Exponential(1)."""
        deltas = np.ediff1d(np.concatenate([[0], t]))
        x = 1 - np.exp(-np.sort(deltas))
        return _ks_stat(x, sqrt_n)

    poisson_times = [_concat_times(times) for times in poisson_times_per_mark]
    return np.array([ks_inter_single_sequence(t) for t in poisson_times])


def ks_arrival(poisson_times_per_mark: List[List[np.ndarray]], **kwargs):
    """Check if transformed arrival times t_i / T are sampled from Uniform(0, 1)."""

    def ks_arrival_single_sequence(t: np.ndarray, sqrt_n=True):
        """Check if arrival times t_i / T are sampled from Uniform(0, 1)."""
        # t_rescaled should consist of IID sampled from Uniform(0, 1)
        x = np.sort(t[:-1] / t[-1])
        return _ks_stat(x, sqrt_n)

    poisson_times = [_concat_times(times) for times in poisson_times_per_mark]
    return np.array([ks_arrival_single_sequence(t) for t in poisson_times])


def sum_of_squared_spacings(poisson_times_per_mark: List[List[np.ndarray]], **kwargs):
    def soss_single_sequence(t: np.ndarray):
        deltas = np.ediff1d(np.concatenate([[0], t]))
        scores = np.linalg.norm(deltas) ** 2
        scores = scores / t[-1]
        return scores

    poisson_times = [_concat_times(times) for times in poisson_times_per_mark]
    return np.array([soss_single_sequence(t) for t in poisson_times])


def chi_squared(
    poisson_times_per_mark: List[List[np.ndarray]], num_buckets=10, **kwargs
):
    def chi_squared_single_sequence(t, num_buckets):
        T = t[-1]
        bucket_size = T / num_buckets
        N_expected = (len(t) - 1) / num_buckets
        if N_expected == 0:
            return 0.0
        else:
            stat = 0.0
            for i in range(num_buckets):
                start = i * bucket_size
                end = (i + 1) * bucket_size
                N = np.sum((t > start) & (t <= end))
                stat += (N - N_expected) ** 2 / N_expected
            return stat

    poisson_times = [_concat_times(times) for times in poisson_times_per_mark]
    return np.array(
        [chi_squared_single_sequence(t, num_buckets=num_buckets) for t in poisson_times]
    )
