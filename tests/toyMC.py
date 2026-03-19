#!/usr/bin/env python3
import numpy as np

from scipy.stats import norm, gamma


# ==============================
#     Constants
# ==============================

SEED = 42
N_EVENTS = 100_000
XI = 0.4  # Gen-Poisson dispersion

PED_MEAN = -600.0
PED_SIGMA = 100.0
SPE_MEAN = 6000.0
SPE_SIGMA = 1000.0


# ==============================
#     Poisson sampling
# ==============================


def sample_poisson_tweedie(
    n_events, lam, ped_mean, ped_sigma, spe_mean, spe_sigma, seed=None
):
    """Draw n_events charges from the Compound-Poisson-Gamma model.

    Each event:
      - Draw k ~ Poisson(lam)
      - charge ~ N(ped_mean, ped_sigma^2) + sum of k independent Gamma SPE draws

    Parameters
    ----------
    n_events  : int
    lam       : float   light intensity = -log(1 - occ)
    ped_mean  : float
    ped_sigma : float
    spe_mean  : float   mean of single Gamma SPE
    spe_sigma : float   std  of single Gamma SPE
    seed      : int or None

    Returns
    -------
    charges : ndarray, shape (n_events,)
    ks      : ndarray, shape (n_events,)   PE counts per event
    """
    rng = np.random.default_rng(seed)
    alpha = (spe_mean / spe_sigma) ** 2
    theta = spe_mean / alpha

    ks = rng.poisson(lam, size=n_events)
    charges = rng.normal(ped_mean, ped_sigma, size=n_events)

    nonzero = np.where(ks > 0)[0]
    total_spe = int(ks[nonzero].sum())
    spe_draws = rng.gamma(shape=alpha, scale=theta, size=total_spe)

    idx = 0
    for i, k in zip(nonzero, ks[nonzero]):
        charges[i] += spe_draws[idx : idx + k].sum()
        idx += k

    return charges, ks


# ==============================
#     Gen-Poisson sampling
# ==============================


def _gen_poisson_pmf(lam, xi, k_max=50):
    """Generalized Poisson PMF up to k_max, normalised.

    p_k = lam * (lam + xi*k)^{k-1} * exp(-lam - xi*k) / k!
    p_0 = exp(-lam)
    """
    ks = np.arange(0, k_max + 1, dtype=float)
    logp = np.where(
        ks == 0,
        -lam,
        np.log(lam)
        + (ks - 1) * np.log(np.maximum(lam + xi * ks, 1e-300))
        - lam
        - xi * ks
        - np.array([float(np.sum(np.log(np.arange(1, int(k) + 1)))) for k in ks]),
    )
    p = np.exp(logp - logp.max())
    p = np.maximum(p, 0.0)
    return p / p.sum()


def sample_genpoisson_tweedie(
    n_events, lam, xi, ped_mean, ped_sigma, spe_mean, spe_sigma, seed=None
):
    """Draw n_events charges from the Compound-Generalized-Poisson-Gamma model.

    Parameters
    ----------
    n_events  : int
    lam       : float   light intensity = -log(1 - occ)
    xi        : float   Gen-Poisson dispersion parameter (0 = plain Poisson)
    ped_mean  : float
    ped_sigma : float
    spe_mean  : float
    spe_sigma : float
    seed      : int or None

    Returns
    -------
    charges : ndarray, shape (n_events,)
    ks      : ndarray, shape (n_events,)   PE counts per event
    """
    rng = np.random.default_rng(seed)
    alpha = (spe_mean / spe_sigma) ** 2
    theta = spe_mean / alpha

    pmf = _gen_poisson_pmf(lam, xi, k_max=50)
    ks = rng.choice(len(pmf), size=n_events, p=pmf)

    charges = rng.normal(ped_mean, ped_sigma, size=n_events)

    nonzero = np.where(ks > 0)[0]
    total_spe = int(ks[nonzero].sum())
    spe_draws = rng.gamma(shape=alpha, scale=theta, size=total_spe)

    idx = 0
    for i, k in zip(nonzero, ks[nonzero]):
        charges[i] += spe_draws[idx : idx + k].sum()
        idx += k

    return charges, ks
