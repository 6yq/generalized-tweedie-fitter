#!/usr/bin/env python3
import operator

import numpy as np
from typing import NamedTuple
from scipy.optimize import minimize
from scipy.stats import norm


# =============================================
#     Parameter bound / constraint checks
# =============================================


def isInBound(param, bound):
    """Return True if param lies within bound = (lo, hi), None means unbounded."""
    assert len(bound) == 2, "Bound must be a 2-tuple."
    lo, hi = bound
    if lo is not None and hi is not None:
        assert lo <= hi, "Lower bound must not exceed upper bound."
        return lo <= param <= hi
    if lo is None and hi is None:
        return True
    if lo is None:
        return param < hi
    return param > lo


def isParamsInBound(params, bounds):
    """Return True if every parameter lies within its bound."""
    for p, b in zip(params, bounds):
        if not isInBound(p, b):
            return False
    return True


def isParamsWithinConstraints(args, constraints):
    """Return True if args satisfy all linear constraints.

    Each constraint is a dict::

        {"coeffs": [(idx, coeff), ...], "threshold": rhs, "op": ">"}

    meaning  sum(coeff * args[idx])  op  rhs.
    """
    ops = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
    }
    for c in constraints:
        lhs = sum(coeff * args[idx] for idx, coeff in c["coeffs"])
        if not ops[c.get("op", ">")](lhs, c["threshold"]):
            return False
    return True


# ==============================
#     Numerical integration
# ==============================


def composite_simpson(pdf_slice, interval, sample):
    """Composite Simpson's rule over one bin.

    Parameters
    ----------
    pdf_slice : array-like, length sample+1
    interval : float
        Bin width.
    sample : int
        Number of sub-intervals (must be even).
    """
    result = pdf_slice[0] + pdf_slice[-1]
    result += 4 * sum(pdf_slice[i] for i in range(1, sample, 2))
    result += 2 * sum(pdf_slice[i] for i in range(2, sample, 2))
    return result * interval / 3


# ==============================
#     Bin merging
# ==============================


def merge_bins(hist, y, threshold=5):
    """Merge low-count bins toward the peak from both sides.

    Parameters
    ----------
    hist : ndarray   observed counts
    y    : ndarray   expected counts (same length)
    threshold : int  merge until all bins have counts above this value

    Returns
    -------
    hist_, y_ : ndarray
    """
    hist_ = hist.copy()
    y_ = y.copy()

    while True:
        peak_idx = np.argmax(hist_)
        low = np.where(hist_ <= threshold)[0]
        if low.size == 0:
            break

        idx = low[0]
        merged = False

        if idx < peak_idx:
            hist_ = np.append(
                np.append(hist_[:idx], hist_[idx] + hist_[idx + 1]), hist_[idx + 2 :]
            )
            y_ = np.append(np.append(y_[:idx], y_[idx] + y_[idx + 1]), y_[idx + 2 :])
            merged = True
        elif idx > peak_idx:
            idx = low[-1]
            hist_ = np.append(
                np.append(hist_[: idx - 1], hist_[idx - 1] + hist_[idx]),
                hist_[idx + 1 :],
            )
            y_ = np.append(
                np.append(y_[: idx - 1], y_[idx - 1] + y_[idx]), y_[idx + 1 :]
            )
            merged = True

        if not merged:
            break

    return hist_, y_


# ==============================
#     Extra-parameter registry
# ==============================


class ExtraParam(NamedTuple):
    name: str
    init: float
    bound: tuple  # (lo, hi); None means unbounded on that side


PEDESTAL_PARAMS: list[ExtraParam] = [
    ExtraParam("ped_mean", init=0.0, bound=(None, None)),
    ExtraParam("ped_sigma", init=0.05, bound=(1e-6, None)),
]

# Uncomment and append when threshold effect is added:
# THRESHOLD_PARAMS: list[ExtraParam] = [
#     ExtraParam("thres_center", init=0.08, bound=(0.0,  None)),
#     ExtraParam("thres_scale",  init=0.02, bound=(1e-6, None)),
# ]


# ==============================
#     Auto-initialisation
# ==============================


def compute_init(hist, edges):
    """Estimate pedestal and SPE initial parameters via a 4-Gaussian mixture fit.

    The 4 Gaussians correspond to n = 1, 2, 3, 4 PE, all visible inside the
    histogram window::

        p(q) = sum_{n=1}^{4}  w_n * N(mu_0 + n*G,  sqrt(sigma_0^2 + n*sigma_G^2))

    where mu_0 is the pedestal mean, G is the gain, sigma_0 is the pedestal
    sigma, and sigma_G is the SPE charge resolution.  The pedestal itself
    (n=0) lives outside the window and is not a mixture component — its
    parameters are derived: ped_mean = mu_0 = mu1 - G, ped_sigma = sigma_0.

    Parameters
    ----------
    hist  : ndarray   bin counts
    edges : ndarray   bin edges, length len(hist)+1

    Returns
    -------
    ped_mean  : float   mu_0 = mu1 - G
    ped_sigma : float   sigma_0  (robust estimate: spe_sigma / sqrt(2))
    spe_mean  : float   G
    spe_sigma : float   sigma_G
    """
    from scipy.signal import find_peaks

    xs = (edges[:-1] + edges[1:]) / 2.0
    counts = hist.astype(float)
    if counts.sum() == 0:
        raise ValueError("Histogram is empty.")

    # robust gain estimate: use spacing between the two most prominent peaks
    # distance=3 bins minimum separation; prominence >= 1% of max to filter noise
    peaks, props = find_peaks(
        counts,
        prominence=counts.max() * 0.01,
        distance=max(3, len(counts) // 40),
    )
    if len(peaks) >= 2:
        # sort by prominence descending, take the two tallest, then sort by position
        prom = props["prominences"]
        top2 = np.sort(peaks[np.argsort(prom)[-2:]])
        G_rough = float(xs[top2[1]] - xs[top2[0]])
        mu1_rough = float(xs[top2[0]])
    elif len(peaks) == 1:
        G_rough = float(xs[peaks[0]])
        mu1_rough = float(xs[peaks[0]])
    else:
        G_rough = float(np.average(xs, weights=counts))
        mu1_rough = G_rough

    def _dmix(q, mu1, G, var_G, var_0, w):
        d = np.zeros_like(q, dtype=float)
        for i, n in enumerate(range(1, 5)):
            sigma_n = np.sqrt(max(var_0 + n * var_G, 1e-12))
            d += w[i] * norm.pdf(q, loc=mu1 + (n - 1) * G, scale=sigma_n)
        return d

    def _nll(par):
        mu1, G, var_G, var_0, w1, w2, w3 = par
        w4 = 1.0 - w1 - w2 - w3
        if w4 < 0 or G < 0 or var_G < 0 or var_0 < 0:
            return np.inf
        pdf = _dmix(xs, mu1, G, var_G, var_0, [w1, w2, w3, w4])
        return -float(np.sum(counts * np.log(np.maximum(pdf, 1e-300))))

    p0 = [
        mu1_rough,  # mu1      (1PE peak position)
        G_rough,  # G        (gain)
        (0.20 * G_rough) ** 2,  # var_G    (SPE variance)
        (0.05 * G_rough) ** 2,  # var_0    (pedestal variance)
        0.50,  # w1
        0.30,  # w2
        0.15,  # w3
    ]

    res = minimize(
        _nll,
        p0,
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-4, "fatol": 1e-4},
    )

    mu1, G, var_G, var_0 = res.x[:4]
    spe_mean = float(G)
    spe_sigma = float(np.sqrt(max(var_G, 1e-12)))
    ped_mean = float(mu1 - G)
    ped_sigma = spe_sigma / np.sqrt(2.0)

    print(f"[AUTO] ped_mean={ped_mean:.4g}  ped_sigma={ped_sigma:.4g}", flush=True)
    print(f"[AUTO] spe_mean={spe_mean:.4g}  spe_sigma={spe_sigma:.4g}", flush=True)

    return ped_mean, ped_sigma, spe_mean, spe_sigma


# ==============================
#     Chi-square functions
# ==============================


def merged_pearson_chi2(hist, y, zero, z, dof):
    """Pearson chi-square after merging low-count bins (threshold=5)."""
    y_reg, hist_reg = merge_bins(y, hist, threshold=5)
    ndf = len(hist_reg) - dof
    chi2 = float(np.sum((y_reg - hist_reg) ** 2 / y_reg) + (z - zero) ** 2 / z)
    return chi2, ndf


def modified_neyman_chi2_A(hist, y, zero, z, dof):
    """Modified Neyman chi-square A:  sum(y/n - n)."""
    ndf = len(hist) - dof
    mask = hist > 0
    chi2 = float(
        np.sum(y[mask] / hist[mask] - hist[mask]) + (z / zero - zero if zero > 0 else 0)
    )
    return chi2, ndf


def modified_neyman_chi2_B(hist, y, zero, z, dof):
    """Modified Neyman chi-square B:  sum((y-n)^2 / n)."""
    ndf = len(hist) - dof
    mask = hist > 0
    chi2 = float(
        np.sum((y[mask] - hist[mask]) ** 2 / hist[mask])
        + ((z - zero) ** 2 / zero if zero > 0 else 0)
    )
    return chi2, ndf


def mighell_chi2(hist, y, zero, z, dof):
    """Mighell chi-square:  sum((y+1-n)^2 / (n+1))."""
    ndf = len(hist) - dof
    chi2 = float(
        np.sum((y + 1 - hist) ** 2 / (hist + 1))
        + ((z + 1 - zero) ** 2 / (zero + 1) if zero >= 0 else 0)
    )
    return chi2, ndf
