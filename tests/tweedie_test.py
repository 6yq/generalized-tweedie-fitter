#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

from ..models import Tweedie_Fitter
from .plot import plot_histogram_with_fit
from .toyMC import (
    sample_poisson_tweedie,
    N_EVENTS,
    PED_MEAN,
    PED_SIGMA,
    SPE_MEAN,
    SPE_SIGMA,
    SEED,
)

plt.style.use("tests/matplotlibrc")


# ==============================
#     Constants
# ==============================

BIN_WIDTH = 100.0
BIN_LO = 1000.0
BIN_HI = 50000.0
LAMS = [0.5, 0.8, 1.5, 2.0, 3.0]


# ==============================
#     Helpers
# ==============================


def _make_figure(n_comps=4):
    # legend rows needed for 3-column layout
    n_leg_rows = (n_comps + 2) // 3
    # each legend row ~0.04 in height ratio units; residual fixed at 0.10
    leg_ratio = n_leg_rows * 0.06
    resid_ratio = 0.10
    main_ratio = 1.0 - leg_ratio - resid_ratio
    # expand figure height proportionally so main plot stays a good size
    fig_height = 6 + n_leg_rows * 0.35

    fig = plt.figure(figsize=(plt.rcParams["figure.figsize"][0], fig_height))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[main_ratio, resid_ratio, leg_ratio],
        hspace=0.05,
    )
    ax_main = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1], sharex=ax_main)
    ax_leg = fig.add_subplot(gs[2])
    ax_leg.axis("off")
    return fig, ax_main, ax_resid, ax_leg


def _npe_components(fit, n_max):
    """Return per-bin count arrays and labels from 1PE up to n_max PE."""
    comps = [fit.estimate_count_n(n) for n in range(1, n_max + 1)]
    labels = [f"{n} PE" for n in range(1, n_max + 1)]
    return comps, labels


_COMP_COLORS = [
    "#2196F3",
    "#4CAF50",
    "#FF9800",
    "#E91E63",
    "#9C27B0",
    "#00BCD4",
    "#FF5722",
]
_COMP_STYLES = ["-.", ":", "--", "-.", ":", "--", "-."]


def _n_max(lam, threshold=0.01):
    """Largest n where p_n >= threshold * p_mode."""
    from math import exp, factorial

    mode = max(0, int(lam))
    p_mode = exp(-lam) * lam**mode / factorial(mode)
    n = mode
    while True:
        n += 1
        p_n = exp(-lam) * lam**n / factorial(n)
        if p_n < threshold * p_mode:
            return n - 1


def _build_hist(charges):
    bins = np.arange(BIN_LO, BIN_HI + BIN_WIDTH, BIN_WIDTH)
    hist, _ = np.histogram(charges[charges >= BIN_LO], bins=bins)
    return hist, bins


# ==============================
#     Fit and plot
# ==============================


def fit_and_plot_poisson(charges, lam, pp):
    hist, bins = _build_hist(charges)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fit = Tweedie_Fitter(
        hist=hist,
        bins=bins,
        A=N_EVENTS,
        lam_init=lam,
        q_min=charges.min(),
        auto_init=True,
        seterr="ignore",
    )
    fit.fit(tol=1e-3, max_calls=20000)

    n_max = _n_max(fit.lam)
    comps, lbls = _npe_components(fit, n_max)

    fig, ax_main, ax_resid, ax_leg = _make_figure(n_comps=len(comps))
    plot_histogram_with_fit(
        bins=bins,
        hist=hist,
        xsp=bin_centers,
        smooth=fit.ys,
        comps=comps,
        labels=lbls,
        comp_colors=_COMP_COLORS[: len(comps)],
        comp_styles=_COMP_STYLES[: len(comps)],
        params=fit.ser_args,
        occ=1 - np.exp(-fit.lam),
        occ_std=fit.lam_std * np.exp(-fit.lam),
        gm=fit.extra_args[0] + fit.gms,
        gm_std=fit.ser_args_std[0],
        chiSq=fit.chi_sq_neyman_B,
        ndf=fit.ndf,
        ys=fit.ys,
        logscale=False,
        ax_main=ax_main,
        ax_resid=ax_resid,
        ax_leg=ax_leg,
        fig=fig,
    )
    ax_main.set_title(f"Tweedie (Poisson)  $\\lambda_{{\\rm true}}={lam}$")
    pp.savefig(fig)
    plt.close(fig)


# ==============================
#     Main
# ==============================


def main():
    with PdfPages("fit_tweedie.pdf") as pp:
        for lam in tqdm(LAMS, desc="Fitting multiple intensities"):
            charges, _ = sample_poisson_tweedie(
                N_EVENTS, lam, PED_MEAN, PED_SIGMA, SPE_MEAN, SPE_SIGMA, seed=SEED
            )
            fit_and_plot_poisson(charges, lam, pp)


if __name__ == "__main__":
    main()
