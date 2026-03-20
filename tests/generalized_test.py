#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from ..models import Gen_Tweedie_Fitter
from .plot import plot_histogram_with_fit
from .toyMC import (
    sample_genpoisson_tweedie,
    N_EVENTS,
    XI,
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


def _make_figure():
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 0.15], hspace=0.05)
    return fig, fig.add_subplot(gs[0]), fig.add_subplot(gs[1])


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


def fit_and_plot_genpoisson(charges, lam, pp):
    hist, bins = _build_hist(charges)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fit = Gen_Tweedie_Fitter(
        hist=hist,
        bins=bins,
        A=N_EVENTS,
        lam_init=lam,
        q_min=charges.min(),
        auto_init=True,
        seterr="ignore",
    )
    fit.fit(tol=1e-3, max_calls=10000)

    n_max = _n_max(fit.lam)
    comps, lbls = _npe_components(fit, n_max)

    fig, ax_main, ax_resid = _make_figure()
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
        fig=fig,
    )
    ax_main.set_title(
        f"Generalized Tweedie (Gen-Poisson)  $\\lambda_{{\\rm true}}={lam}$"
    )
    pp.savefig(fig)
    plt.close(fig)


# ==============================
#     Main
# ==============================


def main():
    with PdfPages("fit_generalized_tweedie.pdf") as pp:
        for lam in LAMS:
            charges, _ = sample_genpoisson_tweedie(
                N_EVENTS, lam, XI, PED_MEAN, PED_SIGMA, SPE_MEAN, SPE_SIGMA, seed=SEED
            )
            fit_and_plot_genpoisson(charges, lam, pp)


if __name__ == "__main__":
    main()
