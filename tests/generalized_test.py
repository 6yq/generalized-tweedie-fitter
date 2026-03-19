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
LAM = 0.7


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


def _n_max(lam):
    return int(2.75 + 3 * lam)


def _build_hist(charges):
    bins = np.arange(BIN_LO, BIN_HI + BIN_WIDTH, BIN_WIDTH)
    hist, _ = np.histogram(charges[charges >= BIN_LO], bins=bins)
    return hist, bins


# ==============================
#     Fit and plot
# ==============================


def fit_and_plot_genpoisson(charges, pp):
    hist, bins = _build_hist(charges)

    fit = Gen_Tweedie_Fitter(
        hist=hist,
        bins=bins,
        A=N_EVENTS,
        lam_init=LAM,
        auto_init=True,
        seterr="ignore",
    )
    fit.fit(tol=1e-3, max_calls=20000)

    n_max = _n_max(fit.lam)
    comps, lbls = _npe_components(fit, n_max)

    n_max = _n_max(fit.lam)
    comps, lbls = _npe_components(fit, n_max)
    bin_centers = (bins[:-1] + bins[1:]) / 2

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
    ax_main.set_title("Generalized Tweedie (Gen-Poisson)")
    pp.savefig(fig)
    plt.close(fig)


# ==============================
#     Main
# ==============================


def main():
    charges, _ = sample_genpoisson_tweedie(
        N_EVENTS, LAM, XI, PED_MEAN, PED_SIGMA, SPE_MEAN, SPE_SIGMA, seed=SEED
    )

    with PdfPages("fit_generalized_tweedie.pdf") as pp:
        fit_and_plot_genpoisson(charges, pp)


if __name__ == "__main__":
    main()
