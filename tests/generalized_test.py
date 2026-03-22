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
from .tweedie_test import (
    _make_figure,
    _npe_components,
    _n_max,
    _build_hist,
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

    fig, ax_main, ax_resid, ax_leg = _make_figure()
    plot_histogram_with_fit(
        bins=bins,
        hist=hist,
        xsp=bin_centers,
        smooth=fit.ys,
        comps=comps,
        labels=lbls,
        params=fit.ser_args,
        occ=1 - np.exp(-fit.lam),
        occ_std=fit.lam_std * np.exp(-fit.lam),
        ped_mean=fit.extra_args[0],
        gm=fit.ser_args[0],
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
    ax_main.set_title(f"Generalized Tweedie $\\lambda_{{\\rm true}}={lam}$")
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
