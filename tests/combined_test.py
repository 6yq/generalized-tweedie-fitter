#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from ..core import CombinedFitter
from ..core.utils import modified_neyman_chi2_B
from ..models import Gen_Tweedie_Fitter
from .plot import plot_histogram_with_fit
from .tweedie_test import (
    _make_figure,
    _n_max,
    _build_hist,
)
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


def _make_fitter(charges, lam):
    hist, bins = _build_hist(charges, BIN_LO, BIN_HI, BIN_WIDTH)
    return (
        Gen_Tweedie_Fitter(
            hist=hist,
            bins=bins,
            A=N_EVENTS,
            lam_init=lam,
            q_min=charges.min(),
            auto_init=True,
            seterr="ignore",
        ),
        hist,
        bins,
    )


def _plot_one(combined, fit_i, hist, bins, lam_true, ch_label, pp):
    """Plot one spectrum's result using shared SER from combined fit."""
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = float(bins[1] - bins[0])

    fit_obj = combined.fitters[fit_i]
    local_args = combined._build_local_args(combined.fitted_params, fit_i)

    # inject local_args into fitter's full_args so estimate_count_n works
    fit_obj.full_args = local_args
    fit_obj.full_args_std = combined._build_local_args(combined.param_errors, fit_i)

    xsp_plot = fit_obj.xsp[abs(fit_obj._shift) :]
    smooth_plot = fit_obj._estimate_smooth(local_args)[abs(fit_obj._shift) :]
    ys, z_est = fit_obj._estimate_count(local_args)

    lam_fitted = combined.lams[fit_i]
    lam_std = combined.lams_std[fit_i]

    # ==============================
    #     Per-spectrum chi2 / ndf
    # ==============================
    # ndf_i = (n_bins + 1 zero bin) - 2 per-spectrum params (logA_i, lam_i)
    # The shared pedestal+SER params are NOT subtracted here: they are paid
    # for once by the joint fit and contribute fractionally to each spectrum.
    # Showing ndf = n_bins - 1 is an approximation that makes the per-plot
    # chi2/ndf interpretable as a goodness-of-fit indicator for this spectrum.
    chi2, _ = modified_neyman_chi2_B(hist, ys, fit_obj.zero, z_est, dof=0)
    ndf_i = len(hist) + 1 - 2

    n = _n_max(lam_fitted)
    comps_binned = [fit_obj.estimate_count_n(k) for k in range(1, n + 1)]
    lbls = [f"{k} PE" for k in range(1, n + 1)]

    spe_mean = float(combined.ser_args[0])
    spe_mean_std = float(combined.ser_args_std[0])
    spe_sigma = float(combined.ser_args[1])
    spe_sigma_std = float(combined.ser_args_std[1])
    spe_res = spe_sigma / spe_mean * 100.0
    spe_res_std = spe_res * np.sqrt(
        (spe_sigma_std / spe_sigma) ** 2 + (spe_mean_std / spe_mean) ** 2
    )

    fig, ax_main, ax_resid, ax_leg = _make_figure(n_comps=len(comps_binned))
    plot_histogram_with_fit(
        bins=bins,
        hist=hist,
        xsp=xsp_plot,
        smooth=smooth_plot,
        bin_centers=bin_centers,
        comps=comps_binned,
        labels=lbls,
        params=combined.ser_args,
        occ=1 - np.exp(-lam_fitted),
        occ_std=lam_std * np.exp(-lam_fitted),
        ped_mean=combined.extra_args[0],
        gm=spe_mean,
        gm_std=spe_mean_std,
        spe_sigma=spe_sigma,
        spe_sigma_std=spe_sigma_std,
        spe_res=spe_res,
        spe_res_std=spe_res_std,
        chiSq=chi2,
        ndf=ndf_i,
        ys=ys,
        logscale=False,
        ax_main=ax_main,
        ax_resid=ax_resid,
        ax_leg=ax_leg,
        fig=fig,
    )
    ax_main.set_title(f"{ch_label} $\\lambda_{{\\rm true}}={lam_true}$")
    pp.savefig(fig)
    plt.close(fig)


# ==============================
#     Main
# ==============================


def main():
    # sample all spectra
    all_charges = []
    for lam in LAMS:
        charges, _ = sample_genpoisson_tweedie(
            N_EVENTS, lam, XI, PED_MEAN, PED_SIGMA, SPE_MEAN, SPE_SIGMA, seed=SEED
        )
        all_charges.append(charges)

    # build individual fitters
    fitters = []
    hists = []
    bins_list = []
    for charges, lam in zip(all_charges, LAMS):
        fit, hist, bins = _make_fitter(charges, lam)
        fitters.append(fit)
        hists.append(hist)
        bins_list.append(bins)

    # combined fit
    combined = CombinedFitter(fitters)
    combined.fit(tol=1e-3, max_calls=5000)

    # plot one page per spectrum
    with PdfPages("fit_combined.pdf") as pp:
        for i, (hist, bins, lam) in enumerate(zip(hists, bins_list, LAMS)):
            _plot_one(
                combined,
                i,
                hist,
                bins,
                lam,
                f"Combined Gen-Tweedie {i+1}/{len(LAMS)}",
                pp,
            )


if __name__ == "__main__":
    main()
