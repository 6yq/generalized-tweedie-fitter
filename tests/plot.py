import math

import numpy as np
import matplotlib.pyplot as plt


def plot_histogram_with_fit(
    bins,
    hist,
    xsp,
    smooth,
    comps,
    labels,
    params,
    ch=None,
    gp=None,
    gp_std=None,
    gm=None,
    gm_std=None,
    occ=None,
    occ_std=None,
    chiSq=None,
    ndf=None,
    zero=None,
    zeroReal=None,
    ys=None,
    comp_colors=None,
    comp_styles=None,
    logscale=False,
    ax_main=None,
    ax_resid=None,
    fig=None,
):
    if ax_main is None:
        fig, ax_main = plt.subplots()

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # ==============================
    #     Occupancy label
    # ==============================

    if occ is not None and occ < 0.01:
        exp = int(np.floor(np.log10(occ)))
        mu = -math.log1p(-occ)
        mu_std = occ_std / (1.0 - occ)
        occ_str = rf"({mu/10**exp:.3f}$\pm${mu_std/10**exp:.3f})" rf"$\times10^{exp}$"
    elif occ is not None:
        mu = -math.log1p(-occ)
        mu_std = occ_std / (1.0 - occ)
        occ_str = rf"{mu:.3f}$\pm${mu_std:.3f}"
    else:
        occ_str = ""

    # ==============================
    #     Histogram
    # ==============================

    ax_main.stairs(
        hist, bins, fill=True, color="C0", alpha=0.2, label=rf"$\mu$ = {occ_str}"
    )
    ax_main.errorbar(
        bin_centers,
        hist,
        yerr=np.sqrt(hist),
        marker="o",
        linestyle="none",
        ecolor="black",
        mec="black",
        mfc="red",
        ms=3,
    )

    # ==============================
    #     Overall fit curve
    # ==============================

    ax_main.plot(
        xsp,
        smooth,
        color="red",
        label=(f"$\\chi^2$/ndf = {chiSq:.2f}/{ndf}" if chiSq is not None else "fit"),
    )

    # ==============================
    #     Gain marker
    # ==============================

    if len(params) == 6 and gp is not None:
        ax_main.axvline(
            gp,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=(
                f"Gp={gp:.2f}$\\pm${gp_std:.2f}"
                if gp_std is not None
                else f"Gp={gp:.2f}"
            ),
        )
        ax_main.axvline(
            gm,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=(
                f"Gm={gm:.2f}$\\pm${gm_std:.2f}"
                if gm_std is not None
                else f"Gm={gm:.2f}"
            ),
        )
    elif gm is not None:
        ax_main.axvline(
            gm,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=(
                f"G={gm:.2f}$\\pm${gm_std:.2f}" if gm_std is not None else f"G={gm:.2f}"
            ),
        )

    # ==============================
    #     Per-PE components
    # ==============================

    _default_colors = [
        "#888888",
        "#2196F3",
        "#4CAF50",
        "#FF9800",
        "#E91E63",
        "#9C27B0",
        "#00BCD4",
        "#FF5722",
    ]
    _default_styles = ["--", "-.", ":", (0, (3, 1, 1, 1)), "--", "-.", ":"]

    colors = comp_colors if comp_colors is not None else _default_colors
    styles = comp_styles if comp_styles is not None else _default_styles

    for i, (comp, lbl) in enumerate(zip(comps, labels)):
        ax_main.plot(
            xsp,
            comp,
            color=colors[i % len(colors)],
            linestyle=styles[i % len(styles)],
            alpha=0.8,
            label=lbl,
        )

    # ==============================
    #     Axes formatting
    # ==============================

    if logscale:
        ax_main.set_yscale("log")

    ax_main.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_main.set_ylabel("Entries")
    ax_main.legend(frameon=False)
    if ch is not None:
        ax_main.set_title(f"ch {ch}")

    # ==============================
    #     Residual panel
    # ==============================

    if ax_resid is not None and ys is not None:
        ax_resid.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax_resid.plot(bin_centers, hist - ys, "o", color="black", ms=3)
        ax_resid.grid(True, which="both", linestyle="--", alpha=0.3)
        ax_resid.set_ylabel("Residual")
        ax_resid.set_xlabel("Q")
        ax_main.label_outer()
    else:
        ax_main.set_xlabel("Q")

    return fig, ax_main, ax_resid
