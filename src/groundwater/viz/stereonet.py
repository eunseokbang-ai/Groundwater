"""Equal-area stereonet plots of discontinuity poles, colored by set."""
import matplotlib.pyplot as plt
import mplstereonet
import numpy as np

from ..discontinuity.geometry import DiscontinuitySet



def plot_stereonet(
    discontinuity_sets: list[DiscontinuitySet],
    field_dips: np.ndarray | None = None,
    field_dip_dirs: np.ndarray | None = None,
):
    """Return a matplotlib Figure with poles per patch, mean-plane great circles
    per set, and (optionally) field-measured poles overlaid as black crosses
    for visual validation against ground truth.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="stereonet")

    from ..report import SET_COLORS             # the set colours of every other view (set map, 3D, sections)

    for dset in discontinuity_sets:
        color = SET_COLORS[dset.set_id % len(SET_COLORS)]
        dips = np.array([p.dip for p in dset.patches])
        dip_dirs = np.array([p.dip_direction for p in dset.patches])
        strikes = (dip_dirs - 90.0) % 360.0
        ax.pole(strikes, dips, marker="o", color=color, markersize=4, label=f"Set {dset.set_id} (n={len(dips)})")
        ax.plane(
            (dset.mean_dip_direction - 90.0) % 360.0,
            dset.mean_dip,
            color=color,
            linewidth=2,
        )

    if field_dips is not None and field_dip_dirs is not None and len(field_dips):
        # mplstereonet mutates its dip/strike arrays in place, so pass writable copies
        # (a pandas .to_numpy() array can come back read-only).
        field_dips = np.array(field_dips, dtype=float, copy=True)
        field_dip_dirs = np.array(field_dip_dirs, dtype=float, copy=True)
        field_strikes = (field_dip_dirs - 90.0) % 360.0
        ax.pole(field_strikes, field_dips, marker="+", color="black", markersize=10, label="Field measured")

    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=8)
    fig.tight_layout()
    return fig
