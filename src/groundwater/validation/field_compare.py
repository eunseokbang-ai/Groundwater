"""Compare automatically extracted discontinuity sets against field-measured
joint orientations (scanline mapping / borehole observations) for calibration
and QA, per the project plan's validation module.
"""
import numpy as np
import pandas as pd

from ..discontinuity.geometry import DiscontinuitySet


def load_field_measurements(path: str) -> pd.DataFrame:
    """CSV columns required: dip, dip_direction. Optional: id, x, y, z."""
    df = pd.read_csv(path)
    required = {"dip", "dip_direction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Field measurement CSV is missing columns: {missing}")
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    return df


def dip_dipdir_to_normal(dip_deg: float, dip_direction_deg: float) -> np.ndarray:
    dip = np.radians(dip_deg)
    dip_dir = np.radians(dip_direction_deg)
    return np.array([np.sin(dip) * np.sin(dip_dir), np.sin(dip) * np.cos(dip_dir), np.cos(dip)])


def angular_difference(dip1: float, dipdir1: float, dip2: float, dipdir2: float) -> float:
    """Angle in degrees between two planes (via their normals, sign-independent)."""
    n1 = dip_dipdir_to_normal(dip1, dipdir1)
    n2 = dip_dipdir_to_normal(dip2, dipdir2)
    cos_angle = abs(np.clip(n1 @ n2, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def compare_to_field(discontinuity_sets: list[DiscontinuitySet], field_df: pd.DataFrame) -> pd.DataFrame:
    """For each field measurement, find the nearest auto-extracted set (by
    orientation) and report the angular error. A large error for many
    measurements signals the clustering parameters need retuning.
    """
    rows = []
    for _, row in field_df.iterrows():
        best_set_id, best_error = None, float("inf")
        for dset in discontinuity_sets:
            err = angular_difference(row["dip"], row["dip_direction"], dset.mean_dip, dset.mean_dip_direction)
            if err < best_error:
                best_error, best_set_id = err, dset.set_id
        rows.append(
            {
                "field_id": row["id"],
                "field_dip": row["dip"],
                "field_dip_direction": row["dip_direction"],
                "matched_set_id": best_set_id,
                "angular_error_deg": best_error,
            }
        )
    return pd.DataFrame(rows)


def summarize_errors(comparison: pd.DataFrame) -> dict:
    if comparison.empty:
        return {"n": 0, "mean_error_deg": None, "rmse_deg": None, "max_error_deg": None}
    errors = comparison["angular_error_deg"].to_numpy()
    return {
        "n": len(errors),
        "mean_error_deg": float(errors.mean()),
        "rmse_deg": float(np.sqrt(np.mean(errors**2))),
        "max_error_deg": float(errors.max()),
    }
