"""View '3D': the point cloud coloured by joint set."""
import numpy as np
import plotly.graph_objects as go

from groundwater import report
from groundwater.viz.dashboard.components import _view3d
from groundwater.viz.dashboard.state import _result_key, _session_cache


MAX_3D_POINTS = 80_000


def _downsample_for_display(points: np.ndarray, colors: np.ndarray, max_points: int):
    """The same sample every rerun (a new random one each time would redraw the chart and reset the camera)."""
    if len(points) <= max_points:
        return points, colors
    idx = np.sort(np.random.default_rng(0).choice(len(points), max_points, replace=False))
    return points[idx], colors[idx]


def render_3d_view(result):
    # the same figure object every rerun of this result: the 3D page is then built once and not sent again
    fig = _session_cache("_3d_fig", _result_key(), lambda: _main_3d_figure(result))
    _view3d(fig, "v3d_main", 700)


def _main_3d_figure(result):
    display_points, display_set_ids = _downsample_for_display(np.asarray(result.point_cloud.points),
                                                              result.point_labels[:, 0], MAX_3D_POINTS)
    fig = go.Figure()
    unassigned = display_set_ids == -1
    if unassigned.any():
        p = display_points[unassigned]
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                                   marker=dict(size=1.5, color="lightgray"), name="미분류"))
    for sid in sorted(s for s in set(display_set_ids.tolist()) if s != -1):
        p = display_points[display_set_ids == sid]
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                                   marker=dict(size=2, color=report.SET_COLORS[sid % len(report.SET_COLORS)]),
                                   name=f"Set {sid}"))
    _, outward = result.wall_frame.axes_at(result.wall_frame.length / 2)
    eye = dict(x=float(outward[0]) * 1.6, y=float(outward[1]) * 1.6, z=0.5)      # in front of the wall, a little above
    fig.update_layout(
        scene=dict(aspectmode="data", camera=dict(eye=eye), xaxis_title="동 X (m)", yaxis_title="북 Y (m)",
                   zaxis_title="표고 Z (m)"),
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(itemsizing="constant"),
    )
    return fig
