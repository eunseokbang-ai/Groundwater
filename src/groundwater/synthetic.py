"""Synthetic quarry wall with known joint sets, for tests and the demo dataset.

The wall is a rough, slightly leaning surface unrolled along a plan polyline
(the excavated face itself -- expect it to come out as one set per wall
section). Each joint is a rectangular face of fixed orientation protruding from
the wall, with the wall darkened along the joint's trace, the way an open joint
looks in a photograph.
"""
from dataclasses import dataclass

import numpy as np

from .facade.frame import WallFrame
from .validation.field_compare import dip_dipdir_to_normal

WALL_GRAY = 0.6
JOINT_BROWN = np.array([0.55, 0.42, 0.30])
TRACE_DARK = 0.12


@dataclass
class SyntheticJoint:
    dip: float
    dip_direction: float
    chainage: float  # where the trace crosses the wall (centre), metres along the polyline
    elevation: float
    length: float = 2.0  # along the trace
    depth: float = 0.4  # how far the face protrudes toward the viewer


def synthetic_wall(
    rng: np.random.Generator,
    polyline,
    joints: list[SyntheticJoint],
    base_elevation: float = 100.0,
    height: float = 8.0,
    spacing: float = 0.04,
    lean_deg: float = 3.0,
    roughness: float = 0.003,
    trace_halfwidth: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (points, colors) as (N, 3) arrays in world coordinates."""
    frame = WallFrame(polyline)
    u = np.arange(0.0, frame.length, spacing)
    v = np.arange(base_elevation, base_elevation + height, spacing)
    gu, gv = (a.ravel() for a in np.meshgrid(u, v))
    gw = -(gv - base_elevation) * np.tan(np.radians(lean_deg)) + rng.normal(scale=roughness, size=gu.shape)
    wall = frame.to_world(np.column_stack([gu, gv, gw]))
    wall_colors = np.clip(WALL_GRAY + rng.normal(scale=0.03, size=(len(wall), 1)), 0, 1).repeat(3, axis=1)

    joint_points, joint_colors = [], []
    for joint in joints:
        normal = dip_dipdir_to_normal(joint.dip, joint.dip_direction)
        w0 = -(joint.elevation - base_elevation) * np.tan(np.radians(lean_deg))
        centre = frame.to_world(np.array([[joint.chainage, joint.elevation, w0]]))[0]
        _, wall_normal = frame.axes_at(joint.chainage)

        along = np.cross(normal, wall_normal)
        along /= np.linalg.norm(along)
        outward = np.cross(normal, along)
        if outward @ wall_normal < 0:
            outward = -outward

        a = np.arange(-joint.length / 2, joint.length / 2, spacing)
        b = np.arange(0.0, joint.depth, spacing)
        ga, gb = (x.ravel() for x in np.meshgrid(a, b))
        face = centre + ga[:, None] * along + gb[:, None] * outward
        joint_points.append(face + rng.normal(scale=roughness, size=face.shape))
        joint_colors.append(np.clip(JOINT_BROWN + rng.normal(scale=0.03, size=(len(face), 3)), 0, 1))

        rel = wall - centre
        along_dist = rel @ along
        off_line = rel - along_dist[:, None] * along
        on_trace = (np.abs(along_dist) <= joint.length / 2) & (np.linalg.norm(off_line, axis=1) <= trace_halfwidth)
        wall_colors[on_trace] = TRACE_DARK

    points = np.vstack([wall, *joint_points])
    colors = np.vstack([wall_colors, *joint_colors])
    return points, colors
