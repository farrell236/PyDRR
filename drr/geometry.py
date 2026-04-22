import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DRRGeometry:
    """Single projection geometry definition."""

    source_mm: np.ndarray
    detector_center_mm: np.ndarray
    detector_u_mm: np.ndarray
    detector_v_mm: np.ndarray
    detector_size_px: Tuple[int, int]          # (H, W)
    detector_spacing_mm: Tuple[float, float]   # (row_spacing, col_spacing)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a unit vector."""
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n


def make_detector_basis_from_forward(
    forward_xyz: np.ndarray,
    world_up_xyz: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
) -> Tuple[np.ndarray, np.ndarray]:
    """Build detector basis vectors from source->detector direction."""
    f = normalize(forward_xyz)
    u = np.cross(f, world_up_xyz)

    if np.linalg.norm(u) < 1e-6:
        world_up_xyz = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u = np.cross(f, world_up_xyz)

    u = normalize(u)
    v = normalize(np.cross(u, f))
    return u, v


def make_circular_orbit_pose(
    iso_center_mm: np.ndarray,
    angle_deg: float,
    sid_mm: float = 1000.0,
    idd_mm: float = 500.0,
    detector_size_px: Tuple[int, int] = (512, 512),
    detector_spacing_mm: Tuple[float, float] = (1.0, 1.0),
) -> DRRGeometry:
    """Construct a circular orbit pose in the XY plane around isocenter."""
    theta = math.radians(angle_deg)

    source_offset = np.array([
        sid_mm * math.cos(theta),
        sid_mm * math.sin(theta),
        0.0,
    ], dtype=np.float32)

    source_mm = iso_center_mm + source_offset
    detector_center_mm = iso_center_mm - normalize(source_offset) * idd_mm

    forward = detector_center_mm - source_mm
    detector_u_mm, detector_v_mm = make_detector_basis_from_forward(forward)

    return DRRGeometry(
        source_mm=source_mm,
        detector_center_mm=detector_center_mm,
        detector_u_mm=detector_u_mm,
        detector_v_mm=detector_v_mm,
        detector_size_px=detector_size_px,
        detector_spacing_mm=detector_spacing_mm,
    )


def detector_pixel_centers_world(geom: DRRGeometry) -> np.ndarray:
    """Return detector pixel centers as an array of shape (H, W, 3)."""
    H, W = geom.detector_size_px
    row_spacing, col_spacing = geom.detector_spacing_mm

    rows = np.arange(H, dtype=np.float32) - (H - 1) / 2.0
    cols = np.arange(W, dtype=np.float32) - (W - 1) / 2.0
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    pts = (
        geom.detector_center_mm[None, None, :]
        + rr[..., None] * row_spacing * geom.detector_v_mm[None, None, :]
        + cc[..., None] * col_spacing * geom.detector_u_mm[None, None, :]
    )
    return pts.astype(np.float32)
