from typing import Optional

import numpy as np

from .geometry import DRRGeometry
from .volume import Volume


def normalize_image(
    img: np.ndarray,
    invert: bool = True,
    p_lo: float = 1.0,
    p_hi: float = 99.5,
) -> np.ndarray:
    """Robust normalization for visualization and PNG export."""
    x = img.astype(np.float32)
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)

    if hi <= lo:
        lo = float(x.min())
        hi = float(x.max())

    x = np.clip(x, lo, hi)
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    if invert:
        x = 1.0 - x
    return x


def save_png(path: str, img: np.ndarray, invert: bool = True) -> None:
    """Save a DRR as an 8-bit PNG."""
    import imageio.v2 as imageio

    x = normalize_image(img, invert=invert)
    imageio.imwrite(path, (x * 255).astype(np.uint8))


def print_volume_debug(vol: Volume) -> None:
    """Print basic volume metadata and intensity range."""
    print("Loaded volume")
    print(f"  shape_zyx         : {vol.shape_zyx}")
    print(f"  spacing_zyx (mm)  : {vol.spacing_zyx}")
    print(f"  origin_zyx (mm)   : {vol.origin_zyx}")
    print(f"  direction         :\n{vol.direction}")
    print(f"  intensity range   : [{vol.data.min():.3f}, {vol.data.max():.3f}]")
    print()


def print_geometry_debug(geom: DRRGeometry, iso_center_mm: Optional[np.ndarray] = None) -> None:
    """Print detector and source geometry diagnostics."""
    H, W = geom.detector_size_px
    row_spacing, col_spacing = geom.detector_spacing_mm

    det_h_mm = H * row_spacing
    det_w_mm = W * col_spacing
    sdd = np.linalg.norm(geom.detector_center_mm - geom.source_mm)

    print("DRR geometry")
    print(f"  source_mm         : {geom.source_mm}")
    print(f"  detector_center   : {geom.detector_center_mm}")
    print(f"  detector_u        : {geom.detector_u_mm}")
    print(f"  detector_v        : {geom.detector_v_mm}")
    print(f"  detector_size_px  : {geom.detector_size_px}")
    print(f"  detector_spacing  : {geom.detector_spacing_mm}")
    print(f"  detector_size_mm  : (H={det_h_mm:.2f}, W={det_w_mm:.2f})")
    print(f"  source-detector distance (mm): {sdd:.2f}")

    if iso_center_mm is not None:
        sid = np.linalg.norm(geom.source_mm - iso_center_mm)
        idd = np.linalg.norm(geom.detector_center_mm - iso_center_mm)
        mag = sdd / sid if sid > 0 else np.nan
        print(f"  source-isocenter distance (mm): {sid:.2f}")
        print(f"  iso-detector distance (mm)    : {idd:.2f}")
        print(f"  magnification approx          : {mag:.3f}")
        print(f"  approx iso-plane FOV mm       : (H={det_h_mm / mag:.2f}, W={det_w_mm / mag:.2f})")
    print()


def print_projection_stats(drr: np.ndarray, name: str = "DRR") -> None:
    """Print summary statistics for a rendered projection."""
    print(f"{name} stats")
    print(f"  shape             : {drr.shape}")
    print(f"  min               : {drr.min():.6f}")
    print(f"  max               : {drr.max():.6f}")
    print(f"  mean              : {drr.mean():.6f}")
    print(f"  p1 / p99          : {np.percentile(drr, 1):.6f} / {np.percentile(drr, 99):.6f}")
    print()
