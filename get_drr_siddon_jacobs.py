#!/usr/bin/env python3
"""
Python CLI inspired by the ITK-based `getDRRSiddonJacobsRayTracing` binary.

This script renders a single DRR from a 3D CT/CBCT volume using the refactored
`drr` package. Rendering and multiprocessing live in the core codebase
(`drr.renderer`), and this file is intentionally a thin CLI wrapper.

Notes
-----
- The current projector assumes the volume is axis aligned and ignores the
  SimpleITK direction matrix.
- `-res` is interpreted as DRR pixel spacing in the *isocenter plane*, matching
  the original binary help text. Internally this is converted to detector-plane
  spacing using magnification.
- `-iso` is interpreted in continuous voxel indices as x y z.
- `-2dcx` is interpreted as detector central-axis position in continuous pixel
  indices as col row.
- `-t`, `-rx`, `-ry`, `-rz` are implemented as a rigid transform of the volume
  relative to the scanner. Equivalently, the script applies the inverse transform
  to source and detector geometry.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from drr.volume import (
    load_volume_sitk,
    volume_center_world_xyz,
    voxel_zyx_to_world_xyz,
)
from drr.geometry import DRRGeometry
from drr.renderer import generate_drr
from drr.projector import ray_integral_siddon_jacobs
from drr.visualize import print_projection_stats, print_volume_debug, normalize_image


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------


def rotation_matrix_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float64)



def rotation_matrix_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)



def rotation_matrix_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)



def compose_rotation_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Compose x, y, z intrinsic-style rotations as Rz @ Ry @ Rx."""
    return rotation_matrix_z(rz_deg) @ rotation_matrix_y(ry_deg) @ rotation_matrix_x(rx_deg)



def apply_inverse_volume_pose_to_point(
    point_xyz: np.ndarray,
    iso_center_xyz: np.ndarray,
    translation_xyz: np.ndarray,
    rotation_xyz: np.ndarray,
) -> np.ndarray:
    """
    Apply the inverse of the volume rigid transform to a world-space point.

    If the volume is transformed as:
        p' = iso + t + R @ (p - iso)
    then relative geometry in the volume frame is obtained with:
        p = iso + R^T @ (p' - iso - t)
    """
    return iso_center_xyz + rotation_xyz.T @ (point_xyz - iso_center_xyz - translation_xyz)


# -----------------------------------------------------------------------------
# Geometry construction
# -----------------------------------------------------------------------------


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n



def make_detector_basis_from_forward(
    forward_xyz: np.ndarray,
    world_up_xyz: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
) -> Tuple[np.ndarray, np.ndarray]:
    f = normalize(forward_xyz)
    u = np.cross(f, world_up_xyz)
    if np.linalg.norm(u) < 1e-6:
        world_up_xyz = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u = np.cross(f, world_up_xyz)
    u = normalize(u)
    v = normalize(np.cross(u, f))
    return u, v



def make_itk_like_pose(
    iso_center_mm: np.ndarray,
    rp_deg: float,
    sid_mm: float,
    isocenter_plane_spacing_mm: Tuple[float, float],
    detector_size_px: Tuple[int, int],
    detector_center_index: Optional[Tuple[float, float]] = None,
    volume_translation_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    volume_rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> DRRGeometry:
    """
    Build a DRRGeometry inspired by the ITK CLI.

    Canonical geometry before volume pose:
      - source is on +X at distance `sid_mm` from isocenter
      - detector plane is opposite the source across isocenter at equal distance
        (i.e. SDD = 2 * SID)
      - `rp_deg` rotates the acquisition around world Z

    The resulting detector plane spacing is computed so that the pixel spacing at
    the isocenter plane equals `isocenter_plane_spacing_mm`.
    """
    H, W = detector_size_px
    res_row_iso, res_col_iso = isocenter_plane_spacing_mm

    # Match the ITK help text: user specifies spacing at isocenter plane.
    # With detector opposite the source across isocenter, SDD = 2 * SID, so M = 2.
    sdd_mm = 2.0 * sid_mm
    magnification = sdd_mm / sid_mm
    det_row_spacing_mm = res_row_iso * magnification
    det_col_spacing_mm = res_col_iso * magnification

    theta = math.radians(rp_deg)

    # Orbit in XY plane, around Z.
    source_offset = np.array([
        sid_mm * math.cos(theta),
        sid_mm * math.sin(theta),
        0.0,
    ], dtype=np.float64)

    source_mm = iso_center_mm + source_offset
    detector_center_mm = iso_center_mm - source_offset

    forward = detector_center_mm - source_mm
    detector_u_mm, detector_v_mm = make_detector_basis_from_forward(forward.astype(np.float32))
    detector_u_mm = detector_u_mm.astype(np.float64)
    detector_v_mm = detector_v_mm.astype(np.float64)

    # Optional principal point / central-axis offset in detector coordinates.
    # detector_center_index is interpreted as (col, row), in continuous pixel indices.
    if detector_center_index is None:
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
    else:
        cx, cy = detector_center_index

    center_col = (W - 1) / 2.0
    center_row = (H - 1) / 2.0
    delta_cols = cx - center_col
    delta_rows = cy - center_row

    detector_center_mm = (
        detector_center_mm
        + delta_cols * det_col_spacing_mm * detector_u_mm
        + delta_rows * det_row_spacing_mm * detector_v_mm
    )

    # Apply inverse volume transform to the geometry, equivalent to transforming
    # the volume by +t,+R relative to a fixed scanner.
    tx, ty, tz = volume_translation_mm
    rx, ry, rz = volume_rotation_deg
    t_xyz = np.array([tx, ty, tz], dtype=np.float64)
    R = compose_rotation_xyz(rx, ry, rz)

    source_mm = apply_inverse_volume_pose_to_point(source_mm, iso_center_mm, t_xyz, R)
    detector_center_mm = apply_inverse_volume_pose_to_point(detector_center_mm, iso_center_mm, t_xyz, R)

    # Basis vectors are directions, so rotate them by inverse rotation only.
    detector_u_mm = R.T @ detector_u_mm
    detector_v_mm = R.T @ detector_v_mm

    return DRRGeometry(
        source_mm=source_mm.astype(np.float32),
        detector_center_mm=detector_center_mm.astype(np.float32),
        detector_u_mm=detector_u_mm.astype(np.float32),
        detector_v_mm=detector_v_mm.astype(np.float32),
        detector_size_px=(H, W),
        detector_spacing_mm=(float(det_row_spacing_mm), float(det_col_spacing_mm)),
    )


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def save_image(path: str, img: np.ndarray, invert: bool, p_lo: float, p_hi: float) -> None:
    import SimpleITK as sitk
    ext = Path(path).suffix.lower()

    if ext in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}:
        import imageio.v2 as imageio
        x = normalize_image(img, invert=invert, p_lo=p_lo, p_hi=p_hi)
        imageio.imwrite(path, (x * 255).astype(np.uint8))
        return

    # Save raw float image for medical-image extensions.
    out = sitk.GetImageFromArray(img.astype(np.float32))
    sitk.WriteImage(out, path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='get_drr_siddon_jacobs.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            'Calculate a Digitally Reconstructed Radiograph from a CT/CBCT image '
            'using a Siddon/Jacobs-style ray-tracing projector.'
        ),
    )

    p.add_argument('input', help='Input 3D image filename readable by SimpleITK')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    p.add_argument('-res', nargs=2, type=float, metavar=('ROW_MM', 'COL_MM'), default=(0.51, 0.51),
                   help='DRR pixel spacing in the isocenter plane in mm')
    p.add_argument('-size', nargs=2, type=int, metavar=('H', 'W'), default=(512, 512),
                   help='DRR size in pixels')
    p.add_argument('-scd', type=float, default=1000.0,
                   help='Source to isocenter distance in mm')
    p.add_argument('-t', nargs=3, type=float, metavar=('TX', 'TY', 'TZ'), default=(0.0, 0.0, 0.0),
                   help='Volume translation in x, y, z in mm')
    p.add_argument('-rx', type=float, default=0.0, help='Volume rotation about x axis in degrees')
    p.add_argument('-ry', type=float, default=0.0, help='Volume rotation about y axis in degrees')
    p.add_argument('-rz', type=float, default=0.0, help='Volume rotation about z axis in degrees')
    p.add_argument('-2dcx', nargs=2, type=float, metavar=('COL', 'ROW'), default=None,
                   help='Central axis detector position in continuous pixel indices (col row)')
    p.add_argument('-iso', nargs=3, type=float, metavar=('IX', 'IY', 'IZ'), default=None,
                   help='CT isocenter in continuous voxel indices (x y z)')
    p.add_argument('-rp', type=float, default=0.0, help='Projection angle in degrees')
    p.add_argument('-threshold', type=float, default=0.0,
                   help='Ignore CT values below this threshold')
    p.add_argument('-o', '--output', required=True, help='Output image filename')

    # Practical additions.
    p.add_argument('--invert', action='store_true',
                   help='Invert grayscale when saving display-oriented formats like PNG')
    p.add_argument('--no-clamp-negative', action='store_true',
                   help='Do not clamp negative intensities to zero above the threshold')
    p.add_argument('--p-lo', type=float, default=1.0,
                   help='Lower percentile for PNG-style normalization')
    p.add_argument('--p-hi', type=float, default=99.5,
                   help='Upper percentile for PNG-style normalization')
    p.add_argument(
        '--n-cores',
        type=int,
        default=None,
        help='Number of CPU cores/processes for parallel rendering. Omit for serial rendering.',
    )
    p.add_argument(
        '--mp-chunksize',
        type=int,
        default=1,
        help='Row chunksize for multiprocessing work scheduling in drr.renderer.',
    )

    return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()

    vol = load_volume_sitk(args.input)

    if args.verbose:
        print_volume_debug(vol)

    if args.iso is None:
        iso_center_mm = volume_center_world_xyz(vol).astype(np.float64)
        iso_desc = 'volume center'
    else:
        ix, iy, iz = args.iso
        # convert x,y,z continuous voxel index -> z,y,x for helper
        iso_center_mm = voxel_zyx_to_world_xyz(
            np.array([[iz, iy, ix]], dtype=np.float32),
            vol.spacing_zyx,
            vol.origin_zyx,
        )[0].astype(np.float64)
        iso_desc = f'user-specified voxel index (x,y,z)=({ix:.3f}, {iy:.3f}, {iz:.3f})'

    geom = make_itk_like_pose(
        iso_center_mm=iso_center_mm,
        rp_deg=args.rp,
        sid_mm=args.scd,
        isocenter_plane_spacing_mm=(float(args.res[0]), float(args.res[1])),
        detector_size_px=(int(args.size[0]), int(args.size[1])),
        detector_center_index=None if args.__dict__['2dcx'] is None else tuple(args.__dict__['2dcx']),
        volume_translation_mm=tuple(args.t),
        volume_rotation_deg=(args.rx, args.ry, args.rz),
    )

    projector_kwargs = {
        'hu_air_threshold': float(args.threshold),
        'clamp_negative_to_zero': not args.no_clamp_negative,
    }

    if args.n_cores is not None and args.n_cores < 1:
        raise ValueError('--n-cores must be a positive integer when provided.')

    if args.verbose:
        H, W = geom.detector_size_px
        dr, dc = geom.detector_spacing_mm
        sdd = float(np.linalg.norm(geom.detector_center_mm - geom.source_mm))
        print(f'Isocenter source       : {iso_desc}')
        print(f'Isocenter world xyz mm : {iso_center_mm}')
        print(f'Source mm              : {geom.source_mm}')
        print(f'Detector center mm     : {geom.detector_center_mm}')
        print(f'Detector spacing mm    : (row={dr:.4f}, col={dc:.4f})')
        print(f'Detector size px       : (H={H}, W={W})')
        print(f'SDD mm                 : {sdd:.4f}')
        print(f'Projector kwargs       : {projector_kwargs}')
        print(f'n_cores                : {args.n_cores}')
        print(f'mp_chunksize           : {args.mp_chunksize}')
        print()

    drr = generate_drr(
        vol=vol,
        geom=geom,
        projector_fn=ray_integral_siddon_jacobs,
        projector_kwargs=projector_kwargs,
        show_progress=True,
        n_cores=args.n_cores,
        mp_chunksize=max(1, int(args.mp_chunksize)),
    )

    if args.verbose:
        print_projection_stats(drr)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    save_image(args.output, drr, invert=args.invert, p_lo=args.p_lo, p_hi=args.p_hi)

    if args.verbose:
        print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
