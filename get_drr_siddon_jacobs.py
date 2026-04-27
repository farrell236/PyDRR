#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from drr.geometry import DRRGeometry, make_detector_basis_from_forward
from drr.renderer import generate_drr
from drr.visualize import print_geometry_debug, print_projection_stats, print_volume_debug, save_png
from drr.volume import load_volume_sitk, voxel_zyx_to_world_xyz


def parse_args():
    p = argparse.ArgumentParser(
        description="Calculate a Digitally Reconstructed Radiograph from a CT/CBCT image using a Siddon/Jacobs-style ray-tracing projector."
    )
    p.add_argument("input", type=str, help="Input 3D image filename readable by SimpleITK")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output (default: False)")
    p.add_argument("-res", nargs=2, type=float, metavar=("ROW_MM", "COL_MM"), default=(0.51, 0.51),
                   help="DRR pixel spacing in the isocenter plane in mm (default: %(default)s)")
    p.add_argument("-size", nargs=2, type=int, metavar=("H", "W"), default=(512, 512),
                   help="DRR size in pixels (default: %(default)s)")
    p.add_argument("-scd", type=float, default=1000.0,
                   help="Source to isocenter distance in mm (default: %(default)s)")
    p.add_argument("-t", nargs=3, type=float, metavar=("TX", "TY", "TZ"), default=(0.0, 0.0, 0.0),
                   help="Volume translation in x, y, z in mm (default: %(default)s)")
    p.add_argument("-rx", type=float, default=0.0, help="Volume rotation about x axis in degrees (default: %(default)s)")
    p.add_argument("-ry", type=float, default=0.0, help="Volume rotation about y axis in degrees (default: %(default)s)")
    p.add_argument("-rz", type=float, default=0.0, help="Volume rotation about z axis in degrees (default: %(default)s)")
    p.add_argument("-2dcx", nargs=2, type=float, metavar=("COL", "ROW"), default=None,
                   help="Central axis detector position in continuous pixel indices (col row) (default: None)")
    p.add_argument("-iso", nargs=3, type=float, metavar=("IX", "IY", "IZ"), default=None,
                   help="CT isocenter in continuous voxel indices (x y z) (default: None)")
    p.add_argument("-rp", type=float, default=0.0, help="Projection angle in degrees (default: %(default)s)")
    p.add_argument("-threshold", type=float, default=0.0,
                   help="Ignore CT values below this threshold (default: %(default)s)")
    p.add_argument("-o", "--output", required=True, type=str,
                   help="Output image filename (default: None)")
    p.add_argument("--invert", action="store_true",
                   help="Invert grayscale when saving display-oriented formats like PNG (default: False)")
    p.add_argument("--no-clamp-negative", action="store_true",
                   help="Do not clamp negative intensities to zero above the threshold (default: False)")
    p.add_argument("--p-lo", type=float, default=1.0,
                   help="Lower percentile for PNG-style normalization (default: %(default)s)")
    p.add_argument("--p-hi", type=float, default=99.5,
                   help="Upper percentile for PNG-style normalization (default: %(default)s)")
    p.add_argument("--n-cores", type=int, default=None,
                   help="Number of CPU cores/processes for parallel rendering. Omit for serial rendering. (default: None)")
    p.add_argument("--mp-chunksize", type=int, default=1,
                   help="Row chunksize for multiprocessing work scheduling in drr.renderer. (default: %(default)s)")
    p.add_argument("--backend", choices=["cpu", "cuda"], default="cpu",
                   help="Rendering backend. 'cpu' uses the Python projector, 'cuda' uses CuPy RawKernel. (default: %(default)s)")
    return p.parse_args()


def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float):
    import numpy as np

    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def build_geometry(vol, args):
    import numpy as np

    if args.iso is None:
        shape_zyx = vol.data.shape
        center_zyx = ((np.array(shape_zyx, dtype=np.float32) - 1.0) / 2.0)
        iso_center_mm = voxel_zyx_to_world_xyz(center_zyx[None], vol.spacing_zyx, vol.origin_zyx)[0]
    else:
        ix, iy, iz = map(float, args.iso)
        center_zyx = np.array([iz, iy, ix], dtype=np.float32)
        iso_center_mm = voxel_zyx_to_world_xyz(center_zyx[None], vol.spacing_zyx, vol.origin_zyx)[0]

    iso_center_mm = iso_center_mm + np.array(args.t, dtype=np.float32)

    theta = np.deg2rad(float(args.rp))
    source_offset = np.array([
        args.scd * np.cos(theta),
        args.scd * np.sin(theta),
        0.0,
    ], dtype=np.float32)

    # detector center at same source-to-isocenter distance on opposite side
    detector_offset = -source_offset.copy()

    R = rotation_matrix_xyz(args.rx, args.ry, args.rz)
    source_offset = (R @ source_offset.reshape(3, 1)).ravel()
    detector_offset = (R @ detector_offset.reshape(3, 1)).ravel()

    source_mm = iso_center_mm + source_offset
    detector_center_mm = iso_center_mm + detector_offset

    forward = detector_center_mm - source_mm
    detector_u_mm, detector_v_mm = make_detector_basis_from_forward(forward)

    H, W = map(int, args.size)
    row_spacing, col_spacing = map(float, args.res)

    if args.__dict__["2dcx"] if "2dcx" in args.__dict__ else False:
        pass

    if getattr(args, "2dcx", None) is not None:
        # argparse stores invalid identifier via getattr only if manually set; keep fallback below
        pass

    center_shift_mm = np.zeros((3,), dtype=np.float32)
    cx_arg = getattr(args, "2dcx", None)
    if cx_arg is None:
        cx_arg = getattr(args, "_2dcx", None)
    if cx_arg is not None:
        col_idx, row_idx = map(float, cx_arg)
        default_col = (W - 1) / 2.0
        default_row = (H - 1) / 2.0
        dc = col_idx - default_col
        dr = row_idx - default_row
        center_shift_mm = (
            dc * col_spacing * detector_u_mm
            + dr * row_spacing * detector_v_mm
        ).astype(np.float32)

    detector_center_mm = detector_center_mm + center_shift_mm

    return DRRGeometry(
        source_mm=source_mm.astype(np.float32),
        detector_center_mm=detector_center_mm.astype(np.float32),
        detector_u_mm=detector_u_mm.astype(np.float32),
        detector_v_mm=detector_v_mm.astype(np.float32),
        detector_size_px=(H, W),
        detector_spacing_mm=(row_spacing, col_spacing),
    ), iso_center_mm.astype(np.float32)


def main():
    args = parse_args()
    vol = load_volume_sitk(args.input)

    # argparse cannot create attribute named "2dcx"; recover from namespace dict if present
    if "2dcx" not in args.__dict__:
        for k in list(args.__dict__.keys()):
            if k.endswith("2dcx"):
                args.__dict__["2dcx"] = args.__dict__[k]

    geom, iso_center_mm = build_geometry(vol, args)

    if args.verbose:
        print_volume_debug(vol)
        print_geometry_debug(geom, iso_center_mm=iso_center_mm)

    drr = generate_drr(
        vol=vol,
        geom=geom,
        projector_kwargs={
            "hu_air_threshold": args.threshold,
            "clamp_negative_to_zero": not args.no_clamp_negative,
        },
        n_cores=args.n_cores,
        mp_chunksize=args.mp_chunksize,
        backend=args.backend,
        show_progress=True,
    )

    if args.verbose:
        print_projection_stats(drr, "DRR")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_png(str(output_path), drr, invert=args.invert)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
