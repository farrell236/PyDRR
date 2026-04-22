"""Phase-1 DRR demo.

Usage:
    python examples/demo_phase1.py /path/to/your_cbct.nii.gz
"""

import argparse
from pathlib import Path

import numpy as np

from drr import (
    load_volume_sitk,
    volume_center_world_xyz,
    make_circular_orbit_pose,
    generate_drr,
    generate_orbit_drrs,
    save_png,
    print_volume_debug,
    print_geometry_debug,
    print_projection_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render phase-1 DRRs from a CBCT volume.")
    parser.add_argument("volume_path", type=str, help="Path to the CBCT volume (e.g. .nii.gz)")
    parser.add_argument("--output_dir", type=str, default="outputs_phase1", help="Directory for saved PNGs")
    parser.add_argument("--single_angle", type=float, default=90.0, help="Angle in degrees for the single-view DRR")
    parser.add_argument("--sid_mm", type=float, default=1000.0, help="Source-to-isocenter distance in mm")
    parser.add_argument("--idd_mm", type=float, default=500.0, help="Isocenter-to-detector distance in mm")
    parser.add_argument("--det_h", type=int, default=512, help="Detector height in pixels")
    parser.add_argument("--det_w", type=int, default=512, help="Detector width in pixels")
    parser.add_argument("--det_spacing_row", type=float, default=1.0, help="Detector row spacing in mm")
    parser.add_argument("--det_spacing_col", type=float, default=1.0, help="Detector col spacing in mm")
    parser.add_argument("--hu_air_threshold", type=float, default=-900.0, help="Threshold below which voxels are treated as air")
    parser.add_argument("--orbit_start", type=float, default=-30.0, help="Start angle for orbit demo")
    parser.add_argument("--orbit_end", type=float, default=30.0, help="End angle for orbit demo")
    parser.add_argument("--orbit_num", type=int, default=7, help="Number of orbit views to render")
    parser.add_argument("--invert", action="store_true", help="Save with X-ray-like black anatomy on white background")
    parser.add_argument("--n-cores", type=int, default=None, help="Number of CPU worker processes for row-parallel rendering")
    parser.add_argument("--mp-chunksize", type=int, default=1, help="Multiprocessing chunksize for row scheduling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vol = load_volume_sitk(args.volume_path)
    print_volume_debug(vol)

    iso = volume_center_world_xyz(vol)
    print("Isocenter (world xyz mm):", iso)
    print()

    detector_size_px = (args.det_h, args.det_w)
    detector_spacing_mm = (args.det_spacing_row, args.det_spacing_col)
    projector_kwargs = {
        "hu_air_threshold": args.hu_air_threshold,
        "clamp_negative_to_zero": True,
    }

    geom = make_circular_orbit_pose(
        iso_center_mm=iso,
        angle_deg=args.single_angle,
        sid_mm=args.sid_mm,
        idd_mm=args.idd_mm,
        detector_size_px=detector_size_px,
        detector_spacing_mm=detector_spacing_mm,
    )
    print_geometry_debug(geom, iso_center_mm=iso)

    drr = generate_drr(
        vol=vol,
        geom=geom,
        projector_kwargs=projector_kwargs,
        show_progress=True,
        n_cores=args.n_cores,
        mp_chunksize=args.mp_chunksize,
    )
    print_projection_stats(drr, "Single DRR")

    single_path = output_dir / "drr_single.png"
    save_png(str(single_path), drr, invert=args.invert)
    print(f"Saved {single_path}")
    print()

    angles = list(np.linspace(args.orbit_start, args.orbit_end, args.orbit_num))
    drrs = generate_orbit_drrs(
        vol=vol,
        angles_deg=angles,
        sid_mm=args.sid_mm,
        idd_mm=args.idd_mm,
        detector_size_px=detector_size_px,
        detector_spacing_mm=detector_spacing_mm,
        projector_kwargs=projector_kwargs,
        n_cores=args.n_cores,
        mp_chunksize=args.mp_chunksize,
    )

    for i, img in enumerate(drrs):
        out_path = output_dir / f"drr_orbit_{i:02d}.png"
        save_png(str(out_path), img, invert=args.invert)
        print(f"Saved {out_path}")

    print()
    print(f"Saved {len(drrs)} orbit DRRs")


if __name__ == "__main__":
    main()
