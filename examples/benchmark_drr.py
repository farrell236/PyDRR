#!/usr/bin/env python3
"""Benchmark single-DRR rendering across CPU serial, CPU multiprocessing, and CUDA.

Example:
    python -m examples.benchmark_drr /path/to/volume.nii.gz \
        --angle 30 \
        --det-h 512 --det-w 512 \
        --det-spacing-row 0.51 --det-spacing-col 0.51 \
        --sid-mm 1000 --idd-mm 500 \
        --cpu-cores 8 \
        --cuda-warmup
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from drr import (
    load_volume_sitk,
    volume_center_world_xyz,
    make_circular_orbit_pose,
    generate_drr,
    print_volume_debug,
    print_geometry_debug,
    print_projection_stats,
    save_png,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark DRR rendering backends.")
    p.add_argument("volume_path", type=str, help="Input CT/CBCT volume readable by SimpleITK")
    p.add_argument("--output-dir", type=str, default="benchmark_outputs", help="Directory for optional saved images")
    p.add_argument("--save-images", action="store_true", help="Save rendered PNGs for each backend")
    p.add_argument("--angle", type=float, default=30.0, help="Projection angle in degrees")
    p.add_argument("--sid-mm", type=float, default=1000.0, help="Source-to-isocenter distance in mm")
    p.add_argument("--idd-mm", type=float, default=500.0, help="Isocenter-to-detector distance in mm")
    p.add_argument("--det-h", type=int, default=512, help="Detector height in pixels")
    p.add_argument("--det-w", type=int, default=512, help="Detector width in pixels")
    p.add_argument("--det-spacing-row", type=float, default=0.51, help="Detector row spacing in mm")
    p.add_argument("--det-spacing-col", type=float, default=0.51, help="Detector column spacing in mm")
    p.add_argument("--threshold", type=float, default=-900.0, help="HU threshold below which voxels are ignored")
    p.add_argument("--cpu-cores", type=int, default=8, help="CPU worker processes for multiprocessing benchmark")
    p.add_argument("--repeats", type=int, default=1, help="Number of timed repeats per backend")
    p.add_argument("--skip-cpu-serial", action="store_true", help="Skip serial CPU benchmark")
    p.add_argument("--skip-cpu-mp", action="store_true", help="Skip multiprocessing CPU benchmark")
    p.add_argument("--skip-cuda", action="store_true", help="Skip CUDA benchmark")
    p.add_argument("--cuda-warmup", action="store_true", help="Run one untimed CUDA warmup render before timing")
    p.add_argument("--invert", action="store_true", help="Invert PNGs if --save-images is used")
    return p.parse_args()


def time_backend(
    name: str,
    vol,
    geom,
    projector_kwargs: Dict,
    backend: str,
    n_cores: Optional[int],
    repeats: int,
) -> Dict:
    times: List[float] = []
    last_img = None

    for i in range(repeats):
        t0 = time.perf_counter()
        last_img = generate_drr(
            vol=vol,
            geom=geom,
            projector_kwargs=projector_kwargs,
            show_progress=False,
            n_cores=n_cores,
            mp_chunksize=1,
            backend=backend,
        )
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"[{name}] run {i+1}/{repeats}: {dt:.3f} s")

    assert last_img is not None
    return {
        "name": name,
        "times": times,
        "mean_s": float(statistics.mean(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "image": last_img,
    }


def maybe_save(output_dir: Path, stem: str, img: np.ndarray, invert: bool) -> None:
    out_path = output_dir / f"{stem}.png"
    save_png(str(out_path), img, invert=invert)
    print(f"Saved {out_path}")


def print_summary(results: List[Dict]) -> None:
    if not results:
        return

    print("\nBenchmark summary")
    print("-" * 72)
    base = results[0]["mean_s"]
    for r in results:
        speedup = base / r["mean_s"] if r["mean_s"] > 0 else float("nan")
        print(
            f"{r['name']:<14} "
            f"mean={r['mean_s']:.3f}s  "
            f"min={r['min_s']:.3f}s  "
            f"max={r['max_s']:.3f}s  "
            f"speedup_vs_first={speedup:.2f}x"
        )

    cpu_serial = next((r for r in results if r["name"] == "cpu_serial"), None)
    if cpu_serial is not None:
        print("\nSpeedup vs cpu_serial")
        for r in results:
            speedup = cpu_serial["mean_s"] / r["mean_s"] if r["mean_s"] > 0 else float("nan")
            print(f"  {r['name']:<14} {speedup:.2f}x")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vol = load_volume_sitk(args.volume_path)
    print_volume_debug(vol)

    iso = volume_center_world_xyz(vol)
    print("Isocenter (world xyz mm):", iso)
    print()

    geom = make_circular_orbit_pose(
        iso_center_mm=iso,
        angle_deg=args.angle,
        sid_mm=args.sid_mm,
        idd_mm=args.idd_mm,
        detector_size_px=(args.det_h, args.det_w),
        detector_spacing_mm=(args.det_spacing_row, args.det_spacing_col),
    )
    print_geometry_debug(geom, iso_center_mm=iso)

    projector_kwargs = {
        "hu_air_threshold": args.threshold,
        "clamp_negative_to_zero": True,
    }

    results: List[Dict] = []

    if not args.skip_cpu_serial:
        results.append(
            time_backend(
                name="cpu_serial",
                vol=vol,
                geom=geom,
                projector_kwargs=projector_kwargs,
                backend="cpu",
                n_cores=None,
                repeats=args.repeats,
            )
        )
        print_projection_stats(results[-1]["image"], "cpu_serial")

    if not args.skip_cpu_mp:
        results.append(
            time_backend(
                name="cpu_mp",
                vol=vol,
                geom=geom,
                projector_kwargs=projector_kwargs,
                backend="cpu",
                n_cores=args.cpu_cores,
                repeats=args.repeats,
            )
        )
        print_projection_stats(results[-1]["image"], "cpu_mp")

    if not args.skip_cuda:
        if args.cuda_warmup:
            print("[cuda] warmup render...")
            _ = generate_drr(
                vol=vol,
                geom=geom,
                projector_kwargs=projector_kwargs,
                show_progress=False,
                backend="cuda",
            )
        results.append(
            time_backend(
                name="cuda",
                vol=vol,
                geom=geom,
                projector_kwargs=projector_kwargs,
                backend="cuda",
                n_cores=None,
                repeats=args.repeats,
            )
        )
        print_projection_stats(results[-1]["image"], "cuda")

    print_summary(results)

    if args.save_images:
        for r in results:
            maybe_save(output_dir, f"drr_{r['name']}", r["image"], invert=args.invert)


if __name__ == "__main__":
    main()