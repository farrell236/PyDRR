from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .geometry import DRRGeometry, detector_pixel_centers_world, make_circular_orbit_pose
from .projector import ray_integral_siddon_jacobs
from .volume import Volume, volume_center_world_xyz


ProjectorFn = Callable[..., float]

# Globals used by multiprocessing workers.
_MP_VOL: Optional[Volume] = None
_MP_SOURCE_MM: Optional[np.ndarray] = None
_MP_DET_PTS: Optional[np.ndarray] = None
_MP_PROJECTOR_FN: Optional[ProjectorFn] = None
_MP_PROJECTOR_KWARGS: Optional[Dict] = None


def _init_row_worker(
    vol: Volume,
    source_mm: np.ndarray,
    det_pts: np.ndarray,
    projector_fn: ProjectorFn,
    projector_kwargs: Dict,
) -> None:
    global _MP_VOL, _MP_SOURCE_MM, _MP_DET_PTS, _MP_PROJECTOR_FN, _MP_PROJECTOR_KWARGS
    _MP_VOL = vol
    _MP_SOURCE_MM = source_mm
    _MP_DET_PTS = det_pts
    _MP_PROJECTOR_FN = projector_fn
    _MP_PROJECTOR_KWARGS = projector_kwargs



def _render_row(row_idx: int) -> Tuple[int, np.ndarray]:
    if _MP_VOL is None or _MP_SOURCE_MM is None or _MP_DET_PTS is None or _MP_PROJECTOR_FN is None or _MP_PROJECTOR_KWARGS is None:
        raise RuntimeError("Multiprocessing renderer worker is not initialized")

    row_pts = _MP_DET_PTS[row_idx]
    row = np.zeros((row_pts.shape[0],), dtype=np.float32)
    for c, end_xyz in enumerate(row_pts):
        row[c] = _MP_PROJECTOR_FN(
            vol=_MP_VOL,
            start_xyz=_MP_SOURCE_MM,
            end_xyz=end_xyz,
            **_MP_PROJECTOR_KWARGS,
        )
    return row_idx, row



def generate_drr(
    vol: Volume,
    geom: DRRGeometry,
    projector_fn: ProjectorFn = ray_integral_siddon_jacobs,
    projector_kwargs: Optional[Dict] = None,
    show_progress: bool = True,
    n_cores: Optional[int] = None,
    mp_chunksize: int = 1,
) -> np.ndarray:
    """Render one DRR image for a given projection geometry.

    Args:
        vol: Input volume.
        geom: Source/detector geometry for one projection.
        projector_fn: Per-ray projector. Must be picklable for multiprocessing.
        projector_kwargs: Keyword arguments forwarded to ``projector_fn``.
        show_progress: Show tqdm progress bars.
        n_cores: If None or <= 1, render serially. If > 1, render detector rows
            in parallel using ``multiprocessing.Pool``.
        mp_chunksize: Row scheduling chunk size for multiprocessing.
    """
    if projector_kwargs is None:
        projector_kwargs = {}

    if mp_chunksize < 1:
        raise ValueError("mp_chunksize must be >= 1")

    det_pts = detector_pixel_centers_world(geom)
    H, W, _ = det_pts.shape
    drr = np.zeros((H, W), dtype=np.float32)

    use_mp = n_cores is not None and int(n_cores) > 1

    if not use_mp:
        row_iter = range(H)
        if show_progress:
            row_iter = tqdm(row_iter, desc="Rendering DRR", leave=False)

        for r in row_iter:
            for c in range(W):
                drr[r, c] = projector_fn(
                    vol=vol,
                    start_xyz=geom.source_mm,
                    end_xyz=det_pts[r, c],
                    **projector_kwargs,
                )
        return drr

    n_cores = int(n_cores)
    row_iter_mp = range(H)
    with mp.Pool(
        processes=n_cores,
        initializer=_init_row_worker,
        initargs=(vol, geom.source_mm, det_pts, projector_fn, projector_kwargs),
    ) as pool:
        results_iter = pool.imap(_render_row, row_iter_mp, chunksize=mp_chunksize)
        if show_progress:
            results_iter = tqdm(results_iter, total=H, desc=f"Rendering DRR ({n_cores} cores)", leave=False)

        for r, row in results_iter:
            drr[r] = row

    return drr



def generate_orbit_drrs(
    vol: Volume,
    angles_deg: List[float],
    sid_mm: float = 1000.0,
    idd_mm: float = 500.0,
    detector_size_px: Tuple[int, int] = (512, 512),
    detector_spacing_mm: Tuple[float, float] = (1.0, 1.0),
    projector_fn: ProjectorFn = ray_integral_siddon_jacobs,
    projector_kwargs: Optional[Dict] = None,
    n_cores: Optional[int] = None,
    mp_chunksize: int = 1,
    show_progress: bool = True,
) -> List[np.ndarray]:
    """Render DRRs for a sequence of circular-orbit angles.

    Multiprocessing is applied within each DRR render across detector rows.
    """
    if projector_kwargs is None:
        projector_kwargs = {}

    iso_center = volume_center_world_xyz(vol)
    drrs = []

    angle_iter = angles_deg
    if show_progress:
        angle_iter = tqdm(angles_deg, desc="Orbit DRRs")

    for ang in angle_iter:
        geom = make_circular_orbit_pose(
            iso_center_mm=iso_center,
            angle_deg=ang,
            sid_mm=sid_mm,
            idd_mm=idd_mm,
            detector_size_px=detector_size_px,
            detector_spacing_mm=detector_spacing_mm,
        )
        drr = generate_drr(
            vol=vol,
            geom=geom,
            projector_fn=projector_fn,
            projector_kwargs=projector_kwargs,
            show_progress=False,
            n_cores=n_cores,
            mp_chunksize=mp_chunksize,
        )
        drrs.append(drr)

    return drrs
