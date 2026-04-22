import math
from typing import Optional, Tuple

import numpy as np

from .volume import Volume


def ray_box_intersection(
    ray_origin_xyz: np.ndarray,
    ray_dir_xyz: np.ndarray,
    box_min_xyz: np.ndarray,
    box_max_xyz: np.ndarray,
    eps: float = 1e-8,
) -> Optional[Tuple[float, float]]:
    """Intersect a ray with an axis-aligned box using the slab method."""
    tmin = -np.inf
    tmax = np.inf

    for i in range(3):
        if abs(ray_dir_xyz[i]) < eps:
            if ray_origin_xyz[i] < box_min_xyz[i] or ray_origin_xyz[i] > box_max_xyz[i]:
                return None
        else:
            t1 = (box_min_xyz[i] - ray_origin_xyz[i]) / ray_dir_xyz[i]
            t2 = (box_max_xyz[i] - ray_origin_xyz[i]) / ray_dir_xyz[i]
            t_near_i = min(t1, t2)
            t_far_i = max(t1, t2)
            tmin = max(tmin, t_near_i)
            tmax = min(tmax, t_far_i)
            if tmin > tmax:
                return None

    return tmin, tmax


def ray_integral_siddon_jacobs(
    vol: Volume,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    hu_air_threshold: Optional[float] = -900.0,
    clamp_negative_to_zero: bool = True,
    eps: float = 1e-8,
) -> float:
    """Siddon/Jacobs-style line integral through an axis-aligned voxel grid.

    Notes:
        - Assumes the volume is axis aligned.
        - Uses voxel values directly as attenuation surrogates.
        - Air can be suppressed with `hu_air_threshold`.
    """
    data = vol.data
    nz, ny, nx = data.shape

    spacing_zyx = np.asarray(vol.spacing_zyx, dtype=np.float64)
    origin_zyx = np.asarray(vol.origin_zyx, dtype=np.float64)

    sz, sy, sx = float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2])
    oz, oy, ox = float(origin_zyx[0]), float(origin_zyx[1]), float(origin_zyx[2])

    start_xyz = np.asarray(start_xyz, dtype=np.float64)
    end_xyz = np.asarray(end_xyz, dtype=np.float64)

    ray = end_xyz - start_xyz
    ray_length = np.linalg.norm(ray)
    if ray_length < eps:
        return 0.0

    d = ray / ray_length

    x_min = ox
    x_max = ox + nx * sx
    y_min = oy
    y_max = oy + ny * sy
    z_min = oz
    z_max = oz + nz * sz

    hit = ray_box_intersection(
        ray_origin_xyz=start_xyz,
        ray_dir_xyz=d,
        box_min_xyz=np.array([x_min, y_min, z_min], dtype=np.float64),
        box_max_xyz=np.array([x_max, y_max, z_max], dtype=np.float64),
        eps=eps,
    )
    if hit is None:
        return 0.0

    t_enter, t_exit = hit
    t_enter = max(t_enter, 0.0)
    t_exit = min(t_exit, ray_length)
    if t_exit <= t_enter:
        return 0.0

    p = start_xyz + (t_enter + 1e-10) * d
    px, py, pz = p

    ix = int(math.floor((px - ox) / sx))
    iy = int(math.floor((py - oy) / sy))
    iz = int(math.floor((pz - oz) / sz))

    ix = min(max(ix, 0), nx - 1)
    iy = min(max(iy, 0), ny - 1)
    iz = min(max(iz, 0), nz - 1)

    if d[0] > eps:
        step_x = 1
        next_x_boundary = ox + (ix + 1) * sx
        t_max_x = t_enter + (next_x_boundary - px) / d[0]
        t_delta_x = sx / d[0]
    elif d[0] < -eps:
        step_x = -1
        next_x_boundary = ox + ix * sx
        t_max_x = t_enter + (next_x_boundary - px) / d[0]
        t_delta_x = -sx / d[0]
    else:
        step_x = 0
        t_max_x = np.inf
        t_delta_x = np.inf

    if d[1] > eps:
        step_y = 1
        next_y_boundary = oy + (iy + 1) * sy
        t_max_y = t_enter + (next_y_boundary - py) / d[1]
        t_delta_y = sy / d[1]
    elif d[1] < -eps:
        step_y = -1
        next_y_boundary = oy + iy * sy
        t_max_y = t_enter + (next_y_boundary - py) / d[1]
        t_delta_y = -sy / d[1]
    else:
        step_y = 0
        t_max_y = np.inf
        t_delta_y = np.inf

    if d[2] > eps:
        step_z = 1
        next_z_boundary = oz + (iz + 1) * sz
        t_max_z = t_enter + (next_z_boundary - pz) / d[2]
        t_delta_z = sz / d[2]
    elif d[2] < -eps:
        step_z = -1
        next_z_boundary = oz + iz * sz
        t_max_z = t_enter + (next_z_boundary - pz) / d[2]
        t_delta_z = -sz / d[2]
    else:
        step_z = 0
        t_max_z = np.inf
        t_delta_z = np.inf

    integral = 0.0
    t = t_enter

    while t < t_exit:
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
            break

        t_next = min(t_max_x, t_max_y, t_max_z, t_exit)
        seg_len = t_next - t

        if seg_len > 0:
            val = float(data[iz, iy, ix])
            if hu_air_threshold is not None and val < hu_air_threshold:
                val = 0.0
            elif clamp_negative_to_zero:
                val = max(val, 0.0)
            integral += val * seg_len

        t = t_next

        crossed_x = t_max_x <= t + eps
        crossed_y = t_max_y <= t + eps
        crossed_z = t_max_z <= t + eps

        if crossed_x:
            ix += step_x
            t_max_x += t_delta_x
        if crossed_y:
            iy += step_y
            t_max_y += t_delta_y
        if crossed_z:
            iz += step_z
            t_max_z += t_delta_z

    return float(integral)
