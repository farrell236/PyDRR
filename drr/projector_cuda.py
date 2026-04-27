from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import DRRGeometry, detector_pixel_centers_world
from .volume import Volume


def _require_cupy():
    try:
        import cupy as cp
    except Exception as e:
        raise ImportError(
            "CuPy is required for backend='cuda'. Install a CUDA-compatible CuPy build, "
            "for example: pip install cupy-cuda12x"
        ) from e
    return cp


def upload_volume_to_gpu(vol: Volume):
    cp = _require_cupy()
    return {
        "data": cp.asarray(vol.data, dtype=cp.float32),
        "spacing_zyx": cp.asarray(vol.spacing_zyx, dtype=cp.float32),
        "origin_zyx": cp.asarray(vol.origin_zyx, dtype=cp.float32),
        "shape_zyx": np.asarray(vol.data.shape, dtype=np.int32),
    }


_CUDA_SRC = r'''
extern "C" __global__
void siddon_drr_kernel(
    const float* vol,
    const int nz,
    const int ny,
    const int nx,
    const float sz,
    const float sy,
    const float sx,
    const float oz,
    const float oy,
    const float ox,
    const float* det_pts,
    const float source_x,
    const float source_y,
    const float source_z,
    const int N,
    const float hu_air_threshold,
    const int clamp_negative_to_zero,
    float* out
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    const float eps = 1e-8f;

    float end_x = det_pts[3 * idx + 0];
    float end_y = det_pts[3 * idx + 1];
    float end_z = det_pts[3 * idx + 2];

    float ray_x = end_x - source_x;
    float ray_y = end_y - source_y;
    float ray_z = end_z - source_z;

    float ray_length = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    if (ray_length < eps) {
        out[idx] = 0.0f;
        return;
    }

    float dx = ray_x / ray_length;
    float dy = ray_y / ray_length;
    float dz = ray_z / ray_length;

    float x_min = ox;
    float x_max = ox + nx * sx;
    float y_min = oy;
    float y_max = oy + ny * sy;
    float z_min = oz;
    float z_max = oz + nz * sz;

    float tmin = -1e30f;
    float tmax =  1e30f;

    if (fabsf(dx) < eps) {
        if (source_x < x_min || source_x > x_max) {
            out[idx] = 0.0f;
            return;
        }
    } else {
        float t1 = (x_min - source_x) / dx;
        float t2 = (x_max - source_x) / dx;
        float tn = fminf(t1, t2);
        float tf = fmaxf(t1, t2);
        tmin = fmaxf(tmin, tn);
        tmax = fminf(tmax, tf);
        if (tmin > tmax) {
            out[idx] = 0.0f;
            return;
        }
    }

    if (fabsf(dy) < eps) {
        if (source_y < y_min || source_y > y_max) {
            out[idx] = 0.0f;
            return;
        }
    } else {
        float t1 = (y_min - source_y) / dy;
        float t2 = (y_max - source_y) / dy;
        float tn = fminf(t1, t2);
        float tf = fmaxf(t1, t2);
        tmin = fmaxf(tmin, tn);
        tmax = fminf(tmax, tf);
        if (tmin > tmax) {
            out[idx] = 0.0f;
            return;
        }
    }

    if (fabsf(dz) < eps) {
        if (source_z < z_min || source_z > z_max) {
            out[idx] = 0.0f;
            return;
        }
    } else {
        float t1 = (z_min - source_z) / dz;
        float t2 = (z_max - source_z) / dz;
        float tn = fminf(t1, t2);
        float tf = fmaxf(t1, t2);
        tmin = fmaxf(tmin, tn);
        tmax = fminf(tmax, tf);
        if (tmin > tmax) {
            out[idx] = 0.0f;
            return;
        }
    }

    float t_enter = fmaxf(tmin, 0.0f);
    float t_exit  = fminf(tmax, ray_length);
    if (t_exit <= t_enter) {
        out[idx] = 0.0f;
        return;
    }

    float px = source_x + (t_enter + 1e-10f) * dx;
    float py = source_y + (t_enter + 1e-10f) * dy;
    float pz = source_z + (t_enter + 1e-10f) * dz;

    int ix = (int)floorf((px - ox) / sx);
    int iy = (int)floorf((py - oy) / sy);
    int iz = (int)floorf((pz - oz) / sz);

    ix = max(0, min(ix, nx - 1));
    iy = max(0, min(iy, ny - 1));
    iz = max(0, min(iz, nz - 1));

    int step_x, step_y, step_z;
    float t_max_x, t_max_y, t_max_z;
    float t_delta_x, t_delta_y, t_delta_z;

    if (dx > eps) {
        step_x = 1;
        float next_x_boundary = ox + (ix + 1) * sx;
        t_max_x = t_enter + (next_x_boundary - px) / dx;
        t_delta_x = sx / dx;
    } else if (dx < -eps) {
        step_x = -1;
        float next_x_boundary = ox + ix * sx;
        t_max_x = t_enter + (next_x_boundary - px) / dx;
        t_delta_x = -sx / dx;
    } else {
        step_x = 0;
        t_max_x = 1e30f;
        t_delta_x = 1e30f;
    }

    if (dy > eps) {
        step_y = 1;
        float next_y_boundary = oy + (iy + 1) * sy;
        t_max_y = t_enter + (next_y_boundary - py) / dy;
        t_delta_y = sy / dy;
    } else if (dy < -eps) {
        step_y = -1;
        float next_y_boundary = oy + iy * sy;
        t_max_y = t_enter + (next_y_boundary - py) / dy;
        t_delta_y = -sy / dy;
    } else {
        step_y = 0;
        t_max_y = 1e30f;
        t_delta_y = 1e30f;
    }

    if (dz > eps) {
        step_z = 1;
        float next_z_boundary = oz + (iz + 1) * sz;
        t_max_z = t_enter + (next_z_boundary - pz) / dz;
        t_delta_z = sz / dz;
    } else if (dz < -eps) {
        step_z = -1;
        float next_z_boundary = oz + iz * sz;
        t_max_z = t_enter + (next_z_boundary - pz) / dz;
        t_delta_z = -sz / dz;
    } else {
        step_z = 0;
        t_max_z = 1e30f;
        t_delta_z = 1e30f;
    }

    float integral = 0.0f;
    float t = t_enter;

    while (t < t_exit) {
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) break;

        float t_next = fminf(fminf(t_max_x, t_max_y), fminf(t_max_z, t_exit));
        float seg_len = t_next - t;

        if (seg_len > 0.0f) {
            int linear_idx = iz * (ny * nx) + iy * nx + ix;
            float val = vol[linear_idx];

            if (val < hu_air_threshold) {
                val = 0.0f;
            } else if (clamp_negative_to_zero && val < 0.0f) {
                val = 0.0f;
            }

            integral += val * seg_len;
        }

        t = t_next;

        int crossed_x = (t_max_x <= t + eps);
        int crossed_y = (t_max_y <= t + eps);
        int crossed_z = (t_max_z <= t + eps);

        if (crossed_x) { ix += step_x; t_max_x += t_delta_x; }
        if (crossed_y) { iy += step_y; t_max_y += t_delta_y; }
        if (crossed_z) { iz += step_z; t_max_z += t_delta_z; }
    }

    out[idx] = integral;
}
'''


def render_drr_cuda(
    vol: Volume,
    geom: DRRGeometry,
    hu_air_threshold: Optional[float] = -900.0,
    clamp_negative_to_zero: bool = True,
    stream=None,
) -> np.ndarray:
    cp = _require_cupy()
    kernel = cp.RawKernel(_CUDA_SRC, "siddon_drr_kernel")

    gpu_vol = upload_volume_to_gpu(vol)
    det_pts = detector_pixel_centers_world(geom).astype(np.float32)
    H, W, _ = det_pts.shape
    N = H * W

    det_pts_gpu = cp.asarray(det_pts.reshape(-1, 3), dtype=cp.float32)
    out_gpu = cp.zeros((N,), dtype=cp.float32)

    sz, sy, sx = [float(x) for x in vol.spacing_zyx]
    oz, oy, ox = [float(x) for x in vol.origin_zyx]
    nz, ny, nx = [int(x) for x in vol.data.shape]
    source_x, source_y, source_z = [float(x) for x in geom.source_mm]

    threads = 256
    blocks = (N + threads - 1) // threads

    args = (
        gpu_vol["data"],
        np.int32(nz),
        np.int32(ny),
        np.int32(nx),
        np.float32(sz),
        np.float32(sy),
        np.float32(sx),
        np.float32(oz),
        np.float32(oy),
        np.float32(ox),
        det_pts_gpu,
        np.float32(source_x),
        np.float32(source_y),
        np.float32(source_z),
        np.int32(N),
        np.float32(-900.0 if hu_air_threshold is None else hu_air_threshold),
        np.int32(1 if clamp_negative_to_zero else 0),
        out_gpu,
    )

    if stream is None:
        kernel((blocks,), (threads,), args)
    else:
        kernel((blocks,), (threads,), args, stream=stream)

    out = cp.asnumpy(out_gpu).reshape(H, W)
    return out.astype(np.float32)
