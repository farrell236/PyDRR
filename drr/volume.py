from dataclasses import dataclass
from typing import Tuple

import numpy as np
import SimpleITK as sitk


@dataclass
class Volume:
    """Axis-aligned 3D volume container.

    Attributes:
        data: Volume intensities with shape (Z, Y, X).
        spacing_zyx: Voxel spacing in mm as (sz, sy, sx).
        origin_zyx: World origin in mm as (oz, oy, ox).
        direction: Original SimpleITK direction matrix. Stored for reference only;
            the phase-1 projector currently assumes axis alignment.
    """

    data: np.ndarray
    spacing_zyx: np.ndarray
    origin_zyx: np.ndarray
    direction: np.ndarray

    @property
    def shape_zyx(self) -> Tuple[int, int, int]:
        return self.data.shape


def load_volume_sitk(path: str) -> Volume:
    """Load a 3D volume from disk with SimpleITK.

    Returns:
        Volume with array data in (Z, Y, X) order.
    """
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)

    spacing_xyz = np.array(img.GetSpacing(), dtype=np.float32)
    origin_xyz = np.array(img.GetOrigin(), dtype=np.float32)
    direction = np.array(img.GetDirection(), dtype=np.float32).reshape(3, 3)

    return Volume(
        data=arr,
        spacing_zyx=spacing_xyz[::-1].copy(),
        origin_zyx=origin_xyz[::-1].copy(),
        direction=direction,
    )


def voxel_zyx_to_world_xyz(
    ijk_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    origin_zyx: np.ndarray,
) -> np.ndarray:
    """Approximate voxel index -> world xyz conversion for axis-aligned volumes."""
    ijk_zyx = np.asarray(ijk_zyx, dtype=np.float32)
    return ijk_zyx[..., ::-1] * spacing_zyx[::-1] + origin_zyx[::-1]


def world_xyz_to_voxel_zyx(
    xyz_mm: np.ndarray,
    spacing_zyx: np.ndarray,
    origin_zyx: np.ndarray,
) -> np.ndarray:
    """Approximate world xyz -> voxel index conversion for axis-aligned volumes."""
    xyz_mm = np.asarray(xyz_mm, dtype=np.float32)
    return (xyz_mm[..., ::-1] - origin_zyx) / spacing_zyx


def volume_center_world_xyz(vol: Volume) -> np.ndarray:
    """Return the center of the volume in world xyz coordinates."""
    shape_zyx = np.array(vol.shape_zyx, dtype=np.float32)
    center_zyx = (shape_zyx - 1.0) / 2.0
    return voxel_zyx_to_world_xyz(center_zyx[None], vol.spacing_zyx, vol.origin_zyx)[0]
