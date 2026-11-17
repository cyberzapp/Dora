"""Experimental spatial correction around a generated chunk boundary only."""
import cv2
import numpy as np

from head_stabilization import transforms


def damp_joins(samples, size, chunk_frames):
    """Taper stronger 2D smoothing in/out; keep the existing path elsewhere.

    Rotations and displacements are mixed at the tracked eye center, rather
    than mixing matrices (which would introduce scale). No frames are retimed.
    """
    if type(chunk_frames) is not int or chunk_frames < 17:
        raise ValueError('Use a verified generated chunk size of at least 17 frames')
    base, base_report = transforms(samples, size)
    stronger, _ = transforms(samples, size, window=17, max_translation=6, max_rotation=1.5)
    centers = np.asarray(samples, dtype=np.float64)[:, 2:] * size
    result = base.copy()
    radius = 7
    changed = []
    for join in range(chunk_frames, len(samples), chunk_frames):
        for index in range(max(0, join-radius+1), min(len(samples), join+radius)):
            weight = (1 + np.cos(np.pi*(index-join)/radius))/2
            first, second = base[index], stronger[index]
            center = centers[index]
            a, b = (np.arctan2(m[0, 1], m[0, 0]) for m in (first, second))
            angle = np.rad2deg(a + weight * (b-a))
            target = (1-weight)*(first[:, :2] @ center + first[:, 2]) + weight*(second[:, :2] @ center + second[:, 2])
            matrix = cv2.getRotationMatrix2D(tuple(center), float(angle), 1)
            matrix[:, 2] += target-center
            result[index] = matrix
            changed.append(index)
    return result, dict(mode='localized-join-damping', base=base_report,
                        stronger_window_frames=17, stronger_max_translation_pixels=6,
                        stronger_max_rotation_degrees=1.5, joins=list(range(chunk_frames,len(samples),chunk_frames)),
                        changed_frames=changed,
                        interpretation='Experimental 2D rigid correction. No lip retiming, blink replacement, or 3D pose editing.')
