"""CPU float32 paste-back restricted to the model's actual synthesis mask."""
import cv2
import numpy as np


def composite_face(original, generated_face, affine, synthesis_mask, face_size):
    """Inputs are RGB uint8; affine maps original pixels into the aligned crop.

    Only the model's synthesis region is replaced. Feather inward so unrelated
    pixels outside the projected region remain exactly the original pixels.
    """
    width, height = face_size
    if not all(np.isfinite(value).all() for value in (generated_face, affine, synthesis_mask)):
        raise ValueError('Non-finite generated pixels, transform or synthesis mask')
    face = cv2.resize(generated_face.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(synthesis_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = np.clip(mask, 0, 1)
    distance = cv2.distanceTransform((mask > .5).astype(np.uint8), cv2.DIST_L2, 5)
    feather = max(2, round(min(width, height) * .025))
    alpha = np.minimum(distance / feather, 1) * mask
    inverse = cv2.invertAffineTransform(np.asarray(affine, dtype=np.float32).reshape(2, 3))
    h, w = original.shape[:2]
    projected_alpha = cv2.warpAffine(alpha, inverse, (w, h), flags=cv2.INTER_LINEAR)
    # Premultiply before warping to prevent black border contamination.
    projected_face = cv2.warpAffine(face * alpha[..., None], inverse, (w, h), flags=cv2.INTER_LINEAR)
    output = projected_face + original.astype(np.float32) * (1 - projected_alpha[..., None])
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)
