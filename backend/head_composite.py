"""Retain source appearance outside an explicitly bounded animated head region."""
import numpy as np
from PIL import ImageOps
import json
from pathlib import Path


def crop_window(size, box):
    """Taper a projected crop inward, keeping pixels outside it fixed."""
    width,height=size;left,top,right,bottom=box
    if not (0<=left<right<=width and 0<=top<bottom<=height):raise ValueError('Crop outside source')
    yy,xx=np.mgrid[top:bottom,left:right]
    distance=np.minimum.reduce([xx+.5-left,right-xx-.5,yy+.5-top,bottom-yy-.5])
    fraction=np.clip(distance/(min(right-left,bottom-top)*.1),0,1)
    result=np.zeros((height,width),dtype=np.float32)
    result[top:bottom,left:right]=.5-.5*np.cos(np.pi*fraction)
    return result


def crop_box(photo, bounds, factor=2.7):
    head_mask(photo, photo.size, bounds)
    if not np.isfinite(factor) or not 2 <= factor <= 4: raise ValueError('Crop factor must be between 2 and 4')
    x, y, w, h = bounds
    width, height = photo.size
    face_w, face_h = w * width, h * height
    side = min(width, height, int(np.ceil(max(face_w, face_h) * factor)))
    cx = (x + w / 2) * width
    top = (1 - y - h) * height
    cy = top + face_h * .45
    left = max(0, min(width - side, round(cx - side / 2)))
    upper = max(0, min(height - side, round(cy - side / 2)))
    return (left, upper, left + side, upper + side)


def source_face_bounds(request):
    bounds = request.get('face_bounds')
    if bounds:
        return bounds
    # Compatibility with already-prepared native outfit photos.
    photo = Path(request['photo'])
    if photo.stem.endswith('-prepared'):
        metadata = photo.with_name(photo.stem.removesuffix('-prepared')+'-landmarks.json')
        if metadata.is_file():
            return json.loads(metadata.read_text())['face']
    raise ValueError('Face preparation metadata is missing. Reimport the outfit photo before rendering with source-body preservation.')


def head_mask(photo, size, face_bounds):
    """Vision normalized bottom-left face bounds, mapped through aspect fitting.

    Feathering is confined to the head/neck region. This is not segmentation or
    a guarantee that a generated face is realistic; the result needs review.
    """
    bounds = np.asarray(face_bounds, dtype=np.float64)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError('Valid source face bounds are required for body preservation')
    x, y, w, h = bounds
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > 1 or y+h > 1:
        raise ValueError('Source face bounds extend outside the photograph')
    fitted = ImageOps.contain(photo, size)
    fw, fh = fitted.size; width, height = size
    ox, oy = int((width-fw)*.5), int((height-fh)*.5)
    left, top, face_w, face_h = ox+x*fw, oy+(1-y-h)*fh, w*fw, h*fh
    cx = left + face_w*.5
    # Extend above the face for hair and below it for the neck, not the torso.
    upper, lower = top-face_h*.75, top+face_h*1.65
    cy, ry = (upper+lower)*.5, (lower-upper)*.5
    rx = face_w*1.2
    yy, xx = np.mgrid[:height, :width]
    radius = np.sqrt(((xx+.5-cx)/rx)**2 + ((yy+.5-cy)/ry)**2)
    transition = np.clip((1-radius)/.15, 0, 1)
    mask = (.5-.5*np.cos(np.pi*transition)).astype(np.float32)
    return mask


def composite_head(source, generated, mask):
    source, generated = np.asarray(source), np.asarray(generated)
    if source.shape != generated.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('Source and generated RGB frame dimensions must match')
    if source.dtype != np.uint8 or generated.dtype != np.uint8:
        raise ValueError('Expected RGB uint8 frames')
    if mask.shape != source.shape[:2] or not np.isfinite(mask).all() or mask.min()<0 or mask.max()>1:
        raise ValueError('Invalid head mask')
    alpha = mask[..., None]
    mixed = np.rint(source.astype(np.float32)*(1-alpha)+generated.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)
    mixed[mask == 0] = source[mask == 0]
    return mixed
