"""Bounded rigid stabilization; no expression transfer or new facial synthesis."""
import cv2
import numpy as np
from scipy.signal import savgol_filter


def transforms(samples,size,window=9,max_translation=3.0,max_rotation=.75):
    values=np.asarray(samples,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=4 or len(values)<window or not np.isfinite(values).all():
        raise ValueError('Complete finite motion samples are required')
    if not np.array_equal(values[:,0],np.arange(len(values))):
        raise ValueError('Motion samples have missing or repeated frames')
    if window<3 or window%2!=1 or min(max_translation,max_rotation)<0:
        raise ValueError('Invalid stabilization bounds')
    width,height=size
    pose=values[:,1:].copy();pose[:,0]=np.rad2deg(np.unwrap(np.deg2rad(pose[:,0])))
    pose[:,1:]*=[width,height]
    smooth=savgol_filter(pose,window,2,axis=0,mode='interp')
    rotation=np.clip(pose[:,0]-smooth[:,0],-max_rotation,max_rotation)
    shift=smooth[:,1:]-pose[:,1:]
    length=np.linalg.norm(shift,axis=1)
    shift*=np.minimum(1,max_translation/np.maximum(length,1e-12))[:,None]
    result=[]
    for index,(angle,center) in enumerate(zip(rotation,pose[:,1:])):
        matrix=cv2.getRotationMatrix2D(tuple(center),float(angle),1)
        matrix[:,2]+=shift[index];result.append(matrix)
    return np.stack(result),dict(window_frames=window,max_translation_pixels=max_translation,
        max_rotation_degrees=max_rotation,observed_max_translation_pixels=float(np.linalg.norm(shift,axis=1).max()),
        observed_max_rotation_degrees=float(abs(rotation).max()),
        interpretation='Small rigid RGB-coordinate corrections only. Mouth timing and nonrigid expressions are not changed; naturalness still requires playback review.')
