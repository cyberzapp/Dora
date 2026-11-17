"""Experimental zero-phase rigid correction of fast 2D head-path variations."""
import cv2
import numpy as np
from scipy.signal import butter,sosfiltfilt


def transforms(samples,size):
    values=np.asarray(samples,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=4 or len(values)<30 or not np.isfinite(values).all():
        raise ValueError('Complete finite motion samples are required')
    if not np.array_equal(values[:,0],np.arange(len(values))):raise ValueError('Noncontiguous tracking')
    width,height=size
    pose=values[:,1:].copy();pose[:,0]=np.rad2deg(np.unwrap(np.deg2rad(pose[:,0])))
    pose[:,1:]*=[width,height]
    # Forward/backward filtering avoids delaying the head against the voice.
    target=sosfiltfilt(butter(2,1.5,fs=25,output='sos'),pose,axis=0)
    angle_delta=pose[:,0]-target[:,0]
    correction_angle=2*np.tanh(angle_delta/2)
    delta=target[:,1:]-pose[:,1:];distance=np.linalg.norm(delta,axis=1)
    # Smooth saturation avoids sharp clipping transitions in the correction.
    delta*= (8*np.tanh(distance/8)/np.maximum(distance,1e-12))[:,None]
    result=[]
    for angle,center,shift in zip(correction_angle,pose[:,1:],delta):
        matrix=cv2.getRotationMatrix2D(tuple(center),float(angle),1)
        matrix[:,2]+=shift;result.append(matrix)
    desired_centers=pose[:,1:]+delta
    return np.stack(result),dict(mode='experimental-slow-head-path',filter='two-pole zero-phase Butterworth 1.5 Hz at 25 fps',
        max_translation_pixels=8,max_rotation_degrees=2,
        observed_max_translation_pixels=float(np.linalg.norm(delta,axis=1).max()),
        observed_max_rotation_degrees=float(abs(correction_angle).max()),
        observed_p95_original_center_step_pixels=float(np.percentile(np.linalg.norm(np.diff(pose[:,1:],axis=0),axis=1),95)),
        observed_p95_adjusted_center_step_pixels=float(np.percentile(np.linalg.norm(np.diff(desired_centers,axis=0),axis=1),95)),
        interpretation='2D rigid path correction only. No frame replacement, retiming, blink edit or mouth synthesis; 3D deformation can remain.')
