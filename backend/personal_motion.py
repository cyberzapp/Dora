"""Blend measured personal pose/eyes with an audio-driven LivePortrait mouth.

Semantic region indices follow the pinned official LivePortrait animation_region
implementation. This is reference-motion transfer, not personalized training.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

LIP_POINTS = (6, 12, 14, 17, 19, 20)
EYE_POINTS = (11, 13, 15, 16, 18)


def copy_info(info):
    return {key: value.detach().clone() if isinstance(value, torch.Tensor) else value for key, value in info.items()}


class SourceAnchoredMouth:
    """Experimental absolute mouth transfer with source pose and other expressions.

    Consumed by KJ's single_frame branch, not the relative-first-frame branch.
    This constrains keypoints, not exact output pixels or perceived realism.
    """
    def __init__(self, source):
        self.source=copy_info(source);self.index=0
        if self.source['exp'].shape!=(1,21,3) or not torch.isfinite(self.source['exp']).all():
            raise ValueError('Invalid source expression geometry')

    def __call__(self, driving):
        expression=driving['exp']
        if expression.shape!=(1,21,3) or not torch.isfinite(expression).all():
            raise ValueError('Invalid driving expression geometry')
        result=copy_info(self.source)
        result['exp'][:,LIP_POINTS,:]=expression[:,LIP_POINTS,:].to(result['exp'])
        self.index+=1
        return result


class PersonalMotionMixer:
    def __init__(self, reference, pose_strength=.65, expression_strength=.65, position_strength=.15):
        if not reference: raise ValueError('No personal reference motion')
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in (pose_strength, expression_strength, position_strength)):
            raise ValueError('Motion strengths must be between zero and one')
        self.reference = [copy_info(info) for info in reference]
        self.pose_strength, self.expression_strength, self.position_strength = pose_strength, expression_strength, position_strength
        self.audio_base = None; self.index = 0
        for info in self.reference:
            if info['exp'].shape != (1, 21, 3) or not torch.isfinite(info['exp']).all():
                raise ValueError('Invalid personal expression geometry')
        angles = [[float(info[key].reshape(-1)[0]) for key in ('pitch','yaw','roll')] for info in self.reference]
        # KJ's camera matrix is scipy's xyz Euler rotation, transposed.
        matrices = Rotation.from_euler('xyz', angles, degrees=True).as_matrix().transpose(0,2,1)
        relative = matrices @ matrices[0].T
        limited = Rotation.from_rotvec(Rotation.from_matrix(relative).as_rotvec()*pose_strength).as_matrix() @ matrices[0]
        self.angles = Rotation.from_matrix(limited.transpose(0,2,1)).as_euler('xyz',degrees=True)

    def __call__(self, audio_info):
        if self.index >= len(self.reference): raise ValueError('Personal reference is shorter than the generated performance; refusing a motion loop')
        if self.audio_base is None: self.audio_base = copy_info(audio_info)
        reference = self.reference[self.index]; initial = self.reference[0]
        result = copy_info(audio_info); base = self.audio_base['exp']
        change = (reference['exp']-initial['exp']).to(base)
        expression = base + self.expression_strength*change
        # Preserve complete recorded blinks, while retaining all audio mouth coordinates.
        expression[:, EYE_POINTS, :] = base[:, EYE_POINTS, :] + change[:, EYE_POINTS, :]
        expression[:, LIP_POINTS, :] = audio_info['exp'][:, LIP_POINTS, :]
        result['exp'] = expression
        for axis,key in enumerate(('pitch','yaw','roll')):
            result[key] = audio_info[key].new_tensor(self.angles[self.index,axis]).reshape_as(audio_info[key])
        for key in ('scale','t'):
            result[key] = (initial[key]+self.position_strength*(reference[key]-initial[key])).to(audio_info[key])
        self.index += 1
        return result
