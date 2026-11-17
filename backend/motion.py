"""Local LivePortrait adapter using MediaPipe instead of InsightFace detector weights.

This adapter retains LivePortrait's 203-point refinement and paste-back pipeline.
MediaPipe supplies the initial crop; no commercial-restricted InsightFace weights load.
"""
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]

def install_detector():
    import cv2
    import mediapipe as mp
    import numpy as np
    from src.utils import crop
    original = crop.parse_pt2_from_pt_x
    def orientation(points, use_lip=True):
        if len(points) != 468:
            return original(points, use_lip=use_lip)
        left = points[[33, 133, 159, 145]].mean(axis=0)
        right = points[[362, 263, 386, 374]].mean(axis=0)
        lips = points[[61, 291]].mean(axis=0)
        return np.stack([(left+right)/2, lips] if use_lip else [left, right]).astype(np.float32)
    crop.parse_pt2_from_pt_x = orientation
    class MediaPipeDetector:
        def __init__(self, **kwargs):
            self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=.5)
        def prepare(self, **kwargs): pass
        def warmup(self): pass
        def get(self, image, **kwargs):
            result = self.mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks: return []
            height, width = image.shape[:2]
            points = np.array([[p.x*width, p.y*height] for p in result.multi_face_landmarks[0].landmark], dtype=np.float32)
            # The upstream property name is retained, but crop dispatch uses actual point count.
            return [types.SimpleNamespace(landmark_2d_106=points, bbox=np.array([*points.min(0), *points.max(0)]))]
    module = types.ModuleType('src.utils.face_analysis_diy')
    module.FaceAnalysisDIY = MediaPipeDetector
    sys.modules[module.__name__] = module


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--driving', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    checkout = ROOT / 'vendor/LivePortrait'
    sys.path.insert(0, str(checkout)); os.chdir(checkout)
    install_detector()
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    # Keep ONNX landmark refinement on CPU, neural animation on Metal.
    import src.utils.cropper as cropper
    runner = cropper.HumanLandmark
    class CPULandmark(runner):
        def __init__(self, **kwargs):
            kwargs['onnx_provider'] = 'cpu'
            super().__init__(**kwargs)
    cropper.HumanLandmark = CPULandmark
    config = ArgumentConfig(source=args.source, driving=args.driving, output_dir=args.output,
                            flag_crop_driving_video=True, flag_use_half_precision=False,
                            flag_pasteback=True, driving_multiplier=.5)
    def subset(cls): return cls(**{k:v for k,v in vars(config).items() if k in cls.__dataclass_fields__})
    pipeline = LivePortraitPipeline(inference_cfg=subset(InferenceConfig), crop_cfg=subset(CropConfig))
    pipeline.execute(config)

if __name__ == '__main__': main()
