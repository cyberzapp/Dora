"""Compare an actual Dub-It result to its real-camera conditioning input.

The camera input retains old mouth motion; it is not a generated/synced result.
No automatic perceptual acceptance, frame stabilization or audio retiming.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from download_model_manifest import digest
from compare_flashhead_lead import export_side_by_side


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True,type=lambda s:Path(s).resolve())
    parser.add_argument('--output',required=True,type=lambda s:Path(s).resolve())
    args=parser.parse_args()
    report=json.loads((args.candidate/'provenance.json').read_text())
    movie=args.candidate/'review.mov'
    lock=json.loads((ROOT/'research/dubit-source-lock.json').read_text())
    if (report.get('generated_video') is not True or report.get('adapter_inference_performed') is not True
            or report.get('adapter')!=lock or report.get('original_decoded_audio_verified') is not True
            or report.get('fps')!=25 or report.get('frames') not in (49,121)
            or report.get('output_sha256')!=digest(movie)):
        raise ValueError('A completed genuine pinned Dub-It generation is required')
    if set(report.get('adapter_binding_evidence',{}))!={'1','2'}:
        raise ValueError('Both actual adapter-binding stages are required')
    reference=report['reference'];camera=Path(reference['output']);audio=Path(reference['audio'])
    if digest(camera)!=reference['output_sha256'] or digest(audio)!=reference['source_audio_sha256']:
        raise ValueError('Camera conditioning or source audio changed')
    frames=report['frames']
    seconds=frames/25
    for path in (camera,movie):
        subprocess.run([sys.executable,str(ROOT/'scripts/verify_export.py'),str(audio),str(path),
            '--seconds',str(seconds),'--expected-frames',str(frames),'--fps','25'],check=True)
    args.output.mkdir(parents=True,exist_ok=False)
    mouth_held = reference.get('derived_conditioning', {}).get('driving_mouth_keypoints_held') is True
    source_label = ('KJ input held mouth' if mouth_held else 'KJ photo motion input') if reference.get('conditioning_kind')=='kj-photo-motion' else 'Camera input old mouth'
    labels=(source_label,'DubIt generated mouth')
    export_side_by_side([camera,movie],audio,seconds,frames,args.output/'comparison.mov',labels)
    needed=[round(i*(frames-1)/6) for i in range(7)]
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
    sheet=Image.new('RGB',(7*228,2*268),(20,21,25));draw=ImageDraw.Draw(sheet)
    for row,path in enumerate((camera,movie)):
        capture=cv2.VideoCapture(str(path))
        try:
            for index in range(frames):
                ok,pixels=capture.read()
                if not ok:raise ValueError('Incomplete matched-frame review')
                if index in needed:
                    column=needed.index(index);x=column*228;y=row*268
                    draw.text((x+4,y+4),source_label if row==0 else 'DubIt generated',font=font,fill='white')
                    draw.text((x+4,y+23),f'Frame {index} | {index/25:.2f}s',font=font,fill='white')
                    image=Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB))
                    sheet.paste(ImageOps.contain(image,(224,224),Image.Resampling.LANCZOS),(x+2,y+43))
        finally:capture.release()
    sheet.save(args.output/'matched-frames.jpg',quality=95)
    review=dict(generated_video=str(movie),generated_video_sha256=digest(movie),
        camera_input=str(camera),camera_input_sha256=digest(camera),
        labels=labels,original_audio_preserved=True,frames_retimed=False,
        full_clip_seconds=seconds,quality_passed=False,
        driving_mouth_keypoints_held=mouth_held,
        limitation='The conditioning reference is a visual input. Playback review is required for lip timing and realism.')
    (args.output/'review.json').write_text(json.dumps(review,indent=2));print(json.dumps(review),flush=True)


if __name__=='__main__':main()
