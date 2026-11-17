"""Compare genuine LTX/FlashHead outputs on the same photo crop and source audio.

No inference, automatic visual approval, synthetic replacement frames or audio
retiming. Repeated-photo codec fixtures are deliberately ineligible candidates.
Both compared videos are model head crops, not final native body compositions.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageOps

from compare_flashhead_lead import call, digest, export_side_by_side
from ltx_audio_preview import GEMMA, MODEL, SOURCE_REVISION

ROOT=Path(__file__).resolve().parents[1]


def expected_crop(photo, crop):
    source=ImageOps.exif_transpose(Image.open(photo)).convert('RGB')
    left,top,right,bottom=crop
    if not (0<=left<right<=source.width and 0<=top<bottom<=source.height and right-left==bottom-top):
        raise ValueError('Invalid reference crop')
    return np.asarray(source.crop(crop).resize((512,512),Image.Resampling.LANCZOS))


def validate_control(folder, photo, audio, frames):
    report=json.loads((folder/'report.json').read_text())
    source=dict(photo=digest(photo),audio=digest(audio))
    if (report.get('generated_video') is not True or report.get('model')!='FlashHead Pro'
            or report.get('frames',0)<frames or report.get('analysis_lead_frames',0)!=0
            or report.get('source_sha256')!=source):
        raise ValueError('Requires the actual matching zero-lead FlashHead control')
    movie=folder/'review.mov'
    recorded=json.loads((folder/'source-body-stabilized.json').read_text())
    if (recorded['source_generated_sha256']!=digest(movie)
            or recorded['source_photo_sha256']!=source['photo']
            or recorded['frames']!=report['frames'] or recorded['crop']!=report['crop']):
        raise ValueError('The earlier generated control differs from its retained provenance')
    if not np.array_equal(expected_crop(photo,report['crop']),np.asarray(Image.open(folder/'input-crop.png').convert('RGB'))):
        raise ValueError('The control used different reference pixels')
    call('scripts/verify_export.py',audio,movie,'--seconds',report['seconds'],
         '--expected-frames',report['frames'],'--fps',25)
    return report,movie


def validate_candidate(folder, photo, audio):
    provenance=folder/'provenance.json'
    if not provenance.is_file():
        raise ValueError('Requires completed AI-generation provenance; codec fixtures are not candidates')
    report=json.loads(provenance.read_text())
    if (report.get('generated_video') is not True or report.get('original_decoded_audio_verified') is not True
            or report.get('model_revision')!=MODEL[1] or report.get('gemma_revision')!=GEMMA[1]
            or report.get('source_revision')!=SOURCE_REVISION
            or report.get('frames') not in (49,121) or report.get('fps')!=25
            or report.get('output_geometry')!=[512,512]
            or report.get('photo_sha256')!=digest(photo) or report.get('audio_sha256')!=digest(audio)):
        raise ValueError('Requires genuine completed generation with the pinned model and matching real inputs')
    movie=folder/'review.mov'
    if Path(report.get('output','')).resolve()!=movie.resolve() or report.get('output_sha256')!=digest(movie):
        raise ValueError('The candidate video differs from its completed generation record')
    if not np.array_equal(expected_crop(photo,report['crop']),np.asarray(Image.open(folder/'source-crop.png').convert('RGB'))):
        raise ValueError('The candidate used different reference pixels')
    if report.get('seconds')!=report['frames']/25:
        raise ValueError('The candidate duration differs from its frame count')
    call('scripts/verify_export.py',audio,movie,'--seconds',report['seconds'],
         '--expected-frames',report['frames'],'--fps',25)
    return report,movie


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,default=ROOT/'.local/flashhead-detail-five')
    parser.add_argument('--photo',type=Path,required=True)
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--external-reference',type=Path,help='HeyGen comparison only; not an app result or word-aligned performance')
    parser.add_argument('--validate-control-only',action='store_true')
    parser.add_argument('--track-primary-face',action='store_true',help='Use explicitly calibrated continuous presenter tracking for timing checks')
    args=parser.parse_args()
    args.control=args.control.resolve();args.photo=args.photo.resolve();args.audio=args.audio.resolve()
    if args.validate_control_only:
        report,movie=validate_control(args.control,args.photo,args.audio,49)
        print(json.dumps(dict(control_verified=True,video=str(movie),video_sha256=digest(movie),
            report_sha256=digest(args.control/'report.json'),source_sha256=report['source_sha256'],
            crop=report['crop'],frames=report['frames'],new_inference_performed=False,quality_passed=False)))
        return
    if args.candidate is None or args.output is None:parser.error('Provide candidate and output folders')
    candidate,video=validate_candidate(args.candidate.resolve(),args.photo,args.audio)
    control,baseline=validate_control(args.control,args.photo,args.audio,candidate['frames'])
    if control['crop']!=candidate['crop']:raise ValueError('Model reference crops differ')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    seconds=candidate['seconds'];frames=candidate['frames']
    export_side_by_side([baseline,video],args.audio,seconds,frames,output/'side-by-side.mov',
                       labels=('Earlier FlashHead','LTX candidate'))
    call('research/expression_review.py',baseline,video,'--frames',frames,'--output',output/'expressions.json')
    call('research/joint_motion_review.py','--expressions',output/'expressions.json','--audio',args.audio,
         '--output',output/'joint-motion.json')
    appearance=['--photo',args.photo,'--candidate',f'Earlier FlashHead={baseline}',
                '--candidate',f'LTX candidate={video}','--candidate-frames',frames,'--rows',4,'--output',output/'appearance']
    if args.external_reference:
        appearance+=['--reference',args.external_reference.resolve(),'--reference-end-seconds',12]
    call('research/appearance_review.py',*appearance)
    call('research/motion_contact_sheet.py','--joint-motion',output/'joint-motion.json',
         '--appearance',output/'appearance/appearance-review.json','--output',output/'motion-points.jpg')
    timing={}
    if seconds>=2:
        for name,path in [('flashhead',baseline),('ltx',video)]:
            command=[sys.executable,str(ROOT/'research/syncnet_timing.py'),'--video',str(path),
                '--seconds',str(seconds),'--output',str(output/f'{name}-syncnet.json')]
            if args.track_primary_face:command.append('--track-primary-face')
            result=subprocess.run(command,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            (output/f'{name}-syncnet.log').write_text(result.stdout+result.stderr)
            completed=result.returncode==0 and (output/f'{name}-syncnet.json').is_file()
            timing[name]=dict(completed=completed,exit_code=result.returncode)
            if not completed:
                timing[name]['error']=result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'Timing diagnostic produced no result'
    timing_complete=seconds>=2 and all(item['completed'] for item in timing.values())
    result=dict(quality_passed=False,visual_review_required=True,new_inference_performed=False,
        raw_model_crops=True,native_body_composition_validated=False,
        photo_sha256=digest(args.photo),audio_sha256=digest(args.audio),crop=candidate['crop'],frames=frames,seconds=seconds,
        control_video_sha256=digest(baseline),candidate_video_sha256=digest(video),
        candidate_provenance_sha256=digest(args.candidate.resolve()/'provenance.json'),
        shared_original_audio=True,audio_retimed=False,timing_diagnostic_performed=seconds>=2,
        timing_diagnostic_completed=timing_complete,timing_diagnostics=timing,
        primary_face_tracking_requested=args.track_primary_face,
        timing_note=('49 frames are below the timing tool minimum; no padding or score is invented' if seconds<2 else
            'Timing estimate is not perceptual acceptance' if timing_complete else 'Some timing checks are unavailable; see diagnostic records. No score is invented.'),
        external_reference_sha256=digest(args.external_reference) if args.external_reference else None,
        external_reference_word_aligned=False,comparison_sha256=digest(output/'side-by-side.mov'))
    (output/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
