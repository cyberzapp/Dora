"""Review matched, verified FlashHead clips with and without an analysis lead-in."""
import argparse
import json
from pathlib import Path
import hashlib
import subprocess
import sys

import numpy as np
import imageio_ffmpeg
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def call(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def export_side_by_side(videos, audio, seconds, frames, output, labels=('Earlier render','Shifted generation')):
    """Single exact source recording and two temporally aligned performances."""
    if output.exists():raise ValueError('Use a fresh side-by-side output')
    partial=output.with_suffix('.partial.mov')
    filters=[]
    if len(labels)!=2 or any(not label.replace(' ','').isalnum() for label in labels):
        raise ValueError('Use two plain text comparison labels')
    for i,label in enumerate(labels):
        filters.append(f'[{i}:v]scale=960:540:force_original_aspect_ratio=decrease,'
                       f'pad=960:540:(ow-iw)/2:(oh-ih)/2,setsar=1,'
                       f'drawbox=x=0:y=0:w=iw:h=32:color=black@0.65:t=fill,'
                       f'drawtext=fontfile=/System/Library/Fonts/Supplemental/Arial.ttf:'
                       f'text={label}:x=12:y=7:fontsize=18:fontcolor=white[v{i}]')
    filters.append('[v0][v1]hstack=inputs=2:shortest=1[comparison]')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(videos[0]),'-i',str(videos[1]),
        '-i',str(audio),'-filter_complex',';'.join(filters),'-map','[comparison]','-map','2:a:0',
        '-t',str(seconds),'-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-c:a','pcm_f32le',str(partial)],check=True)
    call('scripts/verify_export.py',audio,partial,'--seconds',seconds,'--expected-frames',frames,'--fps',25)
    partial.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline_kind = parser.add_mutually_exclusive_group(required=True)
    baseline_kind.add_argument('--baseline', type=Path, help='Earlier research benchmark')
    baseline_kind.add_argument('--native-baseline', type=Path, help='Earlier verified native job')
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folders = [(args.native_baseline or args.baseline).resolve(),args.candidate.resolve()]
    candidate = json.loads((folders[1]/'report.json').read_text())
    required = ('source_sha256','crop','seed','seconds','frames','model_revision','audio_revision',
                'steps','shift','frames_per_chunk','motion_frames','output_frames_per_chunk','runtime_versions')
    if candidate.get('model') != 'FlashHead Pro' or candidate.get('generated_video') is not True:
        raise ValueError('A completed FlashHead candidate is required')
    if args.native_baseline:
        provenance = folders[0]/'provenance.json'
        native = json.loads(provenance.read_text()); identity = native['request']
        videos = [folders[0]/'output.mov',folders[1]/'native-framing.mov']
        composite = json.loads(videos[1].with_suffix('.json').read_text())
        reference = composite.get('native_reference') or {}
        if (reference.get('sha256') != digest(videos[0]) or reference.get('provenance_sha256') != digest(provenance)
                or composite['output_resolution'] != native['output_size'] or native['fps'] != 25
                or native['frames'] != candidate['frames'] or native['source_crop'] != candidate['crop']
                or identity['seed'] != candidate['seed'] or identity['recipe'] != 'pro4-shift5-history8-carry5-v1'
                or any(candidate['runtime_versions'].get(name) != version for name,version in identity['runtime'].items())):
            raise ValueError('Native comparison does not match the verified framing, timeline or inference setup')
        base = dict(candidate,analysis_lead_frames=0,output_chunk_join_frames=list(range(28,native['frames'],28)))
    else:
        base = json.loads((folders[0]/'report.json').read_text())
        if base.get('model') != 'FlashHead Pro' or base.get('generated_video') is not True:
            raise ValueError('A completed FlashHead baseline is required')
        if any(base[key] != candidate[key] for key in required):
            raise ValueError('Photo, speech, crop, sampling settings, or duration differ')
        videos = [folder/'source-body-stabilized.mov' for folder in folders]
        composites = [json.loads(video.with_suffix('.json').read_text()) for video in videos]
        if composites[0]['face_metadata_sha256'] != composites[1]['face_metadata_sha256']:
            raise ValueError('Composites used different source-face metadata')
        if any((meta['frames'], meta['seconds']) != (base['frames'],base['seconds']) for meta in composites):
            raise ValueError('Composite duration differs from generation report')
    reports = [base,candidate]
    if base.get('analysis_lead_frames',0) != 0 or candidate.get('analysis_lead_frames',0) <= 0:
        raise ValueError('Expected a zero-lead baseline and a shifted candidate')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for video in videos:
        call('scripts/verify_export.py', base['source_audio'], video, '--seconds', base['seconds'],
             '--expected-frames', base['frames'], '--fps', 25)
    export_side_by_side(videos, base['source_audio'], base['seconds'], base['frames'], output/'side-by-side.mov')
    expressions = output/'expressions.json'
    call('research/expression_review.py', *videos, '--frames', base['frames'], '--output', expressions)
    call('research/joint_motion_review.py', '--expressions', expressions, '--audio', base['source_audio'],
         '--output', output/'joint-motion-review.json')
    for name, video in zip(('baseline', 'shifted'), videos):
        call('research/syncnet_timing.py', '--video', video, '--seconds', base['seconds'],
             '--output', output/f'{name}-syncnet.json')
        if base['frames'] == 375:
            for start in (0,5,10):
                call('research/syncnet_timing.py', '--video', video, '--start', start, '--seconds', 5,
                     '--output', output/f'{name}-syncnet-{start:02d}s.json')
    call('research/appearance_review.py', '--photo', base['source_photo'],
         '--candidate', f'Original generation grid={videos[0]}',
         '--candidate', f'Shifted generation grid={videos[1]}',
         '--candidate-frames', base['frames'], '--rows', 5, '--output', output/'appearance')
    call('research/motion_contact_sheet.py', '--joint-motion', output/'joint-motion-review.json',
         '--appearance', output/'appearance/appearance-review.json', '--output', output/'motion-review-points.jpg')
    joined = []
    tracked = json.loads(expressions.read_text())['videos']
    for name, report, track in zip(('baseline','shifted'), reports, tracked):
        join_frames = report.get('output_chunk_join_frames', list(range(report['output_frames_per_chunk'],report['frames'],report['output_frames_per_chunk'])))
        transforms = {row['frame']: Rotation.from_euler('xyz',row['relative_xyz_degrees'],degrees=True) for row in track['samples']}
        def jump(frame):
            if frame not in transforms or frame-1 not in transforms:return None
            return float(np.rad2deg((transforms[frame-1].inv()*transforms[frame]).magnitude()))
        sync = json.loads((output/f'{name}-syncnet.json').read_text())
        joined.append(dict(name=name, analysis_lead_frames=report.get('analysis_lead_frames',0),
            estimated_pose_jumps_at_joins=[dict(frame=n, seconds=n/25, degrees=jump(n)) for n in join_frames],
            largest_adjacent_pose_change=track['largest_adjacent_pose_changes'][:1],
            longest_estimated_bilateral_closure_seconds=max((r['duration_seconds'] for r in track['bilateral_closure_runs']),default=0),
            missing_tracking_frames=track['missing_frames'], timing={k:sync[k] for k in ('estimated_offset_seconds','distance','confidence')}))
    summary = dict(candidates=joined, frames=base['frames'], original_audio_verified=True,
                   native_baseline=args.native_baseline is not None,
                   quality_passed=False, appearance_review_required=True,
                   interpretation='Lead-in changes model history and joins. Tracking and timing estimates do not establish perceptual realism.')
    (output/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
