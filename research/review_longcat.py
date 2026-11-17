"""Prepare actual LongCat exports for visual review after generation finishes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--photo',type=Path,required=True)
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--face-metadata',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait-seconds',type=int,default=0)
    args=parser.parse_args();started=time.monotonic();report=None
    print('Waiting for the actual completed LongCat export; no model loaded.',flush=True)
    while report is None:
        try:report=json.loads((args.benchmark/'report.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):pass
        if report is not None:break
        try:progress=json.loads((args.benchmark/'progress.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):progress={}
        if progress.get('stage') in ('failed','cancelled'):
            raise RuntimeError('Render did not complete. Review its actual failure/cancellation before restarting anything.')
        if time.monotonic()-started>=args.wait_seconds:raise TimeoutError('No completed export yet; inspect the live render before taking any action.')
        time.sleep(5)
    if not report.get('generated_video') or not report.get('full_frame_decode_passed'):
        raise ValueError('No structurally verified generated video')
    request=report['request']
    if request['image_sha256']!=digest(args.photo) or request['audio_sha256']!=digest(args.audio):
        raise ValueError('This full-photo review requires the actual benchmark source photo and audio')
    if args.output.exists():raise ValueError('Use a new review directory')
    args.output.mkdir(parents=True)
    preserved=args.output/'source-preserved.mov'
    subprocess.run([sys.executable,ROOT/'research/preserve_photo.py','--candidate',report['output'],
                    '--photo',args.photo,'--face-metadata',args.face_metadata,'--audio',args.audio,
                    '--output',preserved],check=True)
    subprocess.run([sys.executable,ROOT/'research/appearance_review.py','--photo',args.photo,
                    '--candidate','LongCat raw • actual render='+report['output'],
                    '--candidate','LongCat • source preserved='+str(preserved),
                    '--reference',args.reference,'--reference-end-seconds','13',
                    '--output',args.output/'appearance'],check=True)
    result=dict(review_assets_created=True,quality_passed=False,playback_and_human_review_required=True,
                actual_generated_candidate=report['output'],post_composite=str(preserved.resolve()),
                benchmark_report_sha256=digest(args.benchmark/'report.json'),
                note='Post-composite has one additional video encode. Reference samples exclude its end fade and are not speech-synchronized.')
    (args.output/'review-state.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
