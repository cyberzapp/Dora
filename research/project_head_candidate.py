"""Place actual generated crop frames on the real source body; no new inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import cv2
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from head_composite import head_mask,composite_head,crop_window


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data:=stream.read(1024*1024):value.update(data)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--face-metadata',type=Path)
    p.add_argument('--match-native-job',type=Path,help='Use the photo framing and stabilization from an actual verified native FlashHead job')
    p.add_argument('--stabilize-motion',type=Path,help='Complete motion_review.json for this generated crop')
    p.add_argument('--comparison-baseline',action='store_true',help='Use the same affine resampler with zero stabilization')
    p.add_argument('--damp-joins',action='store_true',help='Experimental stronger rigid correction near verified FlashHead joins')
    args=p.parse_args()
    if bool(args.face_metadata)==bool(args.match_native_job):p.error('Choose face metadata or a native job to match')
    if args.match_native_job and (args.stabilize_motion or args.comparison_baseline or args.damp_joins):p.error('Native matching uses the recorded native stabilization mode')
    if args.comparison_baseline and not args.stabilize_motion:p.error('Baseline comparison requires the motion file')
    if args.damp_joins and (not args.stabilize_motion or args.comparison_baseline):p.error('Join damping requires motion tracking and cannot be used with the zero-adjustment baseline')
    folder=args.benchmark.resolve();report=json.loads((folder/'report.json').read_text())
    if report.get('generated_video') is not True:raise ValueError('A completed genuine generation report is required')
    lead=report.get('analysis_lead_frames',0)
    if type(lead) is not int or not 0<=lead<=20 or (lead and report.get('model')!='FlashHead Pro'):
        raise ValueError('Invalid generated analysis lead-in')
    if lead and args.damp_joins:raise ValueError('Join damping is not defined for a shifted window grid')
    photo_path=Path(report['source_photo']);audio_path=Path(report['source_audio'])
    if report['source_sha256']!={'photo':digest(photo_path),'audio':digest(audio_path)}:raise ValueError('Source media changed')
    photo=ImageOps.exif_transpose(Image.open(photo_path)).convert('RGB');pixels=np.asarray(photo)
    box=report['crop'];layout=None;native_reference=None
    if args.match_native_job:
        from native_match_projection import match_native,track_raw
        layout,bounds,native_stabilization,native_reference=match_native(args.match_native_job,report,photo)
    else:
        bounds=json.loads(args.face_metadata.read_text())['face']
        mask=head_mask(photo,photo.size,bounds)*crop_window(photo.size,box)
    output=folder/'source-body.mov'
    if args.stabilize_motion:output=folder/('source-body-affine-baseline.mov' if args.comparison_baseline else 'source-body-stabilized.mov')
    if args.damp_joins:output=folder/'source-body-join-damping.mov'
    if layout is not None:output=folder/'native-framing.mov'
    if output.exists():raise ValueError('Projection already exists; retain previous review evidence')
    reader=imageio_ffmpeg.read_frames(report['output'],pix_fmt='rgb24');metadata=next(reader)
    mw,mh=metadata['size'];fps=metadata['fps']
    if abs(fps-25)>.001:raise ValueError('Unexpected inference frame rate')
    adjustments=None;stabilization=None
    if args.stabilize_motion:
        from head_stabilization import transforms
        motion=json.loads(args.stabilize_motion.read_text())['videos']
        motion=next(item for item in motion if Path(item['path']).resolve()==Path(report['output']).resolve())
        if motion['frames']!=report['frames'] or motion['missing_frames']:
            raise ValueError('Stabilization requires tracking for every generated frame')
        if args.damp_joins:
            if report.get('model')!='FlashHead Pro':raise ValueError('Verified FlashHead chunk metadata required for join damping')
            from join_damping import damp_joins
            adjustments,stabilization=damp_joins(motion['samples'],(mw,mh),report['output_frames_per_chunk'])
        else:
            adjustments,stabilization=transforms(motion['samples'],(mw,mh))
        stabilization['comparison_baseline']=args.comparison_baseline
        if args.comparison_baseline:adjustments[:]=np.eye(2,3)
    cache_paths=sorted(folder.glob('chunk-*.npy'));cache_hashes={}
    if report.get('model')=='FlashHead Pro':
        reader.close()
        chunk_frames=report['output_frames_per_chunk']
        expected_chunks=(lead+report['frames']+chunk_frames-1)//chunk_frames
        if [path.name for path in cache_paths]!=[f'chunk-{i:04d}.npy' for i in range(expected_chunks)]:
            raise ValueError('Missing or discontinuous generated RGB cache')
        def cached_frames():
            for index,path in enumerate(cache_paths):
                cache_hashes[path.name]=digest(path)
                values=np.load(path,mmap_mode='r',allow_pickle=False)
                if values.dtype!=np.uint8 or values.shape!=(chunk_frames,mh,mw,3):
                    raise ValueError('Invalid generated RGB cache')
                start=index*chunk_frames
                for frame in values[max(0,lead-start):min(chunk_frames,lead+report['frames']-start)]:yield frame
        if layout is not None and native_stabilization:
            adjustments,stabilization=track_raw(cached_frames(),report['frames'])
        frame_source=cached_frames()
    else:
        frame_source=(np.frombuffer(data,dtype=np.uint8).reshape(mh,mw,3) for data in reader)
    width,height=photo.size;pw=width+width%2;ph=height+height%2
    if layout is not None:pw,ph=layout.size
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    partial=output.with_suffix('.partial.mov')
    process=subprocess.Popen([ff,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{pw}x{ph}',
                              '-r','25','-i','-','-i',str(audio_path),'-map','0:v','-map','1:a',
                              '-t',str(report['seconds']),'-c:v','libx264','-crf','16',
                              '-pix_fmt','yuv420p','-c:a','pcm_f32le',str(partial)],stdin=subprocess.PIPE)
    count=0;left,top,right,bottom=box
    try:
        for frame in frame_source:
            if layout is not None:
                process.stdin.write(layout.frame(frame,None if adjustments is None else adjustments[count]).tobytes())
                count+=1;continue
            projected=pixels.copy()
            if adjustments is not None:
                # Compose stabilization and source projection into ONE spatial
                # resampling step, preserving detail and avoiding double blur.
                sx,sy=(right-left)/mw,(bottom-top)/mh
                scaling=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
                correction=np.eye(3);correction[:2]=adjustments[count]
                projected[top:bottom,left:right]=cv2.warpAffine(frame,(scaling@correction)[:2],
                    (right-left,bottom-top),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT101)
            else:
                generated=Image.fromarray(frame)
                projected[top:bottom,left:right]=np.asarray(generated.resize((right-left,bottom-top),Image.Resampling.LANCZOS))
            composed=composite_head(pixels,projected,mask)
            if not np.array_equal(composed[mask==0],pixels[mask==0]):raise RuntimeError('Source pixels changed outside animated region')
            padded=np.zeros((ph,pw,3),dtype=np.uint8);padded[:height,:width]=composed
            process.stdin.write(padded.tobytes());count+=1
    finally:
        reader.close();process.stdin.close();return_code=process.wait()
    if return_code or count!=report['frames']:raise RuntimeError('Incomplete projected export')
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_export.py'),str(audio_path),str(partial),
                    '--seconds',str(report['seconds']),'--expected-frames',str(count),'--fps','25'],check=True)
    partial.replace(output)
    output.with_suffix('.json').write_text(json.dumps(dict(source_report=str(folder/'report.json'),
        source_generated_video=report['output'],source_generated_sha256=digest(report['output']),
        output=str(output),frames=count,seconds=report['seconds'],crop=box,
        analysis_lead_frames=lead,output_chunk_join_frames=report.get('output_chunk_join_frames'),
        source_photo_sha256=report['source_sha256']['photo'],face_bounds=bounds,
        face_metadata_sha256=digest(args.face_metadata) if args.face_metadata else None,
        native_reference=native_reference,source_rgb_cache_sha256=cache_hashes,
        stabilization=stabilization,
        model_resolution=[mw,mh],output_resolution=[pw,ph],quality_passed=False,
        source_pixels_preserved_outside_head_before_encoding=True,
        note=('Direct projection of generated RGB caches; one video encode.' if cache_hashes else 'Projection of a decoded generated crop; an additional video encode.')+
             ' Body and surroundings use the uploaded photo; effective generated face detail remains at the model crop resolution.'),indent=2))


if __name__=='__main__':main()
