"""Verify decoded source audio and every exported frame; no quality verdict."""
import argparse
from collections import deque
from fractions import Fraction
import hashlib
import re
import subprocess
import imageio_ffmpeg


def verify_video(path, expected_frames=None, fps=None):
    """Decode every frame and check integer presentation times in bounded memory."""
    process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-v','info','-xerror',
        '-err_detect','explode','-i',str(path),'-map','0:v:0','-an','-vf','showinfo',
        '-nostats','-fps_mode','passthrough','-f','null','-'],
        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    time_base=None;decoded=0;previous=None;diagnostics=deque(maxlen=8)
    target_step=Fraction(1)/Fraction(str(fps)) if fps is not None else None
    try:
        for line in process.stderr:
            diagnostics.append(line.strip())
            if 'showinfo' not in line:continue
            config=re.search(r'config in time_base:\s*(\d+)/(\d+)',line)
            if config:time_base=Fraction(int(config[1]),int(config[2]))
            frame=re.search(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:',line)
            if not frame:continue
            if int(frame[1])!=decoded or time_base is None:
                raise RuntimeError('Video frame timestamp information is incomplete')
            stamp=int(frame[2])*time_base
            if previous is not None and stamp<=previous:
                raise RuntimeError(f'Non-increasing presentation time at frame {decoded}')
            if target_step is not None:
                # Integer timestamps can round by half one time-base tick.
                # Check the absolute grid too, so small per-frame errors cannot accumulate.
                if abs(stamp-decoded*target_step)>time_base/2:
                    raise RuntimeError(f'Irregular video presentation time at frame {decoded}')
            previous=stamp;decoded+=1
        if process.wait():
            raise RuntimeError('Video decode failed: '+'; '.join(diagnostics))
    finally:
        if process.poll() is None:
            process.kill();process.wait()
        process.stderr.close()
    if decoded<=0 or (expected_frames is not None and decoded!=expected_frames):
        raise RuntimeError(f'Incomplete video: decoded {decoded}, expected {expected_frames}')
    return decoded


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('original_audio');parser.add_argument('export')
    parser.add_argument('--seconds',type=float,help='Compare this source excerpt instead of the full audio')
    parser.add_argument('--expected-frames',type=int)
    parser.add_argument('--fps',type=float)
    args=parser.parse_args()
    if args.seconds is not None and args.seconds<=0:parser.error('seconds must be positive')
    def audio_digest(path, seconds=None):
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',path,'-map','0:a:0']
        if seconds is not None:command+=['-t',str(seconds)]
        process=subprocess.Popen(command+['-f','f32le','-'],stdout=subprocess.PIPE)
        digest=hashlib.sha256();length=0
        while block:=process.stdout.read(1024*1024):digest.update(block);length+=len(block)
        if process.wait():raise RuntimeError('Audio decode failed')
        if length==0:raise RuntimeError('Export contains no decoded audio')
        return digest.hexdigest(),length
    if audio_digest(args.original_audio,args.seconds)!=audio_digest(args.export):raise RuntimeError('Decoded audio changed or has the wrong length')
    reader=imageio_ffmpeg.read_frames(args.export,pix_fmt='rgb24')
    try:
        metadata=next(reader)
        fps=metadata['fps']
    finally:reader.close()
    decoded=verify_video(args.export,args.expected_frames,args.fps)
    if args.fps is not None and abs(fps-args.fps)>.01:raise RuntimeError(f'Incorrect fps: {fps}')
    timing=' with regular presentation times' if args.fps is not None else ''
    print(f'Decoded audio unchanged; all {decoded} frames decoded at {fps:g} fps{timing}. Visual quality not assessed.')


if __name__=='__main__':main()
