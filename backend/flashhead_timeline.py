"""Exact streaming timeline for the tested 33-frame / 5-frame-carry recipe."""
from dataclasses import dataclass

FPS=25
SAMPLE_RATE=16000
MODEL_FRAMES=33
MOTION_FRAMES=5
NEW_FRAMES=MODEL_FRAMES-MOTION_FRAMES
NEW_SAMPLES=NEW_FRAMES*SAMPLE_RATE//FPS
HISTORY_SAMPLES=8*SAMPLE_RATE


@dataclass(frozen=True)
class Window:
    index:int
    start_frame:int
    append_frames:int
    analysis_start:int
    analysis_end:int
    analysis_pad_after:int


def plan(analysis_samples,output_frames=None):
    if type(analysis_samples) is not int or not SAMPLE_RATE//2<=analysis_samples<=300*SAMPLE_RATE:
        raise ValueError('Use 0.5–300 seconds of valid 16 kHz analysis audio')
    # Keep the fractional final frame. Original audio is never padded for export.
    frames=(analysis_samples*FPS+SAMPLE_RATE-1)//SAMPLE_RATE
    if output_frames is not None:
        if type(output_frames) is not int or output_frames<=0 or abs(output_frames-frames)>1:
            raise ValueError('Original audio duration does not match the analysis timeline')
        frames=output_frames
    windows=[]
    for index,start in enumerate(range(0,frames,NEW_FRAMES)):
        audio_start=index*NEW_SAMPLES
        audio_end=min(analysis_samples,audio_start+NEW_SAMPLES)
        windows.append(Window(index,start,min(NEW_FRAMES,frames-start),audio_start,audio_end,
                              NEW_SAMPLES-(audio_end-audio_start)))
    return frames,windows
