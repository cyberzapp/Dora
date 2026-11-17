"""Use the bundled FFmpeg to probe normalized local reference videos.

The MLX port otherwise requires a separate ffprobe executable. This adapter
counts actual decoded frames and rejects unresolved camera display rotation.
"""
from contextlib import contextmanager
import imageio_ffmpeg


def probe_reference_video(path):
    from ltx_core_mlx.utils.ffmpeg import VideoInfo
    reader=imageio_ffmpeg.read_frames(path,pix_fmt='rgb24')
    try:metadata=next(reader)
    finally:reader.close()
    if metadata['size']!=metadata['source_size'] or metadata.get('rotate',0)!=0:
        raise ValueError('Normalize reference-camera rotation before conditioning')
    count,_=imageio_ffmpeg.count_frames_and_secs(path)
    if count<1 or metadata['fps']<=0:raise ValueError('Empty reference video')
    width,height=metadata['size']
    return VideoInfo(width=width,height=height,num_frames=count,fps=metadata['fps'],
        duration=count/metadata['fps'],has_audio=bool(metadata.get('audio_codec')))


@contextmanager
def bundled_video_probe():
    import ltx_core_mlx.utils.ffmpeg as ffmpeg
    import ltx_pipelines_mlx.utils.media_io as media
    import ltx_pipelines_mlx.lipdub as lipdub
    modules=(ffmpeg,media,lipdub)
    previous=[module.probe_video_info for module in modules]
    try:
        for module in modules:module.probe_video_info=probe_reference_video
        yield
    finally:
        for module,original in zip(modules,previous):module.probe_video_info=original
