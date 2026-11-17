#!/usr/bin/env python3
"""Offline local worker. JSON request + atomic status; no network inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference_lease import GPUInferenceLease

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'vendor/musetalk-mac/upstream'
WEIGHTS = ['musetalkV15/unet.pth', 'musetalkV15/musetalk.json',
           'sd-vae/config.json', 'sd-vae/diffusion_pytorch_model.bin',
           'whisper/config.json', 'whisper/pytorch_model.bin', 'whisper/preprocessor_config.json',
           'face-parse-bisent/79999_iter.pth', 'face-parse-bisent/resnet18-5c106cde.pth']
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.local/matplotlib'))

class Cancelled(Exception):
    pass

def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)

def validate_request(request):
    if request.get('consent') is not True:
        raise ValueError('Explicit likeness consent is required.')
    for key in (('source_video', 'audio') if request.get('source_video') else ('photo', 'audio')):
        path = Path(request.get(key, ''))
        if not path.is_file():
            raise ValueError(f'{key} file is missing: {path}')
    if request.get('motion') and not Path(request['motion']).is_file():
        raise ValueError('Motion reference is missing.')
    if not isinstance(request.get('portrait', False), bool):
        raise ValueError('portrait must be a Boolean')

class Job:
    def __init__(self, directory, attempt=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cancelled = False
        self.children = []
        self.started = time.monotonic()
        self.attempt = attempt
    def check(self):
        if self.cancelled or (self.directory / 'cancel').exists():
            raise Cancelled('Render cancelled')
    def status(self, progress, detail, status='Rendering', output=''):
        atomic_json(self.directory / 'status.json', dict(status=status, progress=progress, detail=detail, output=str(output), elapsed=round(time.monotonic()-self.started, 2), attempt=self.attempt))
        print(detail, flush=True)
    def run(self, args, cwd=None, on_poll=None):
        self.check()
        process = subprocess.Popen([str(a) for a in args], cwd=cwd, start_new_session=True)
        self.children.append(process)
        try:
            while process.poll() is None:
                self.check()
                if on_poll is not None:
                    on_poll()
                time.sleep(0.1)
            if process.returncode:
                raise RuntimeError(f'{Path(str(args[0])).name} exited with code {process.returncode}')
        finally:
            if process.poll() is None:
                # Model stages can start their own encoders. Cancel the whole
                # process group so an orphan cannot keep consuming Metal memory.
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
            self.children.remove(process)

def doctor():
    report = {'python': sys.version.split()[0], 'model_inference_verified': False}
    try:
        import torch
        report['torch'] = torch.__version__
        report['mps_available'] = torch.backends.mps.is_available()
        if report['mps_available']:
            x = torch.ones((8, 8), device='mps'); y = x @ x
            torch.mps.synchronize()
            report['metal_tensor_test'] = bool(y[0, 0].item() == 8)
        import mediapipe as mp
        report['mediapipe_face_mesh'] = hasattr(mp.solutions, 'face_mesh')
        report['ffmpeg'] = ffmpeg()
        report['missing_weights'] = [p for p in WEIGHTS if not (UPSTREAM / 'models' / p).is_file()]
        evidence = ROOT / '.local/last-success.json'
        if evidence.exists():
            report['last_successful_render'] = json.loads(evidence.read_text())
            report['model_inference_verified'] = report['last_successful_render'].get('torch') == torch.__version__
            report['current_process_can_use_mps'] = report['mps_available']
    except Exception as error:
        report['error'] = str(error)
    print(json.dumps(report, indent=2))
    return int('error' in report or bool(report.get('missing_weights')))

def animate(job, photo, motion):
    """Run LivePortrait separately so its model memory is released before MuseTalk."""
    python = ROOT / '.venv/bin/python'
    checkout = ROOT / 'vendor/LivePortrait'
    if not python.exists() or not (checkout / 'pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth').exists():
        raise RuntimeError('LivePortrait is not installed. Run scripts/install_motion.sh. Select Still pose for lip sync without reference motion.')
    job.status(.03, 'Transferring reference motion with local LivePortrait')
    staged_photo = job.directory / 'source.png'
    import shutil
    shutil.copyfile(photo, staged_photo)
    photo = staged_photo
    motion_dir = job.directory / 'motion'
    motion_dir.mkdir(exist_ok=True)
    # Bound driving reference and normalize timebase before the memory-heavy model.
    driving = job.directory / 'driving.mp4'
    job.run([ffmpeg(), '-y', '-i', motion, '-t', '15', '-an', '-vf', 'fps=25,scale=640:640:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2', str(driving)])
    bin_dir = ROOT / '.local/bin'; bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / 'ffmpeg'
    if not link.exists(): link.symlink_to(ffmpeg())
    os.environ['PATH'] = str(bin_dir) + os.pathsep + os.environ.get('PATH', '')
    job.run([python, ROOT / 'backend/motion.py', '--source', photo, '--driving', driving, '--output', motion_dir], cwd=checkout)
    candidates = [p for p in motion_dir.glob('*.mp4') if '_concat' not in p.stem]
    if len(candidates) != 1:
        raise RuntimeError('LivePortrait did not produce exactly one composited output video. See worker.log.')
    return candidates[0]

def render(request, job):
    validate_request(request)
    missing = [p for p in WEIGHTS if not (UPSTREAM / 'models' / p).is_file()]
    if missing:
        raise RuntimeError('Missing model files. Run scripts/install_runtime.sh. Missing: ' + ', '.join(missing))
    job.status(.01, 'Validating local inputs')
    import cv2
    import numpy as np
    import soundfile as sf
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('Metal/MPS is unavailable to this process. Launch the native app outside a restricted sandbox; no CUDA device is required.')
    # Preserve source audio untouched; only this derived analysis file is resampled.
    wav = job.directory / 'analysis.wav'
    job.run([ffmpeg(), '-y', '-v', 'error', '-i', request['audio'], '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav])
    info = sf.info(wav)
    if not 0.5 <= info.duration <= 300:
        raise ValueError('Use a voice-over between 0.5 and 300 seconds. Split longer lessons into segments.')
    photo = Path(request['photo']).resolve()
    direct_video = bool(request.get('source_video'))
    motion_video = None
    if direct_video:
        motion_video = job.directory / 'normalized-source.mp4'
        job.status(.03, 'Normalizing real video orientation and 25 fps timebase')
        job.run([ffmpeg(), '-y', '-v', 'error', '-i', request['source_video'], '-t', str(info.duration), '-an', '-vf', 'fps=25,scale=1280:1280:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-crf', '16', motion_video])
    elif request.get('motion'):
        motion_video = animate(job, photo, request['motion'])
    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    from musetalk.utils.utils import load_all_model
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.blending import get_image_blending, get_image_prepare_material
    from musetalk.utils.preprocessing import _landmarks_px
    from transformers import WhisperModel
    job.status(.08, 'Loading MuseTalk v1.5 • MPS • float32 • batch 1')
    device = torch.device('mps')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    vae, unet, pe = load_all_model(device=device)
    vae.vae = vae.vae.to(device).eval()
    unet.model = unet.model.to(device).eval()
    pe = pe.to(device).eval()
    processor = AudioProcessor(str(UPSTREAM / 'models/whisper'))
    whisper = WhisperModel.from_pretrained(str(UPSTREAM / 'models/whisper'), local_files_only=True).to(device).eval()
    fp = FaceParsing()
    timesteps = torch.tensor([0], device=device)
    source = cv2.imread(str(photo))
    if direct_video:
        probe = cv2.VideoCapture(str(motion_video)); ok, source = probe.read(); probe.release()
        if not ok: raise ValueError('Source video has no readable frames')
    if source is None: raise ValueError('Cannot decode the prepared photo. Import the original through Avatar Library.')
    capture = cv2.VideoCapture(str(motion_video)) if motion_video else None
    cap_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture else 0
    if capture and cap_count < 1: raise RuntimeError('Motion output has no readable frames')
    if direct_video and cap_count < math.ceil(info.duration * 25):
        raise ValueError('The real source video must be at least as long as the new voice-over. Import a longer recording; frames will not be reversed or silently repeated.')
    source_h, source_w = source.shape[:2]
    out_w, out_h = (720, 1280) if request.get('portrait') else (1280, 720)
    silent = job.directory / 'silent.mp4'
    encoder = subprocess.Popen([ffmpeg(), '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{out_w}x{out_h}', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(silent)], stdin=subprocess.PIPE)
    job.children.append(encoder)
    total = math.ceil(info.duration * 25)
    produced = 0
    latent = None
    box = None
    mask = crop_box = None
    try:
        with torch.inference_mode(), sf.SoundFile(wav) as audio:
            # 10-second chunks bound Whisper feature memory independently of video length.
            for start_frame in range(0, total, 250):
                job.check()
                end_frame = min(total, start_frame + 250)
                start_sample = max(0, int(start_frame / 25 * 16000) - 3200)
                end_sample = min(len(audio), int(end_frame / 25 * 16000) + 3200)
                audio.seek(start_sample)
                samples = audio.read(end_sample - start_sample, dtype='float32')
                features = processor.feature_extractor(samples, return_tensors='pt', sampling_rate=16000).input_features
                # Context is encoded once per bounded chunk. Frame slices use the upstream 50 Hz window.
                hidden = whisper.encoder(features.to(device), output_hidden_states=True).hidden_states
                hidden = torch.stack(hidden, dim=2)
                valid = math.floor(len(samples) / 16000 * 50)
                hidden = hidden[:, :valid]
                hidden = torch.cat([torch.zeros_like(hidden[:, :4]), hidden, torch.zeros_like(hidden[:, :12])], dim=1)
                for index in range(start_frame, end_frame):
                    job.check()
                    frame = source
                    if capture:
                        # Ping-pong the reference without discontinuous first/last jumps.
                        cycle = index % (2 * cap_count)
                        selected = index if direct_video else (cycle if cycle < cap_count else 2*cap_count-cycle-1)
                        if not direct_video or index == 0: capture.set(cv2.CAP_PROP_POS_FRAMES, selected)
                        ok, frame = capture.read()
                        if not ok: raise RuntimeError(f'Cannot decode motion frame {selected}')
                    if latent is None or capture:
                        points = _landmarks_px(frame)
                        if points is None: raise ValueError(f'Face tracking failed at frame {index}; use a clearer, forward-facing reference.')
                        h, w = frame.shape[:2]
                        x1, x2 = max(0, int(points[:, 0].min())), min(w, int(points[:, 0].max()))
                        bottom = int(points[:, 1].max())
                        top = max(0, int(2*points[6, 1]-bottom))
                        bottom = min(h, bottom+10)
                        if x2-x1 < 32 or bottom-top < 32: raise ValueError('Face crop is too small for lip synchronization.')
                        candidate = np.array([x1, top, x2, bottom], dtype=np.float32)
                        if direct_video and box is not None:
                            candidate = .65 * candidate + .35 * np.array(box)
                        box = np.rint(candidate).astype(int).tolist()
                        x1, top, x2, bottom = box
                        crop = cv2.resize(frame[top:bottom, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
                        latent = vae.get_latents_for_unet(crop).to(device)
                        mask, crop_box = get_image_prepare_material(frame, box, mode="jaw", fp=fp)
                        if direct_video and request.get("experimental_mouth_mask", False):
                            # Preserve moustache, cheeks and chin from real footage.
                            # Only the mouth aperture and immediate lip margin are replaced.
                            mouth = points[[61, 291, 0, 17, 13, 14]]
                            left, right = int(mouth[:, 0].min()), int(mouth[:, 0].max())
                            upper, lower = int(mouth[:, 1].min()), int(mouth[:, 1].max())
                            width = right-left
                            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                            center = ((left+right)//2, (upper+lower)//2)
                            axes = (max(2, int(width*.66)), max(3, int(max(lower-upper,width*.40)*.63)))
                            cv2.ellipse(full_mask, center, axes, 0, 0, 360, 255, -1)
                            # Don't replace the moustache above the upper lip.
                            full_mask[:max(0, upper-2)] = 0
                            full_mask = cv2.GaussianBlur(full_mask, (7,7), 0)
                            from PIL import Image
                            mask = np.asarray(Image.fromarray(full_mask).crop(crop_box))
                    audio_index = math.floor((index / 25 - start_sample / 16000) * 50)
                    conditioning = hidden[:, audio_index:audio_index+10]
                    if conditioning.shape[1] != 10: raise RuntimeError('Audio window alignment failed')
                    conditioning = conditioning.reshape(1, 50, 384)
                    prediction = unet.model(latent, timesteps, encoder_hidden_states=pe(conditioning)).sample
                    face = vae.decode_latents(prediction)[0]
                    x1, y1, x2, y2 = box
                    face = cv2.resize(face, (x2-x1, y2-y1))
                    composed = get_image_blending(frame, face, box, mask, crop_box)
                    # Fit instead of crop: preserve source shoulders, clothing, and composition.
                    h, w = composed.shape[:2]
                    scale = min(out_w/w, out_h/h)
                    fitted = cv2.resize(composed, (max(1, round(w*scale)), max(1, round(h*scale))))
                    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    rh, rw = fitted.shape[:2]; ox, oy = (out_w-rw)//2, (out_h-rh)//2
                    canvas[oy:oy+rh, ox:ox+rw] = fitted
                    encoder.stdin.write(canvas.tobytes())
                    produced += 1
                    if produced % 5 == 0 or produced == total:
                        job.status(.12 + .80*produced/total, f'Lip synchronization • frame {produced}/{total}')
                del hidden, features, conditioning
                torch.mps.empty_cache()
        encoder.stdin.close()
        if encoder.wait() != 0: raise RuntimeError('Video encoder failed')
    finally:
        if capture: capture.release()
        if encoder.poll() is None:
            encoder.terminate()
            try: encoder.wait(timeout=5)
            except subprocess.TimeoutExpired: encoder.kill(); encoder.wait()
        job.children.remove(encoder)
    job.check()
    job.status(.95, 'Muxing original voice-over and validating output')
    temporary = job.directory / 'export.partial.mov'
    # MOV supports uncompressed audio. Copy if supported, otherwise lossless PCM conversion.
    # Never mux the mono 16 kHz analysis audio.
    base = [ffmpeg(), '-y', '-v', 'error', '-i', silent, '-i', request['audio'], '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy']
    mode = 'stream-copy'
    try: job.run(base + ['-c:a', 'copy', '-movflags', '+faststart', temporary])
    except RuntimeError:
        mode = 'lossless PCM conversion'
        job.run(base + ['-c:a', 'pcm_f32le', '-movflags', '+faststart', temporary])
    # Full decoding catches corrupt/truncated output before publishing it.
    job.run([ffmpeg(), '-v', 'error', '-i', temporary, '-f', 'null', '-'])
    verification = cv2.VideoCapture(str(temporary))
    frame_count = int(verification.get(cv2.CAP_PROP_FRAME_COUNT)); verification.release()
    if frame_count != total: raise RuntimeError(f'Expected {total} frames but export contains {frame_count}')
    output = job.directory / 'export.mov'
    temporary.replace(output)
    evidence = dict(model='MuseTalk v1.5', device='mps', torch=torch.__version__, frames=produced, duration=info.duration, elapsed=round(time.monotonic()-job.started, 2), audio_mode=mode, source_audio_sha256=hashlib.sha256(Path(request['audio']).read_bytes()).hexdigest(), motion=bool(motion_video), source_mode='real_video' if direct_video else 'photo', visual_source_kind=request.get('visual_source_kind', 'real camera footage' if direct_video else 'photo or motion-transferred photo'), quality_review='Required; inference success does not establish perceptual quality')
    atomic_json(job.directory / 'provenance.json', evidence)
    (ROOT / '.local').mkdir(exist_ok=True)
    atomic_json(ROOT / '.local/last-success.json', evidence)
    job.status(1, f'Completed • {produced} generated frames • audio {mode}', 'Completed', output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['doctor', 'render'])
    parser.add_argument('request', nargs='?')
    args = parser.parse_args()
    if args.command == 'doctor': return doctor()
    if not args.request: parser.error('render requires a request JSON path')
    request = json.loads(Path(args.request).read_text())
    job = Job(request['directory'], attempt=request.get('attempt'))
    def cancel(_signum, _frame):
        job.cancelled = True
        for child in job.children:
            if child.poll() is None: child.terminate()
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        with GPUInferenceLease(job.check,
                            lambda: job.status(0, 'Waiting for the active local render to release memory')):
            if request.get('engine') == 'ditto':
                from ditto_render import render_ditto
                render_ditto(request, job)
            else:
                render(request, job)
        return 0
    except Cancelled:
        job.status(0, 'Cancelled; partial files are not exported', 'Cancelled'); return 130
    except Exception as error:
        traceback.print_exc()
        if job.cancelled: job.status(0, 'Cancelled', 'Cancelled'); return 130
        job.status(0, str(error), 'Failed'); return 1

if __name__ == '__main__':
    sys.exit(main())
