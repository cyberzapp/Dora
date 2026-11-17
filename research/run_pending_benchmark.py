"""Wait for a verified checkpoint, then run one bounded local quality experiment."""
from pathlib import Path
import subprocess,sys,time
root=Path(__file__).resolve().parents[1]
checkpoint=root/'vendor/LatentSync/checkpoints-v15/latentsync_unet.pt'
start=time.monotonic()
while not checkpoint.exists():
    if time.monotonic()-start>3600:raise TimeoutError('Checkpoint not ready after one hour; benchmark not run')
    time.sleep(5)
command=[sys.executable,str(root/'research/latentsync_benchmark.py'),'--video','/path/to/your/local/media/my_video.mp4','--audio',str(root/'.local/smoke/voice-excerpt.wav'),'--seconds','0.64','--steps','20']
raise SystemExit(subprocess.call(command,cwd=root))
