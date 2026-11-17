"""Run the bounded high-resolution quality experiment only after verified download."""
from pathlib import Path
import sys,time,subprocess
root=Path(__file__).resolve().parents[1]
checkpoint=root/'vendor/LatentSync/checkpoints-v16/latentsync_unet.pt'
start=time.monotonic()
while not checkpoint.exists():
    if time.monotonic()-start>7200:raise TimeoutError('Checkpoint unavailable; high-resolution test not run')
    time.sleep(5)
raise SystemExit(subprocess.call([sys.executable,str(root/'research/latentsync_benchmark.py'),'--version','1.6','--video',str(root/'.local/kj-test/review.mov'),'--audio',str(root/'.local/matched-reference/lipsync/audio.wav'),'--seconds','0.64','--steps','20'],cwd=root))
