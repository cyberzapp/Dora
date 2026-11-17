"""Actual KJ node transfer driven by a locally generated LongCat performance."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from worker import Job,Cancelled,atomic_json
from inference_lease import GPUInferenceLease
p=argparse.ArgumentParser();p.add_argument('--photo',required=True);p.add_argument('--driver',required=True);p.add_argument('--audio',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--consent',action='store_true');p.add_argument('--motion-reference');p.add_argument('--pose-strength',type=float,default=.65);p.add_argument('--expression-strength',type=float,default=.65);a=p.parse_args()
if not a.consent:raise ValueError('Likeness consent is required')
job=Job(a.output)
try:
 with GPUInferenceLease(job.check,lambda:job.status(0,'Waiting for the current local render')):
  job.status(.01,'Running actual LivePortraitKJ nodes from the generated performance')
  command=[ROOT/'.venv-comfy/bin/python',ROOT/'backend/comfy_liveportrait.py','--photo',a.photo,'--video',a.driver,'--seconds',str(a.seconds),'--output',a.output]
  if a.motion_reference:command+=['--motion-reference',a.motion_reference,'--pose-strength',str(a.pose_strength),'--expression-strength',str(a.expression_strength)]
  job.run(command)
  video=a.output/'review.mov'
  job.run([ROOT/'.venv/bin/python',ROOT/'scripts/verify_export.py',a.audio,video,'--seconds',str(a.seconds),'--expected-frames',str(round(a.seconds*25)),'--fps','25'])
  atomic_json(a.output/'integration-provenance.json',dict(pipeline='Local LongCat audio-driven performance → actual ComfyUI-LivePortraitKJ source appearance transfer',photo=a.photo,driver=a.driver,audio=a.audio,personal_motion_reference=a.motion_reference,quality_passed=False,visual_review_required=True,personalized_training=False))
  job.status(1,'Actual KJ transfer completed; quality review required','Completed',video)
except Cancelled:job.status(0,'Cancelled','Cancelled')
except Exception as error:job.status(0,str(error),'Failed');raise
