"""Offline runtime diagnostics. Completed downloads do not imply video quality."""
import json,sys,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 manifest=json.loads((ROOT/'backend/model_manifests/longcat-q4.json').read_text())
 folder=ROOT/'.local/longcat-models/q4'
 missing=[x['path'] for x in manifest['files'] if not (folder/x['path']).is_file() or (folder/x['path']).stat().st_size!=x['bytes']]
 report=dict(python=sys.version.split()[0],macos=platform.mac_ver()[0],required_macos_major=26,missing_files=missing,download_complete=(folder/'download-complete.json').exists(),model_inference_verified=False,quality_approved=False)
 try:
  import mlx.core as mx
  import imageio_ffmpeg
  x=mx.ones((8,8));report['metal_tensor_test']=float((x@x)[0,0])==8
  report['device']=mx.device_info();report['ffmpeg_available']=Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file()
  evidence=ROOT/'.local/longcat-first-render/report.json'
  if evidence.exists():
   value=json.loads(evidence.read_text())
   report['model_inference_verified']=bool(value.get('generated_video') and value.get('full_frame_decode_passed') and value.get('decoded_audio_sha256') and value.get('request',{}).get('model_revision')==manifest['revision'] and Path(value.get('output','')).is_file())
 except Exception as error:report['error']=str(error)
 print(json.dumps(report,indent=2));return int(bool(missing) or 'error' in report)
if __name__=='__main__':sys.exit(main())
