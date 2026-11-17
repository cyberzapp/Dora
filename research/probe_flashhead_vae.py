"""Pretrained video-autoencoder reconstruction probe, not a talking avatar."""
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from inference_lease import GPUInferenceLease
from flashhead_benchmark import verify_models,digest


def main():
    import torch
    import numpy as np
    from PIL import Image,ImageOps,ImageDraw
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    verify_models(ROOT/'.local/flashhead-vae-download')
    output=ROOT/'.local/flashhead-vae-probe'
    output.mkdir(exist_ok=False)
    source=Path('/path/to/your/local/media/Portrait_BlackT.png')
    crop=ImageOps.exif_transpose(Image.open(source)).convert('RGB').crop((353,1,1060,708)).resize((512,512),Image.Resampling.LANCZOS)
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.65)
    spec=importlib.util.spec_from_file_location('probe_flashhead_vae',ROOT/'vendor/SoulX-FlashHead/flash_head/wan/modules/vae.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    start=time.monotonic();peak=0;decoded=[]
    with torch.inference_mode():
        vae=module.WanVAE(vae_path=str(ROOT/'.local/flashhead-vae-download/VAE_Wan/Wan2.1_VAE.pth'),dtype=torch.bfloat16,device='mps',parallel=False)
        values=torch.from_numpy(np.array(crop)).permute(2,0,1)[None,:,None].to(device='mps',dtype=torch.bfloat16)/127.5-1
        latent=vae.encode(values.repeat(1,1,33,1,1))
        if not torch.isfinite(latent).all():raise RuntimeError('Non-finite appearance encoding')
        peak=max(peak,torch.mps.driver_allocated_memory())
        for chunk in vae.decode_stream(latent):
            if not torch.isfinite(chunk).all():raise RuntimeError('Non-finite reconstruction')
            peak=max(peak,torch.mps.driver_allocated_memory())
            decoded.extend(((chunk[0].float().permute(1,2,3,0)+1)*127.5).clamp(0,255).byte().cpu().numpy())
    if len(decoded)!=33:raise RuntimeError('Incomplete reconstruction')
    sheet=Image.new('RGB',(1024,560),(22,23,27));draw=ImageDraw.Draw(sheet)
    draw.text((12,12),'Uploaded photo crop - original',fill='white')
    draw.text((524,12),'Video VAE reconstruction only - NOT animation',fill='white')
    sheet.paste(crop,(0,48));sheet.paste(Image.fromarray(decoded[16]),(512,48));sheet.save(output/'reconstruction.jpg',quality=95)
    report=dict(actual_pretrained_vae=True,talking_video_generated=False,quality_passed=False,
                dtype='bfloat16',device='mps',source_sha256=digest(source),frames=33,
                latent_shape=list(latent.shape),elapsed_seconds=time.monotonic()-start,
                max_observed_driver_bytes=peak,
                mid_frame_mean_absolute_rgb_difference=float(np.abs(decoded[16].astype(np.float32)-np.array(crop).astype(np.float32)).mean()))
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    with GPUInferenceLease(on_wait=lambda:print('Waiting for active local inference',flush=True)):
        main()
