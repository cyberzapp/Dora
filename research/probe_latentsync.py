"""Architecture compatibility test only; random weights do not generate a usable video."""
import json,os,sys,time
from pathlib import Path
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
os.environ['MPLCONFIGDIR']=str(Path('.local/matplotlib').resolve())
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'vendor/LatentSync'))
import torch
sys.path.insert(0,str(root/'backend'))
from mps_attention import bounded_attention
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--spatial',type=int,default=32);parser.add_argument('--frames',type=int,default=16);args=parser.parse_args()
torch.mps.set_per_process_memory_fraction(.7)
from omegaconf import OmegaConf
from latentsync.models.unet import UNet3DConditionModel
config=OmegaConf.to_container(OmegaConf.load(root/'vendor/LatentSync/configs/unet/stage2.yaml').model)
start=time.monotonic()
model=UNet3DConditionModel(**config).eval().half().to('mps')
with torch.inference_mode(), bounded_attention():
    sample=torch.randn(1,13,args.frames,args.spatial,args.spatial,device='mps',dtype=torch.float16)
    conditioning=torch.randn(args.frames,50,384,device='mps',dtype=torch.float16)
    output=model(sample,torch.tensor(999,device='mps'),encoder_hidden_states=conditioning).sample
    torch.mps.synchronize()
    result={'test':'full architecture single denoising step, random weights', 'latent_spatial':args.spatial, 'frames':args.frames,'shape':list(output.shape),'finite':bool(torch.isfinite(output).all()),'seconds':time.monotonic()-start,'allocated_gb':torch.mps.current_allocated_memory()/1e9,'driver_gb':torch.mps.driver_allocated_memory()/1e9,'quality_verified':False}
    print(json.dumps(result,indent=2))
    (root/f'.local/latentsync-probe-{args.spatial}.json').write_text(json.dumps(result,indent=2))
