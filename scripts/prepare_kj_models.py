"""Convert already-downloaded upstream weights to KJ's safetensors layout."""
from pathlib import Path
import shutil
import torch
from safetensors.torch import save_file
root=Path(__file__).resolve().parents[1]
source=root/'vendor/LivePortrait/pretrained_weights/liveportrait'
target=root/'vendor/ComfyUI/models/liveportrait';target.mkdir(parents=True,exist_ok=True)
for path in (source/'base_models').glob('*.pth'):
    weights=torch.load(path,map_location='cpu',weights_only=True)
    save_file({k:v.contiguous() for k,v in weights.items()},str(target/(path.stem+'.safetensors')))
weights=torch.load(source/'retargeting_models/stitching_retargeting_module.pth',map_location='cpu',weights_only=True)
combined={}
for prefix in ['retarget_shoulder','retarget_mouth','retarget_eye']:
    for key,value in weights[prefix].items():combined[prefix+'_module.'+key.removeprefix('module.')]=value.contiguous()
save_file(combined,str(target/'stitching_retargeting_module.safetensors'))
shutil.copy2(source/'landmark.onnx',target/'landmark.onnx')
print(target)
