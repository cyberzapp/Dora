"""Bounded pretrained FlashHead Pro execution, extracted from the tested recipe.

An experimental engine component, not a claim of accepted visual quality. One
window is resident at a time; the caller owns media validation and transactions.
"""
import gc
import importlib.util
import io
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors import safe_open
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
sys.path.insert(0,str(ROOT/'vendor/SoulX-FlashHead'))
from flashhead_mps import load_model_module
from flash_head.utils.utils import match_and_blend_colors_torch


class Engine:
    def __init__(self,crop,folder,seed=42,recovery=None,check=lambda:None,progress=lambda *args,**kwargs:None):
        if not torch.backends.mps.is_available():raise RuntimeError('Metal is unavailable for FlashHead')
        self.check=check;self.on_progress=progress;self.dtype=torch.bfloat16;self.device='mps'
        self.max_observed_driver_bytes=0;self.max_observed_tensor_bytes=0
        torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.65)
        self.generator=torch.Generator(device='cpu').manual_seed(seed)
        self.original=torch.from_numpy(np.array(crop.resize((512,512),Image.Resampling.LANCZOS))).permute(2,0,1)[None,:,None].to(device=self.device,dtype=self.dtype)/127.5-1
        spec=importlib.util.spec_from_file_location('avatar_engine_vae',ROOT/'vendor/SoulX-FlashHead/flash_head/wan/modules/vae.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.check();self.progress('Preparing portrait')
        self.vae=module.WanVAE(vae_path=str(Path(folder)/'VAE_Wan/Wan2.1_VAE.pth'),
            dtype=self.dtype,device=self.device,parallel=False,use_tiling=False)
        with torch.inference_mode():
            if recovery is None:
                self.reference=self.vae.encode(self.original.repeat(1,1,33,1,1))
                self.motion=self.reference[:,:1].clone()
            else:
                state=self.read_state(recovery)
                self.reference=state['reference'].to(self.device);self.motion=state['motion'].to(self.device)
                self.generator.set_state(state['rng'])
        if not torch.isfinite(self.reference).all():raise RuntimeError('Portrait encoding failed')
        self.vae.model.to('cpu');gc.collect();torch.mps.empty_cache();self.check()
        self.progress('Loading local portrait model')
        import json
        architecture=load_model_module();config=json.loads((Path(folder)/'Model_Pro/config.json').read_text())
        with torch.device('meta'):self.model=architecture.WanModelAudioProject.from_config(config)
        state={}
        with safe_open(str(Path(folder)/'Model_Pro/diffusion_pytorch_model.safetensors'),framework='pt',device='cpu') as weights:
            for name in weights.keys():
                self.check();state[name]=weights.get_tensor(name).to(dtype=self.dtype)
        self.model.load_state_dict(state,strict=True,assign=True);del state
        if any(p.is_meta for p in self.model.parameters()):raise RuntimeError('Model loading left unresolved weights')
        self.model.eval().requires_grad_(False)
        # Checks inside the actual block loop bound cancellation latency without
        # changing the neural calculations or replacing generated frames.
        self.hooks=[block.register_forward_pre_hook(lambda *_:self.check()) for block in self.model.blocks]
        self.times=[]
        for value in (1000,750,500,250,0):
            t=value/1000
            self.times.append(torch.tensor([1000*5*t/(1+4*t)],device=self.device,dtype=torch.float32))

    def observations(self):
        self.max_observed_driver_bytes=max(self.max_observed_driver_bytes,torch.mps.driver_allocated_memory())
        self.max_observed_tensor_bytes=max(self.max_observed_tensor_bytes,torch.mps.current_allocated_memory())
        return dict(max_observed_driver_bytes=self.max_observed_driver_bytes,
                    max_observed_tensor_bytes=self.max_observed_tensor_bytes,
                    interpretation='Observed at stage/step callbacks, not a hard cap, continuous peak or total system memory.')

    def progress(self,stage,**details):
        self.observations();self.on_progress(stage,**details)

    @staticmethod
    def read_state(data):
        state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
        if set(state)!= {'reference','motion','rng'}:raise ValueError('Invalid recovery-state fields')
        for name,shape in [('reference',(16,9,64,64)),('motion',(16,2,64,64))]:
            value=state[name]
            if not isinstance(value,torch.Tensor) or value.dtype!=torch.bfloat16 or tuple(value.shape)!=shape or not torch.isfinite(value).all():
                raise ValueError('Invalid saved '+name)
        if state['rng'].dtype!=torch.uint8 or state['rng'].ndim!=1:raise ValueError('Invalid saved random state')
        return state

    def state(self):
        stream=io.BytesIO()
        torch.save(dict(reference=self.reference.detach().cpu(),motion=self.motion.detach().cpu(),rng=self.generator.get_state()),stream)
        return stream.getvalue()

    def noise(self,shape):
        return torch.randn(shape,generator=self.generator,dtype=self.dtype,device='cpu').to(self.device)

    @torch.inference_mode()
    def generate(self,context):
        self.check();self.vae.model.to('cpu');gc.collect();torch.mps.empty_cache()
        self.model.to(self.device);context=context.to(self.device)
        latents=self.noise((16,9,64,64))
        for step in range(4):
            self.check();latents[:,:self.motion.shape[1]]=self.motion
            flow=self.model(x=latents[None],timestep=self.times[step],context=context,y=self.reference[None])[0]
            current=(self.times[step]/1000).to(self.dtype)[:,None,None,None]
            following=(self.times[step+1]/1000).to(self.dtype)[:,None,None,None]
            clean=latents-flow*current
            latents=(1-following)*clean+following*self.noise(clean.shape)
            if not torch.isfinite(latents).all():raise RuntimeError('Non-finite generated motion')
            torch.mps.synchronize();self.progress('Generating speech',step=step+1)
        latents[:,:self.motion.shape[1]]=self.motion
        del flow,clean,context;self.model.to('cpu');gc.collect();torch.mps.empty_cache()
        self.vae.model.to(self.device);chunks=[];count=0
        for chunk in self.vae.decode_stream(latents):
            self.check()
            if not torch.isfinite(chunk).all():raise RuntimeError('Non-finite generated pixels')
            chunks.append(chunk.cpu());count+=chunk.shape[2];self.progress('Decoding video',frames=count)
        if count!=33:raise RuntimeError('Incomplete temporal decoding')
        del chunk;torch.mps.empty_cache();videos=torch.cat(chunks,dim=2).to(self.device);del chunks
        videos=match_and_blend_colors_torch(videos,self.original,1.0)
        if not torch.isfinite(videos).all():raise RuntimeError('Color processing failed')
        # Always save actual motion carry, even at a final window, so committed
        # recovery state has one stable shape and no implicit re-encoding later.
        self.motion=self.vae.encode(videos[:,:,-5:].to(self.dtype))
        if not torch.isfinite(self.motion).all():raise RuntimeError('Motion carry encoding failed')
        frames=((videos[0,:,5:].float().permute(1,2,3,0)+1)*127.5).clamp(0,255).byte().cpu().numpy()
        del videos,latents;gc.collect();torch.mps.empty_cache();self.check()
        return frames

    def close(self):
        for hook in getattr(self,'hooks',[]):hook.remove()
        for name in ('model','vae','original','reference','motion','times'):
            if hasattr(self,name):delattr(self,name)
        gc.collect();torch.mps.empty_cache()
