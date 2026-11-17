"""External single-device adapter for pinned official FlashHead code.

No model weights, training or quality claim. Replace unsupported double/complex
rotary arithmetic with equivalent real float32 arithmetic and bound SDPA queries.
The NVIDIA distributed implementation is intentionally unavailable in this path.
"""
import importlib.util
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def _distributed_unavailable(*args, **kwargs):
    raise RuntimeError('This adapter supports one local device; distributed attention is unavailable')


def frequencies(dim, end=1024, theta=10000.0):
    # Compute constants on CPU in the same precision as the official code.
    positions = torch.arange(end, device='cpu', dtype=torch.float64)
    inverse = 1 / theta ** (torch.arange(0, dim, 2, device='cpu', dtype=torch.float64)[:dim//2] / dim)
    phase = torch.outer(positions, inverse)
    return torch.stack((phase.cos(), phase.sin()), dim=-1).float()


def frequencies_3d(dim, end=1024, theta=10000.0):
    return torch.cat([frequencies(dim-2*(dim//3), end, theta),
                      frequencies(dim//3, end, theta),
                      frequencies(dim//3, end, theta)], dim=1)


def sinusoidal_embedding(dim, position):
    # Transfer first: a combined device/dtype conversion can try to create an
    # unsupported Double tensor on MPS before the copy reaches the CPU.
    values = position.detach().cpu().to(dtype=torch.float64)
    phase = torch.outer(values, 10000 ** (-torch.arange(dim//2, device='cpu', dtype=torch.float64)/(dim//2)))
    return torch.cat((phase.cos(), phase.sin()), dim=1).to(device=position.device, dtype=position.dtype)


def rotary(x, freqs, grid_sizes, use_usp=False, sp_size=1, sp_rank=0):
    if use_usp or sp_size != 1 or sp_rank != 0:
        return _distributed_unavailable()
    if x.shape[0] != 1:
        raise ValueError('Pinned FlashHead rotary path expects one sample')
    f, h, w = map(int, grid_sizes)
    if f*h*w != x.shape[1]:
        raise ValueError('Rotary grid does not match latent tokens')
    half = x.shape[-1]//2
    time, vertical, horizontal = freqs.split([half-2*(half//3),half//3,half//3],dim=1)
    if max(f,h,w) > len(freqs):
        raise ValueError('Rotary grid exceeds the precomputed context')
    phase = torch.cat((time[:f].view(f,1,1,-1,2).expand(f,h,w,-1,2),
                       vertical[:h].view(1,h,1,-1,2).expand(f,h,w,-1,2),
                       horizontal[:w].view(1,1,w,-1,2).expand(f,h,w,-1,2)),dim=-2)
    phase = phase.reshape(1,f*h*w,1,half,2).to(device=x.device,dtype=torch.float32)
    pairs = x.float().reshape(*x.shape[:-1],half,2)
    real, imag = pairs.unbind(-1); cosine, sine = phase.unbind(-1)
    return torch.stack((real*cosine-imag*sine,real*sine+imag*cosine),dim=-1).flatten(-2).to(x.dtype)


def attention(q,k,v,num_heads,compatibility_mode=False,query_chunk=256):
    def split(value):
        return value.reshape(value.shape[0],value.shape[1],num_heads,-1).transpose(1,2)
    q,k,v=map(split,(q,k,v))
    parts=[F.scaled_dot_product_attention(q[:,:,start:start+query_chunk],k,v,dropout_p=0.0)
           for start in range(0,q.shape[2],query_chunk)]
    result=torch.cat(parts,dim=2).transpose(1,2)
    return result.reshape(result.shape[0],result.shape[1],-1)


def load_audio_encoder(folder):
    """Strictly load old weight-normalization names into current PyTorch.

    HF's older automatic loader silently leaves the positional convolution random
    with this Torch version. Map its trained g/v tensors explicitly. SpecAugment
    is training-only and disabled here, so its absent mask token is not invented.
    """
    from safetensors.torch import load_file
    from transformers import Wav2Vec2Config
    from flash_head.audio_analysis.wav2vec2 import Wav2Vec2Model
    config=Wav2Vec2Config.from_pretrained(str(folder),local_files_only=True)
    config.mask_time_prob=0.0;config.mask_feature_prob=0.0;config.apply_spec_augment=False
    with torch.device('meta'):model=Wav2Vec2Model(config)
    state={};unused=[]
    for name,tensor in load_file(str(Path(folder)/'model.safetensors'),device='cpu').items():
        if name in ('lm_head.bias','lm_head.weight'):
            unused.append(name);continue
        if not name.startswith('wav2vec2.'):
            raise ValueError('Unexpected speech-encoder weight: '+name)
        name=name.removeprefix('wav2vec2.')
        if name.endswith('pos_conv_embed.conv.weight_g'):
            name=name.removesuffix('weight_g')+'parametrizations.weight.original0'
        elif name.endswith('pos_conv_embed.conv.weight_v'):
            name=name.removesuffix('weight_v')+'parametrizations.weight.original1'
        state[name]=tensor
    model.load_state_dict(state,strict=True,assign=True)
    if any(p.is_meta for p in model.parameters()):raise RuntimeError('Speech encoder has unresolved weights')
    model.eval().requires_grad_(False)
    model.avatar_loading_report=dict(strict=True,unused_ctc_head=unused,
        positional_convolution='trained weight_g/v mapped to PyTorch parametrizations',
        spec_augment='disabled inference-only training augmentation; absent mask token not used')
    return model


def load_model_module():
    if torch.distributed.is_initialized():
        return _distributed_unavailable()
    # Resolve CUDA-only imports only while loading this private module. No global
    # fake backend is left installed, and using a distributed function fails.
    distributed=types.ModuleType('xfuser.core.distributed')
    for name in ('get_sequence_parallel_rank','get_sequence_parallel_world_size','get_sp_group'):
        setattr(distributed,name,_distributed_unavailable)
    long_context=types.ModuleType('xfuser.core.long_ctx_attention')
    long_context.xFuserLongContextAttention=_distributed_unavailable
    temporary={'xfuser':types.ModuleType('xfuser'),'xfuser.core':types.ModuleType('xfuser.core'),
               distributed.__name__:distributed,long_context.__name__:long_context}
    path=ROOT/'vendor/SoulX-FlashHead/flash_head/src/modules/flash_head_model.py'
    spec=importlib.util.spec_from_file_location('avatar_flashhead_model',path)
    module=importlib.util.module_from_spec(spec)
    previous={name:sys.modules.get(name) for name in temporary}
    try:
        sys.modules.update(temporary)
        spec.loader.exec_module(module)
    finally:
        # Restore only our temporary names. Restoring the entire module dictionary
        # would unload unrelated Torch imports and cause duplicate registrations.
        for name,original in previous.items():
            if original is None:sys.modules.pop(name,None)
            else:sys.modules[name]=original
    module.reference_rotary=module.rope_apply
    module.reference_frequencies=module.precompute_freqs_cis_3d
    module.reference_sinusoidal=module.sinusoidal_embedding_1d
    module.precompute_freqs_cis_3d=frequencies_3d
    module.rope_apply=rotary
    module.sinusoidal_embedding_1d=sinusoidal_embedding
    module.flash_attention=attention
    return module
