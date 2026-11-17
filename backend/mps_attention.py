"""Bound MPS attention-score memory by splitting independent query rows."""
from contextlib import contextmanager

@contextmanager
def bounded_feed_forward(model, token_chunk=256):
    """Chunk independent tokens, preserving the network's temporal attention.

    FeedForward acts only on the last feature dimension. This is restricted to
    inference: training dropout must not be reordered by this optimization.
    """
    import torch
    from diffusers.models.attention import FeedForward
    replaced=[]
    for module in model.modules():
        if not isinstance(module, FeedForward):
            continue
        original=module.forward
        had_local='forward' in module.__dict__
        def forward(hidden_states, *args, _module=module, _original=original, **kwargs):
            if (_module.training or hidden_states.device.type!='mps'
                    or hidden_states.ndim<3 or hidden_states.shape[-2]<=token_chunk):
                return _original(hidden_states,*args,**kwargs)
            return torch.cat([
                _original(part,*args,**kwargs)
                for part in hidden_states.split(token_chunk,dim=-2)
            ],dim=-2)
        module.forward=forward
        replaced.append((module,original,had_local))
    try:
        yield
    finally:
        for module,original,had_local in replaced:
            if had_local: module.forward=original
            else: del module.forward

@contextmanager
def bounded_attention(query_chunk=256):
    import torch
    import torch.nn.functional as functional
    original=functional.scaled_dot_product_attention
    def attention(query,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,scale=None,enable_gqa=False):
        options=dict(dropout_p=dropout_p,is_causal=is_causal,scale=scale,enable_gqa=enable_gqa)
        if query.device.type!='mps' or query.shape[-2]<=query_chunk or is_causal or dropout_p:
            return original(query,key,value,attn_mask=attn_mask,**options)
        outputs=[]
        for start in range(0,query.shape[-2],query_chunk):
            mask=attn_mask
            if mask is not None and mask.ndim>=2 and mask.shape[-2]==query.shape[-2]:mask=mask[...,start:start+query_chunk,:]
            outputs.append(original(query[...,start:start+query_chunk,:],key,value,attn_mask=mask,**options))
        return torch.cat(outputs,dim=-2)
    functional.scaled_dot_product_attention=attention
    try:yield
    finally:functional.scaled_dot_product_attention=original
