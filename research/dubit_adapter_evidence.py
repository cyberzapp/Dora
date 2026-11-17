"""Fail closed if the pinned MLX stream silently omits the requested adapter.

This inspects runtime binding evidence after an actual model step. It does not
establish numeric equivalence to PyTorch or any perceptual quality.
"""
from pathlib import Path


def streamed_adapter_evidence(model, expected_path):
    try:
        sources = object.__getattribute__(model, '_lora_sources')
        streamer = object.__getattribute__(model, '_streamer')
        layout = object.__getattribute__(model, '_lora_layout')
        attached = object.__getattribute__(model, '_lora_adapters')
    except AttributeError as error:
        raise ValueError('The expected streamed transformer adapter binding is unavailable') from error
    if len(sources) != 1:
        raise ValueError('Expected exactly one active Dub-It adapter source')
    source = sources[0]
    if (Path(source._lora_path).resolve() != Path(expected_path).resolve() or
            source.fuse or source.strength != 1.0):
        raise ValueError('The active adapter differs from the requested unfused Dub-It weights')
    matched = [index for index in range(streamer.block_count) if source.has_block(index)]
    ranks = source.ranks()
    if not matched or not ranks or any(rank <= 0 for rank in ranks.values()):
        raise ValueError('Dub-It weights matched no usable streamed transformer blocks')
    if layout != (id(source),) or attached is None or len(attached) == 0:
        raise ValueError('Dub-It source exists but runtime adapters were not bound during the model step')
    if attached.skipped:
        raise ValueError('Dub-It targets unsupported transformer layers: '+', '.join(attached.skipped))
    return dict(matched_blocks=matched, targeted_layers=dict(ranks),
                attached_layer_count=len(attached), mode='unfused', strength=source.strength)
