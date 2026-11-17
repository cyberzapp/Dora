"""Position-encoding operators for a future TBDub Metal compatibility run."""

import torch


def mps_frequency_tables(original, *args, **kwargs):
    """Preserve upstream CPU frequency construction and make tables Metal-compatible."""
    return tuple(table.to(torch.complex64) for table in original(*args, **kwargs))


def mps_timestep_embedding(original, dimension, positions):
    """Keep upstream double-precision timestep evaluation on CPU."""
    if positions.device.type != 'mps':
        return original(dimension, positions)
    return original(dimension, positions.cpu()).to(positions.device)


def mps_rotary_embedding(original, tensor, frequencies, head_count):
    """Compute the same complex rotation using real float32 Metal operations."""
    if tensor.device.type != 'mps':
        return original(tensor, frequencies, head_count)
    batch, tokens, channels = tensor.shape
    pairs = tensor.float().reshape(batch, tokens, head_count, -1, 2)
    real, imaginary = pairs.unbind(dim=-1)
    cosine, sine = torch.view_as_real(frequencies).unbind(dim=-1)
    rotated = torch.stack((real*cosine - imaginary*sine, real*sine + imaginary*cosine), dim=-1)
    return rotated.reshape(batch, tokens, channels).to(tensor.dtype)
