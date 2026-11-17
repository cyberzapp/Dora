"""Reference-conditioned attention correction outside the pinned MLX checkout.

Matches Meituan's avatar/attention.py continuation branch even when reference
masking is disabled. The pinned port merges portrait/context queries in that
case; the reference keeps their attention histories separate. Weights and the
one-frame image-to-video path are unchanged. Numerical tests are not quality QA.
"""
import mlx.core as mx
from longcat_video_avatar.models.avatar.attention import Attention


class ContinuationAttention(Attention):
    def __call__(self, x, shape, num_cond_latents=None, return_kv=False,
                 num_ref_latents=None, ref_img_index=None, mask_frame_range=None,
                 ref_target_masks=None):
        if num_cond_latents is None or num_cond_latents <= 1 or (mask_frame_range is not None and mask_frame_range > 0):
            return super().__call__(x, shape, num_cond_latents, return_kv,
                                    num_ref_latents, ref_img_index, mask_frame_range,
                                    ref_target_masks)
        if num_ref_latents != 1 or ref_img_index is None:
            raise ValueError('Continuation requires one explicit portrait anchor')
        if ref_target_masks is not None:
            raise NotImplementedError('Multi-person routing is not supported')
        batch, tokens, channels = x.shape
        temporal = shape[0]
        if tokens % temporal or not 1 < num_cond_latents <= temporal:
            raise ValueError('Invalid continuation attention geometry')
        per_frame = tokens // temporal
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]
        cache = (k, v)
        q, k = self.rope_3d(q, k, shape, frame_index=ref_img_index, num_ref_latents=num_ref_latents)
        end = num_cond_latents * per_frame
        parts = [self._process_attn(q[:, :, :per_frame], k[:, :, :per_frame], v[:, :, :per_frame]),
                 self._process_attn(q[:, :, per_frame:end], k[:, :, per_frame:end], v[:, :, per_frame:end])]
        if num_cond_latents < temporal:
            parts.append(self._process_attn(q[:, :, end:], k, v))
        result = mx.concatenate(parts, axis=2).transpose(0, 2, 1, 3).reshape(batch, tokens, channels)
        result = self.proj(result)
        return (result, cache, None) if return_kv else (result, None)
