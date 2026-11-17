"""Verify actual local LTX prompt conditioning before the video model is ready."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'research')]
from inference_lease import GPUInferenceLease
from ltx_audio_preview import GEMMA,PROMPT,validate_model


def main():
    folder=ROOT/'.local/ltx-gemma-weights';models=ROOT/'.local/ltx-2.3-q8-weights'
    output=ROOT/'.local/ltx-real-text-probe'
    if output.exists():raise ValueError('Retain the previous text-conditioning probe')
    deadline=time.monotonic()+8*3600
    print('Waiting for verified local Gemma weights; no inference yet',flush=True)
    while not (folder/'download-complete.json').is_file():
        if time.monotonic()>deadline:raise TimeoutError('Text weights did not complete in eight hours')
        time.sleep(20)
    manifest_sha=validate_model(folder,GEMMA)
    from download_model_manifest import matches
    lock=json.loads((ROOT/'research/ltx-source-lock.json').read_text())
    connector_record=next(item for item in lock['models'][0]['files'] if item['path']=='connector.safetensors')
    if not matches(models/connector_record['path'],connector_record['bytes'],connector_record['sha256']):
        raise ValueError('The connector is not verified')
    with GPUInferenceLease(on_wait=lambda:print('Waiting for the active local renderer',flush=True)):
        import mlx.core as mx
        from ltx_pipelines_mlx.utils import blocks
        from ltx_core_mlx.model.transformer.model import LTXModelConfig
        output.mkdir()
        started=time.monotonic();mx.set_default_device(mx.gpu);mx.set_memory_limit(11*1024**3);mx.set_cache_limit(0)
        print('Encoding the actual test prompt with Gemma',flush=True)
        encoder=blocks.GemmaLanguageModel();encoder.load(str(folder))
        hidden,mask=encoder.encode_all_layers(PROMPT)
        mx.eval(*hidden,mask)
        if len(hidden)!=49 or any(x.shape!=(1,1024,3840) for x in hidden):raise ValueError('Unexpected text hidden-state geometry')
        if not all(bool(mx.all(mx.isfinite(x)).item()) for x in hidden):raise ValueError('Invalid Gemma features')
        peak=mx.get_peak_memory()
        del encoder;blocks.aggressive_cleanup()
        print('Projecting real text features using the verified connector',flush=True)
        config=LTXModelConfig.from_checkpoint_dir(models)
        connector=blocks.GemmaFeaturesExtractorV2(double_precision_rope=config.double_precision_rope)
        weights=blocks.load_split_safetensors(models/'connector.safetensors',prefix='connector.')
        connector.connector.load_weights(list(weights.items()),strict=True)
        video,audio=connector(hidden,attention_mask=mask);mx.eval(video,audio)
        if not all(bool(mx.all(mx.isfinite(x)).item()) for x in (video,audio)):raise ValueError('Invalid conditioned embeddings')
        mx.save_safetensors(str(output/'prompt-embeddings.safetensors'),{'video':video,'audio':audio})
        report=dict(actual_gemma_encoder=True,actual_connector=True,prompt=PROMPT,
            gemma_revision=GEMMA[1],manifest_sha256=manifest_sha,connector_sha256=connector_record['sha256'],
            hidden_layers=len(hidden),video_embedding_shape=list(video.shape),audio_embedding_shape=list(audio.shape),
            elapsed_seconds=time.monotonic()-started,max_observed_mlx_peak_bytes=max(peak,mx.get_peak_memory()),
            generated_video=False,quality_passed=False)
        (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    main()
