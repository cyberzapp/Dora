"""Sequential real Gemma/connector loading for the local LTX preview."""
import mlx.core as mx
from ltx_pipelines_mlx.utils import blocks
from ltx_core_mlx.model.transformer.model import LTXModelConfig


class StagedPromptEncoder(blocks.PromptEncoder):
    def __init__(self,model_dir,gemma_model_id,progress=lambda *_:None):
        super().__init__(model_dir,gemma_model_id)
        self.progress=progress

    def load(self):
        # The pipeline calls load() before encode(). Parent load retains both
        # large components and would defeat the sequential execution below.
        return None

    def encode(self,prompt):
        self.progress('Encoding prompt with local Gemma weights')
        gemma=blocks.GemmaLanguageModel()
        gemma.load(self.gemma_model_id)
        hidden,mask=gemma.encode_all_layers(prompt)
        mx.eval(*hidden,mask)
        del gemma
        blocks.aggressive_cleanup()
        self.progress('Projecting prompt features with the local connector')
        config=LTXModelConfig.from_checkpoint_dir(self.model_dir)
        connector=blocks.GemmaFeaturesExtractorV2(double_precision_rope=config.double_precision_rope)
        weights=blocks.load_split_safetensors(self.model_dir/'connector.safetensors',prefix='connector.')
        connector.connector.load_weights(list(weights.items()),strict=True)
        video,audio=connector(hidden,attention_mask=mask)
        mx.eval(video,audio)
        del hidden,mask,connector,weights
        blocks.aggressive_cleanup()
        return video,audio
