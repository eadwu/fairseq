import torch

from fairseq.models import (
    register_model, register_model_architecture,
    FairseqEncoderDecoderModel,
)
from fairseq.models.multilingual_transformer import (
    MultilingualTransformerModel, multilingual_transformer_iwslt_de_en
)
from fairseq.models.transformer import TransformerEncoder

from .be_transformer import BatchEnsembleTransformerDecoder


class BEFairseqEncoderDecoderModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

        self.ensemble_size = None
        self.avg_ensemble = False

    def with_state(self, ensemble_size, avg_ensemble):
        self.ensemble_size = ensemble_size
        self.avg_ensemble = avg_ensemble

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        if self.ensemble_size is not None:
            src_tokens_dim = [1 for _ in src_tokens.shape]
            src_tokens_dim[0] = self.ensemble_size
            src_tokens = torch.tile(src_tokens, src_tokens_dim)

            src_lengths_dim = [1 for _ in src_lengths.shape]
            src_lengths_dim[0] = self.ensemble_size
            src_lengths = torch.tile(src_lengths, src_lengths_dim)

            prev_output_tokens_dim = [1 for _ in prev_output_tokens.shape]
            prev_output_tokens_dim[0] = self.ensemble_size
            prev_output_tokens = torch.tile(
                prev_output_tokens, prev_output_tokens_dim
            )

        decoder_out = super().forward(
            src_tokens, src_lengths, prev_output_tokens, **kwargs
        )

        if self.avg_ensemble:
            decoder_out = torch.mean(torch.stack(torch.split(
                decoder_out, decoder_out.shape[0] / self.ensemble_size
            )))

        return decoder_out


@register_model("be_multilingual_transformer")
class BatchEnsembleMultilingualTransformer(MultilingualTransformerModel):
    """A variant of standard multilingual Transformer models whose decoder
    supports uses BatchEnsemble.
    """
    def __init__(
        self, encoders, decoders, instance=BEFairseqEncoderDecoderModel
    ):
        super().__init__(encoders, decoders, instance=instance)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        MultilingualTransformerModel.add_args(parser)

    @classmethod
    def _get_module_class(
        cls, is_encoder, args, lang_dict, embed_tokens, langs
    ):
        module_class = (
            TransformerEncoder
            if is_encoder
            else BatchEnsembleTransformerDecoder
        )

        return module_class(args, lang_dict, embed_tokens)


@register_model_architecture(
    "be_multilingual_transformer",
    "batch_ensemble_multilingual_transformer"
)
def batch_ensemble_multilingual_architecture(args):
    multilingual_transformer_iwslt_de_en(args)


@register_model_architecture(
    "be_multilingual_transformer",
    "batch_ensemble_phat_multilingual_transformer"
)
def batch_ensemble_phat_multilingual_architecture(args):
    # Latent Depth number of layers
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 24)

    batch_ensemble_multilingual_architecture(args)
