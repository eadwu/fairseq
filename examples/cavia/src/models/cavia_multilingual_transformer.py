from fairseq.models import register_model, register_model_architecture
from fairseq.models.multilingual_transformer import (
    MultilingualTransformerModel, multilingual_transformer_iwslt_de_en
)
from fairseq.models.transformer import TransformerEncoder

from .cavia_transformer import CAVIATransformerDecoder


@register_model("cavia_multilingual_transformer")
class CAVIAMultilingualTransformer(MultilingualTransformerModel):
    """A variant of standard multilingual Transformer models whose decoder
    supports CAVIA (adapted using BatchEnsemble).
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        MultilingualTransformerModel.add_args(parser)

    @classmethod
    def _get_module_class(cls, is_encoder, args, lang_dict, embed_tokens, langs):
        module_class = TransformerEncoder if is_encoder else CAVIATransformerDecoder
        return module_class(args, lang_dict, embed_tokens)


@register_model_architecture(
    "cavia_multilingual_transformer", "cavia_multilingual_transformer"
)
def cavia_multilingual_architecture(args):
    multilingual_transformer_iwslt_de_en(args)

    args.share_encoders = True
    args.share_decoders = True
    args.share_encoder_embeddings = True
    args.share_decoder_embeddings = True


@register_model_architecture(
    "cavia_multilingual_transformer", "cavia_phat_multilingual_transformer"
)
def cavia_phat_multilingual_architecture(args):
    cavia_multilingual_architecture(args)

    # Latent Depth number of layers
    args.encoder_layers = 12
    args.decoder_layers = 24
