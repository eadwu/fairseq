from fairseq.models import register_model, register_model_architecture
from fairseq.models.multilingual_transformer import MultilingualTransformerModel
from fairseq.models.transformer import (
    TransformerEncoder,
    base_architecture,
)

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
    def _get_module_class(cls, is_encoder, args, lang_dict, embed_tokens):
        if is_encoder:
            return TransformerEncoder(args, lang_dict, embed_tokens)
        else:
            return CAVIATransformerDecoder(args, lang_dict, embed_tokens)


@register_model_architecture(
    "cavia_multilingual_transformer", "cavia_multilingual_transformer"
)
def latent_multilingual_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    # multilingual_transformer_iwslt_de_en = 6
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    # multilingual_transformer_iwslt_de_en = 6
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.share_encoders = getattr(args, "share_encoders", True)
    args.share_decoders = getattr(args, "share_decoders", True)
    args.share_encoder_embeddings = getattr(args, "share_encoder_embeddings", True)
    args.share_decoder_embeddings = getattr(args, "share_decoder_embeddings", True)

    base_architecture(args)
