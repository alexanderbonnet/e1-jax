import jax
import jax.numpy as jnp

from e1jax import _model, _tokenizer


def test_padding_masking() -> None:
    pad_len = 17
    seq = "ACDEFGHIKLMNPQRSTVWY"
    tokenized = _tokenizer.tokenize(seq)
    padded = _tokenizer.pad_and_mask(tokenized, pad_length=pad_len)

    model = _model.E1(dim=128, num_layers=6, num_heads=4, ff_dim=256, key=jax.random.PRNGKey(0))
    logits, embeddings = model(**tokenized._asdict())
    logits_pad, embeddings_pad = model(**padded._asdict())

    assert jnp.allclose(logits, logits_pad[:-pad_len, :], atol=1e-5)
    assert jnp.allclose(embeddings, embeddings_pad[:-pad_len, :], atol=1e-5)
