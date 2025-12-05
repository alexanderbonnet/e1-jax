from typing import Final, Literal

import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, jaxtyped

from e1jax import tokenizer

GLOBAL_ATTENTION_EVERY_N_LAYERS = 3
ROPE_THETA_WITHIN_SEQ = 10_000
ROPE_THETA_GLOBAL = 500_000

MODEL_HYPERPARAMS: Final[dict[str, dict[str, int]]] = {
    "E1-150m": {"dim": 768, "num_heads": 12, "ff_dim": 2304, "num_layers": 20},
    "E1-300m": {"dim": 1024, "num_heads": 16, "ff_dim": 3072, "num_layers": 20},
    "E1-600m": {"dim": 1280, "num_heads": 20, "ff_dim": 3840, "num_layers": 30},
}


def max_neg_value(array: Float[Array, "..."]) -> Float[Array, "..."]:
    return jnp.finfo(array.dtype).min


@jaxtyped(typechecker=beartype)
def gelu(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    """Matches the default pytorch implementation."""
    return jax.nn.gelu(x, approximate=False)


@jaxtyped(typechecker=beartype)
def fixed_pos_embedding(
    n: int, dim: int, base: int = 10_000
) -> tuple[Float[Array, " n dim"], Float[Array, " n dim"]]:
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
    frequencies = jnp.einsum("i,j->ij", jnp.arange(n), inv_freq)
    emb = jnp.concatenate([frequencies, frequencies], axis=-1)
    return jnp.sin(emb), jnp.cos(emb)


@jaxtyped(typechecker=beartype)
def rotate_half(x: Float[Array, "head seq dim"]) -> Float[Array, "head seq dim"]:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concat((-x2, x1), axis=-1)


@jaxtyped(typechecker=beartype)
def apply_rotary_pos_emb(
    x: Float[Array, "head seq dim"], sin: Float[Array, "seq dim"], cos: Float[Array, "seq dim"]
) -> Float[Array, "head seq dim"]:
    return (x * cos[None, :, :]) + (rotate_half(x) * sin[None, :, :])


@jaxtyped(typechecker=beartype)
class FeedForward(eqx.Module):
    norm: nn.RMSNorm
    linear_a: nn.Linear
    linear_b: nn.Linear
    linear_out: nn.Linear

    def __init__(self, dim: int, intermediate_dim: int, *, key=PRNGKeyArray) -> None:
        key1, key2, key3 = jax.random.split(key, 3)

        self.linear_a = nn.Linear(dim, intermediate_dim, key=key1)
        self.linear_b = nn.Linear(dim, intermediate_dim, key=key2)
        self.linear_out = nn.Linear(intermediate_dim, dim, key=key3)
        self.norm = nn.RMSNorm(dim)

    def __call__(self, x: Float[Array, " ... dim"]) -> Float[Array, " ... dim"]:
        x = jax.vmap(self.norm)(x)
        x = jax.nn.silu(jax.vmap(self.linear_a)(x)) * jax.vmap(self.linear_b)(x)
        return jax.vmap(self.linear_out)(x)


def create_intra_sequence_mask(sequence_ids: Int[Array, " n"]) -> Bool[Array, "n n"]:
    return sequence_ids[:, None] == sequence_ids[None, :]


def create_block_causal_mask(sequence_ids: Int[Array, " n"]) -> Bool[Array, "n n"]:
    n = sequence_ids.shape[0]
    blocks = create_intra_sequence_mask(sequence_ids)
    causal = jnp.tril(jnp.ones((n, n), dtype=bool))
    return blocks + causal


@jaxtyped(typechecker=beartype)
class MultiHeadAttention(eqx.Module):
    clip_qkv: float
    norm: nn.RMSNorm
    to_q: nn.Linear
    to_k: nn.Linear
    to_v: nn.Linear
    to_out: nn.Linear
    head_dim: int
    rope_theta: int
    layer_type: str

    def __init__(
        self,
        dim: int,
        num_heads: int,
        layer_type: Literal["within_seq", "global"],
        *,
        key: PRNGKeyArray,
    ) -> None:
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.norm = nn.RMSNorm(dim, use_bias=False)

        self.to_q = nn.Linear(dim, dim, key=key1)
        self.to_k = nn.Linear(dim, dim, key=key2)
        self.to_v = nn.Linear(dim, dim, key=key3)
        self.to_out = nn.Linear(dim, dim, key=key4)

        self.clip_qkv = 8.0
        self.head_dim = dim // num_heads
        self.rope_theta = ROPE_THETA_WITHIN_SEQ if layer_type == "within_seq" else ROPE_THETA_GLOBAL
        self.layer_type = layer_type

    def __call__(
        self,
        emb: Float[Array, " n dim"],
        sequence_indexes: Int[Array, " n"],
        global_indexes: Int[Array, " n"],
        sequence_ids: Int[Array, " n"],
        mask_pad: Bool[Array, " n"],
    ) -> Float[Array, " n dim"]:
        emb = jax.vmap(self.norm)(emb)

        query, key, value = map(lambda x: jax.vmap(x)(emb), (self.to_q, self.to_k, self.to_v))

        def clip_and_reshape(x: Float[Array, " n dim"]) -> Float[Array, " h n dim"]:
            x = jnp.clip(x, -self.clip_qkv, self.clip_qkv)
            return einops.rearrange(x, "n (h d) -> h n d", d=self.head_dim)

        query, key, value = map(lambda x: clip_and_reshape(x), (query, key, value))

        sin, cos = fixed_pos_embedding(n=emb.shape[0], dim=self.head_dim, base=self.rope_theta)

        attention_mask_pad = jnp.einsum("i,j->ij", mask_pad, mask_pad)
        if self.layer_type == "within_seq":
            attention_mask = create_intra_sequence_mask(sequence_ids)
            sin, cos = map(lambda x: x[sequence_indexes, :], (sin, cos))
        else:
            attention_mask = create_block_causal_mask(sequence_ids)
            sin, cos = map(lambda x: x[global_indexes, :], (sin, cos))

        attention_mask = attention_mask * attention_mask_pad
        query, key = map(lambda x: apply_rotary_pos_emb(x, sin, cos), (query, key))

        attention = jnp.einsum("hik,hjk->hij", query, key) / jnp.sqrt(self.head_dim)
        attention = jnp.where(attention_mask[None, :, :], attention, max_neg_value(attention))
        attention = jax.nn.softmax(attention, axis=-1)

        out = jnp.einsum("hik,hkj->hij", attention, value)
        out = einops.rearrange(out, "h n d -> n (h d)")
        return jax.vmap(self.to_out)(out)


@jaxtyped(typechecker=beartype)
class TransformerLayer(eqx.Module):
    attn: MultiHeadAttention
    ff: FeedForward

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        layer_type: Literal["within_seq", "global"],
        *,
        key: PRNGKeyArray,
    ) -> None:
        key1, key2 = jax.random.split(key, 2)
        self.attn = MultiHeadAttention(dim, num_heads, layer_type, key=key1)
        self.ff = FeedForward(dim, ff_dim, key=key2)

    def __call__(
        self,
        emb: Float[Array, " n dim"],
        sequence_indexes: Int[Array, " n"],
        global_indexes: Int[Array, " n"],
        sequence_ids: Int[Array, " n"],
        mask_pad: Bool[Array, " n"],
    ) -> Float[Array, " n dim"]:
        emb = emb + self.attn(emb, sequence_indexes, global_indexes, sequence_ids, mask_pad)
        emb = emb + self.ff(emb)
        return emb


class MaskedLMHead(eqx.Module):
    linear_in: nn.Linear
    norm: nn.RMSNorm
    linear_out: nn.Linear

    def __init__(self, dim: int, *, key: PRNGKeyArray) -> None:
        key1, key3 = jax.random.split(key, 2)
        self.linear_in = nn.Linear(dim, dim, key=key1)
        self.norm = nn.LayerNorm(dim)
        self.linear_out = nn.Linear(dim, len(tokenizer.TOKENS), key=key3)

    def __call__(self, x: Float[Array, " ... dim"]) -> Float[Array, " ... vocab_size"]:
        x = jax.vmap(self.linear_in)(x)
        x = gelu(x)
        x = jax.vmap(self.norm)(x)
        return jax.vmap(self.linear_out)(x)


@jaxtyped(typechecker=beartype)
class E1(eqx.Module):
    token_embeb: nn.Embedding
    sequence_embed: nn.Embedding
    layers: list[TransformerLayer]
    norm: nn.RMSNorm
    mlm_head: MaskedLMHead

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.token_embeb = nn.Embedding(len(tokenizer.TOKENS), dim, key=key1)
        self.sequence_embed = nn.Embedding(1024, dim, key=key2)
        self.layers = [
            TransformerLayer(dim, num_heads, ff_dim, "global", key=key)
            if (i + 1) % GLOBAL_ATTENTION_EVERY_N_LAYERS == 0
            else TransformerLayer(dim, num_heads, ff_dim, "within_seq", key=key)
            for i, key in enumerate(jax.random.split(key3, num_layers))
        ]
        self.norm = nn.RMSNorm(dim, use_bias=False)
        self.mlm_head = MaskedLMHead(dim, key=key4)

    def __call__(
        self,
        tokens: Int[Array, " n"],
        sequence_indexes: Int[Array, " n"],
        global_indexes: Int[Array, " n"],
        sequence_ids: Int[Array, " n"],
        mask_pad: Bool[Array, " n"],
    ) -> tuple[Float[Array, " n 34"], Float[Array, " n dim"]]:
        emb = jax.vmap(self.token_embeb)(tokens) + jax.vmap(self.sequence_embed)(sequence_indexes)
        for layer in self.layers:
            emb = layer(emb, sequence_indexes, global_indexes, sequence_ids, mask_pad)
        emb = jax.vmap(self.norm)(emb)
        return self.mlm_head(emb), emb
