from pathlib import Path
from typing import Final, Literal

import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import loguru
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, jaxtyped

from e1jax import _constants

GLOBAL_ATTENTION_EVERY_N_LAYERS: Final[int] = 3
ROPE_THETA_WITHIN_SEQ: Final[int] = 10_000
ROPE_THETA_GLOBAL: Final[int] = 500_000
MAX_NUMBER_SEQUENCES: Final[int] = 512

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
    n: int, dim: int, theta: int
) -> tuple[Float[Array, " n dim"], Float[Array, " n dim"]]:
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
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

        self.linear_a = nn.Linear(dim, intermediate_dim, use_bias=False, key=key1)
        self.linear_b = nn.Linear(dim, intermediate_dim, use_bias=False, key=key2)
        self.linear_out = nn.Linear(intermediate_dim, dim, use_bias=False, key=key3)
        self.norm = nn.RMSNorm(dim, use_bias=False)

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

        self.to_q = nn.Linear(dim, dim, use_bias=False, key=key1)
        self.to_k = nn.Linear(dim, dim, use_bias=False, key=key2)
        self.to_v = nn.Linear(dim, dim, use_bias=False, key=key3)
        self.to_out = nn.Linear(dim, dim, use_bias=False, key=key4)

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

        sin, cos = fixed_pos_embedding(n=emb.shape[0], dim=self.head_dim, theta=self.rope_theta)

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
        out = jax.vmap(self.to_out)(out)
        return out


@jaxtyped(typechecker=beartype)
class TransformerLayer(eqx.Module):
    attention: MultiHeadAttention
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
        self.attention = MultiHeadAttention(dim, num_heads, layer_type, key=key1)
        self.ff = FeedForward(dim, ff_dim, key=key2)

    def __call__(
        self,
        emb: Float[Array, " n dim"],
        sequence_indexes: Int[Array, " n"],
        global_indexes: Int[Array, " n"],
        sequence_ids: Int[Array, " n"],
        mask_pad: Bool[Array, " n"],
    ) -> Float[Array, " n dim"]:
        emb = emb + self.attention(emb, sequence_indexes, global_indexes, sequence_ids, mask_pad)
        emb = emb + self.ff(emb)
        return emb


class MaskedLMHead(eqx.Module):
    linear_in: nn.Linear
    linear_out: nn.Linear
    norm: nn.LayerNorm

    def __init__(self, dim: int, *, key: PRNGKeyArray) -> None:
        key1, key2 = jax.random.split(key, 2)

        self.linear_in = nn.Linear(dim, dim, key=key1)
        self.linear_out = nn.Linear(dim, len(_constants.TOKENS), key=key2)
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x: Float[Array, " ... dim"]) -> Float[Array, " ... vocab_size"]:
        x = jax.vmap(self.linear_in)(x)
        x = gelu(x)
        x = jax.vmap(self.norm)(x)
        return jax.vmap(self.linear_out)(x)


@jaxtyped(typechecker=beartype)
class E1(eqx.Module):
    token_embed: nn.Embedding
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
        self.token_embed = nn.Embedding(len(_constants.TOKENS), dim, key=key1)
        self.sequence_embed = nn.Embedding(MAX_NUMBER_SEQUENCES, dim, key=key2)
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
        emb = jax.vmap(self.token_embed)(tokens) + jax.vmap(self.sequence_embed)(sequence_ids)
        for layer in self.layers:
            emb = layer(emb, sequence_indexes, global_indexes, sequence_ids, mask_pad)
        emb = jax.vmap(self.norm)(emb)
        return self.mlm_head(emb), emb

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        cache_dir: str | Path = _constants.DEFAULT_CACHE_DIR,
        force_download: bool = False,
    ) -> "E1":
        weights_path = get_weights_path(name, cache_dir)
        if not weights_path.is_file() or force_download:
            loguru.logger.info(
                "Weights not yet converted to Equinox, downloading them from the "
                "hugging face hub and converting them from torch."
            )
            convert_weights_from_torch(name, cache_dir)

        # The seed is not important here as there is no stochasticity at runtime.
        # It is only needed to initialize the model before the weights get loaded.
        model = cls(**MODEL_HYPERPARAMS[name], key=jax.random.PRNGKey(53))

        return eqx.tree_deserialise_leaves(path_or_file=weights_path, like=model)


def build_conversion_map(num_layers: int) -> dict[str, str]:
    conversion_map = {
        "token_embed.weight": "model.embed_tokens.weight",
        "sequence_embed.weight": "model.embed_seq_id.weight",
        "mlm_head.linear_in.weight": "mlm_head.0.weight",
        "mlm_head.linear_in.bias": "mlm_head.0.bias",
        "mlm_head.norm.weight": "mlm_head.2.weight",
        "mlm_head.norm.bias": "mlm_head.2.bias",
        "mlm_head.linear_out.weight": "mlm_head.3.weight",
        "mlm_head.linear_out.bias": "mlm_head.3.bias",
        "norm.weight": "model.norm.weight",
    }
    for k in range(num_layers):
        conversion_map.update(
            {
                f"layers.[{k}].attention.norm.weight": f"model.layers.{k}.norm_attn_norm.input_layernorm.weight",
                f"layers.[{k}].attention.to_q.weight": f"model.layers.{k}.norm_attn_norm.self_attn.q_proj.weight",
                f"layers.[{k}].attention.to_k.weight": f"model.layers.{k}.norm_attn_norm.self_attn.k_proj.weight",
                f"layers.[{k}].attention.to_v.weight": f"model.layers.{k}.norm_attn_norm.self_attn.v_proj.weight",
                f"layers.[{k}].attention.to_out.weight": f"model.layers.{k}.norm_attn_norm.self_attn.o_proj.weight",
                f"layers.[{k}].ff.linear_a.weight": f"model.layers.{k}.ffn.mlp.w1.weight",
                f"layers.[{k}].ff.linear_b.weight": f"model.layers.{k}.ffn.mlp.w3.weight",
                f"layers.[{k}].ff.linear_out.weight": f"model.layers.{k}.ffn.mlp.w2.weight",
                # the norms are stored in the attention class in the torch implementation
                # here, we pre-norm in the feedforward class
                f"layers.[{k}].ff.norm.weight": f"model.layers.{k}.norm_attn_norm.post_attention_layernorm.weight",
            }
        )
    return conversion_map


def get_weights_path(name: str, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / f"{name}.eqx"


def update_eqx_with_state_dict(
    module: eqx.Module, state_dict: dict, conversion_map: dict[str, str]
) -> eqx.Module:
    path_vals, treedef = jax.tree.flatten_with_path(module)
    updated_path_vals, count = [], 0
    array: jnp.ndarray
    for names, array in path_vals:
        key = ".".join(str(x).strip(".") for x in names)
        try:
            # convert to float to avoid unsupported bfloat16 dtype
            # the model can then be converted back to bfloat16 if needed
            weights = state_dict[conversion_map[key]].float()
            if not array.shape == weights.shape:
                weights = weights.T
            assert array.shape == weights.shape, f"{array.shape} != {weights.shape} for {key=}"
            updated_path_vals.append((names, jnp.asarray(weights)))
            count += 1
        except KeyError:
            updated_path_vals.append((names, array))

    updated_leaves = [v for _, v in updated_path_vals]
    updated_module = jax.tree.unflatten(treedef, updated_leaves)

    if not count == len(conversion_map):
        raise ValueError(
            f"Did not find all keys in conversion map: {count=}, {len(conversion_map)=}"
        )
    return updated_module


def convert_weights_from_torch(name: str, cache_dir: str | Path) -> None:
    import huggingface_hub as hf_hub
    import safetensors

    if name.startswith("Profluent-Bio/"):
        raise ValueError("Remove the leading 'Profluent-Bio/' from the model's name.")

    eqx_path = get_weights_path(name, cache_dir)
    eqx_path.parent.mkdir(parents=True, exist_ok=True)

    download_path = hf_hub.hf_hub_download(
        repo_id=f"Profluent-Bio/{name}", filename="model.safetensors"
    )
    with safetensors.safe_open(download_path, framework="pt", device="cpu") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
    # The key used to initialize the model is not important
    model = E1(**MODEL_HYPERPARAMS[name], key=jax.random.PRNGKey(52))
    conversion_map = build_conversion_map(MODEL_HYPERPARAMS[name]["num_layers"])
    updated_model = update_eqx_with_state_dict(model, state_dict, conversion_map)
    eqx.tree_serialise_leaves(eqx_path, updated_model)
