from typing import NamedTuple

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Int, jaxtyped

from e1jax import _constants


class Tokenized(NamedTuple):
    tokens: Int[Array, " n"]
    sequence_indexes: Int[Array, " n"]
    global_indexes: Int[Array, " n"]
    sequence_ids: Int[Array, " n"]
    mask_pad: Bool[Array, " n"]


@jaxtyped(typechecker=beartype)
def tokenize(sequence: str) -> Tokenized:
    tokens = jnp.array(
        [_constants.TOKENS["<bos>"], _constants.TOKENS["1"]]
        + [_constants.TOKENS.get(char, _constants.TOKENS["X"]) for char in sequence]
        + [_constants.TOKENS["2"], _constants.TOKENS["<eos>"]],
        dtype=jnp.int32,
    )
    n = tokens.shape[0]
    tokenized = Tokenized(
        tokens=tokens,
        sequence_indexes=jnp.arange(n, dtype=jnp.int32),
        global_indexes=jnp.arange(n, dtype=jnp.int32),
        sequence_ids=jnp.array([0] * n, dtype=jnp.int32),
        mask_pad=jnp.array([True] * n, dtype=jnp.bool),
    )
    return tokenized


@jaxtyped(typechecker=beartype)
def pad_and_mask(tokenized: Tokenized, pad_length: int | None) -> Tokenized:
    if pad_length is None:
        return tokenized

    mask_pad = jnp.array([False] * pad_length, dtype=jnp.bool)
    tokens_pad = jnp.array([_constants.TOKENS["<pad>"]] * pad_length, dtype=jnp.int32)
    sequence_indexes_pad = jnp.zeros([pad_length], dtype=jnp.int32)
    global_indexes_pad = jnp.zeros([pad_length], dtype=jnp.int32)
    sequence_ids_pad = jnp.zeros([pad_length], dtype=jnp.int32)
    padded_tokenized = Tokenized(
        tokens=jnp.concatenate([tokenized.tokens, tokens_pad]),
        sequence_indexes=jnp.concatenate([tokenized.sequence_indexes, sequence_indexes_pad]),
        global_indexes=jnp.concatenate([tokenized.global_indexes, global_indexes_pad]),
        sequence_ids=jnp.concatenate([tokenized.sequence_ids, sequence_ids_pad]),
        mask_pad=jnp.concatenate([tokenized.mask_pad, mask_pad]),
    )
    return padded_tokenized
