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
