from typing import Final, NamedTuple

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Int, jaxtyped

TOKENS: Final[dict[str, int]] = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<bos_glm>": 3,
    "<eos_span>": 4,
    "?": 5,
    "1": 6,
    "2": 7,
    "A": 8,
    "B": 9,
    "C": 10,
    "D": 11,
    "E": 12,
    "F": 13,
    "G": 14,
    "H": 15,
    "I": 16,
    "J": 17,
    "K": 18,
    "L": 19,
    "M": 20,
    "N": 21,
    "O": 22,
    "P": 23,
    "Q": 24,
    "R": 25,
    "S": 26,
    "T": 27,
    "U": 28,
    "V": 29,
    "W": 30,
    "X": 31,
    "Y": 32,
    "Z": 33,
}


class Tokenized(NamedTuple):
    tokens: Int[Array, " n"]
    sequence_indexes: Int[Array, " n"]
    global_indexes: Int[Array, " n"]
    sequence_ids: Int[Array, " n"]
    mask_pad: Bool[Array, " n"]


@jaxtyped(typechecker=beartype)
def tokenize(sequence: str) -> Tokenized:
    tokens = jnp.array(
        [TOKENS["<bos>"], TOKENS["1"]]
        + [TOKENS.get(char, TOKENS["X"]) for char in sequence]
        + [TOKENS["2"], TOKENS["<eos>"]],
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
