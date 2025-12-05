from typing import Final

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


@jaxtyped(typechecker=beartype)
def tokenize(sequence: str) -> Int[Array, " n"]:
    tokens = (
        [TOKENS["<cls>"]] + [TOKENS.get(char, TOKENS["X"]) for char in sequence] + [TOKENS["<eos>"]]
    )
    return jnp.array(tokens, dtype=jnp.int32)


@jaxtyped(typechecker=beartype)
def pad_and_mask(
    tokens: Int[Array, " n"], pad_length: int = 0
) -> tuple[Int[Array, " m"], Bool[Array, " m"]]:
    if pad_length is None:
        pad_length = 0

    mask = jnp.array([True] * tokens.shape[0] + [False] * pad_length, dtype=jnp.bool_)
    tokens = jnp.concat((tokens, jnp.zeros(pad_length, dtype=tokens.dtype)), axis=0)

    return tokens, mask
