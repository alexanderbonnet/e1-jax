# e1-jax

A minimal implementation (meant for inference only) of the [E1 protein language model](https://www.profluent.bio/showcase/e1) family with Jax/Equinox.
Logits and embeddings were matched against the [original repository](https://github.com/Profluent-AI/E1).

Dependencies are managed with [uv](https://docs.astral.sh/uv/). To install them, run `uv sync`.

Currently, only **single sequence** inference is supported.

## Example

The implementation is compatible with `equinox.filter_{vmap, jit}` for batched and jitted inference.

The model can be one of `E1-150m`, `E1-300m`, `E1-600m`.

```python
import e1jax

seq = "AAAAA?C"
pad_length = 10
tokenized = e1jax.tokenize(seq)
tokenized = e1jax.pad_and_mask(tokenized, pad_length=pad_length)

model = e1jax.E1.from_pretrained("E1-300m")

logits, embeddings = model(**tokenized._asdict())

# to remove boundary tokens and padding
lb, rb = 2, -2-pad_length
logits, embeddings = logits[lb:rb], embeddings[lb:rb]

assert logits.shape[0] == len(seq)
assert embeddings.shape[0] == len(seq)
```

## Citations

```bash
 @article{Jain_Beazer_Ruffolo_Bhatnagar_Madani_2025,
    title={E1: Retrieval-Augmented Protein Encoder Models},
    url={https://www.biorxiv.org/content/early/2025/11/13/2025.11.12.688125},
    DOI={10.1101/2025.11.12.688125},
    journal={bioRxiv},
    publisher={Cold Spring Harbor Laboratory},
    author={Jain, Sarthak and Beazer, Joel and Ruffolo, Jeffrey A and Bhatnagar, Aadyot and Madani, Ali},
    year={2025}
}
```
