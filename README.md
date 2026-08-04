# gguf-breakdown

One static page explaining what is actually inside a GGUF model file, built
from the real files rather than from documentation.

- **Part one — how it works.** A model file loaded and run, step by step, for
  engineers who don't work with ML. Includes a selector for the four Gemma 4
  variants with numbers derived live from each model's config.
- **Part two — inside two real files.** A byte-exact teardown of
  `gemma-4-12b-it-Q4_K_M.gguf` (7.12 GB) and the full original `Kimi-K3-mxfp4`
  (1.56 TB, four shards).
- **Glossary side panel.** Every term — starting with *tensor* — defined in
  plain language. The first mention of each term anywhere in the prose is
  clickable and opens the panel at that definition.

One file, no dependencies, no server. Open `index.html` off disk.

## Why the numbers can be trusted

Every figure in the teardown was parsed out of the actual files on Hugging Face.
A GGUF stores its header, metadata and tensor index at the front, so an HTTP
range request over the first few MB describes the whole model — **no weight data
is ever downloaded**, including for the 1.56 TB model.

The parse is checked by reconstruction. Header + metadata + index + padding +
tensor bytes must equal the real published file size, and it does, to the byte:

```
shard 1: 445,473,457,024 vs 445,473,457,024  OK
shard 2: 445,466,548,512 vs 445,466,548,512  OK
shard 3: 445,466,548,512 vs 445,466,548,512  OK
shard 4: 224,751,299,552 vs 224,751,299,552  OK
```

Anything not read from a file — KV-cache sizes, bits-per-weight — is computed
from shapes the file does record, and is labelled as such on the page.

## Layout

```
build.py              template + data -> index.html
templates/index.html  page source (no <html>/<head> wrapper; build.py adds it)
data/files.json       the parsed dataset, embedded into the page at build
data/raw/             cached GGUF headers (git-ignored, ~110 MB)
tools/gguf.py         GGUF header reader — library and CLI
tools/extract.py      fetch headers from Hugging Face, write data/files.json
```

## Rebuild

```bash
python3 build.py                 # page from existing data — this is usually all you need
python3 tools/extract.py         # re-fetch headers and regenerate data/files.json
python3 tools/extract.py --refetch
```

`extract.py` caches downloaded headers in `data/raw/`, so a second run costs
nothing. Only `build.py` is needed to change copy or design.

## Inspecting any GGUF

`tools/gguf.py` works on any GGUF file, not just these two:

```bash
python3 tools/gguf.py model.gguf
python3 tools/gguf.py model.gguf --tensors   # every tensor, shape, type, offset
python3 tools/gguf.py model.gguf --kv        # every metadata key and its cost on disk
```

It also accepts a truncated file, which is the point — `head -c 25000000` of a
GGUF, or a range request, is enough:

```
GGUF v3 · 667 tensors · 58 keys
  metadata     15.78 MB
  index        39.86 KB
  data          7.11 GB
  total         7.12 GB  (data starts at 0xF17540)

storage types:
  Q4_K       285 tensors     5.81 GB
  Q6_K        44 tensors     1.29 GB
  F32        338 tensors     3.08 MB
```

## Sources

| | |
|---|---|
| Gemma 4 | [model card](https://ai.google.dev/gemma/docs/core/model_card_4) · [GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) |
| Kimi K3 | [weights](https://huggingface.co/moonshotai/Kimi-K3) · [GGUF](https://huggingface.co/bullerwins/Kimi-K3-GGUF) |

Requires Python 3.9+. Standard library only.
