"""Minimal GGUF header reader.

A GGUF file is: a 24-byte header, a table of key/value metadata, a table of
tensor records, then the tensor data. Everything except the data lives at the
front, so a range request for the first few MB is enough to describe the whole
model — including files that are terabytes long.

Usage as a library:
    from gguf import read_header
    g = read_header("model.gguf")
    g.tensors[0].name, g.tensors[0].shape, g.tensors[0].nbytes

Usage as a CLI:
    python tools/gguf.py model.gguf            # summary
    python tools/gguf.py model.gguf --tensors  # every tensor
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# --- ggml storage types -------------------------------------------------------
# id -> name. Only the types we have actually encountered are listed; anything
# else surfaces as "?<id>" rather than being silently mis-sized.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4",
}

# name -> (weights per block, bytes per block). Quantised types pack a whole
# block of weights plus its scale into a fixed number of bytes.
BLOCK_SIZES = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "F64": (1, 8),
    "I8": (1, 1), "I16": (1, 2), "I32": (1, 4), "I64": (1, 8),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q8_1": (32, 36),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
    "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
    "IQ1_S": (256, 50), "IQ1_M": (256, 56), "IQ2_XXS": (256, 66),
    "IQ2_XS": (256, 74), "IQ2_S": (256, 82), "IQ3_XXS": (256, 98),
    "IQ3_S": (256, 110), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136),
    "TQ1_0": (256, 54), "TQ2_0": (256, 66),
    "MXFP4": (32, 17),
}

# GGUF metadata value types -> (struct code, width)
_SCALAR = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
    5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8),
    12: ("d", 8),
}
_STRING, _ARRAY = 8, 9

# Arrays longer than this are recorded as a count instead of being kept in
# memory — a tokenizer vocabulary is half a million strings.
BIG_ARRAY = 4096


@dataclass
class Tensor:
    name: str
    shape: list[int]
    type: str
    offset: int          # relative to the start of the data section
    nbytes: int

    @property
    def n_weights(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class BigArray:
    """A metadata array too long to keep; we record what it was."""
    kind: str
    count: int
    sample: list = field(default_factory=list)


@dataclass
class Header:
    version: int
    n_tensors: int          # tensors in *this* file (a shard holds a subset)
    n_kv: int
    kv: dict
    tensors: list[Tensor]
    kv_spans: list[tuple[str, int]]   # (key, bytes on disk) — what the file spends
    meta_end: int           # byte after the last metadata value
    index_end: int          # byte after the last tensor record
    data_start: int         # first byte of tensor data, after alignment padding

    @property
    def metadata_bytes(self) -> int:
        return self.meta_end - 24

    @property
    def index_bytes(self) -> int:
        return self.index_end - self.meta_end

    @property
    def data_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def total_bytes(self) -> int:
        """Reconstructed file size. Should equal the real file exactly."""
        return self.data_start + self.data_bytes


def tensor_nbytes(type_name: str, shape: list[int]) -> int:
    """Bytes on disk for a tensor of this shape and storage type."""
    if type_name not in BLOCK_SIZES:
        raise ValueError(f"unknown ggml type {type_name!r}; cannot size it")
    per_block, block_bytes = BLOCK_SIZES[type_name]
    n = 1
    for d in shape:
        n *= d
    if n % per_block:
        raise ValueError(f"{n} weights is not a multiple of block size {per_block}")
    return n // per_block * block_bytes


class _Reader:
    def __init__(self, buf: bytes):
        self.buf, self.pos = buf, 0

    def scalar(self, code: str, width: int):
        try:
            v = struct.unpack_from("<" + code, self.buf, self.pos)[0]
        except struct.error:
            raise EOFError(
                "ran off the end of the buffer — fetch more of the file "
                "(the metadata block can be tens of MB)"
            )
        self.pos += width
        return v

    def string(self) -> str:
        n = self.scalar("Q", 8)
        s = self.buf[self.pos:self.pos + n]
        if len(s) < n:
            raise EOFError("truncated string — fetch more of the file")
        self.pos += n
        return s.decode("utf-8", "replace")

    def value(self, vtype: int):
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype = self.scalar("I", 4)
            count = self.scalar("Q", 8)
            if etype == _STRING:
                if count > BIG_ARRAY:
                    sample = [self.string() for _ in range(min(4, count))]
                    for _ in range(count - len(sample)):
                        self.string()
                    return BigArray("str", count, sample)
                return [self.string() for _ in range(count)]
            code, width = _SCALAR[etype]
            if count > BIG_ARRAY:
                self.pos += count * width
                return BigArray(GGML_TYPES.get(etype, str(etype)), count)
            vals = list(struct.unpack_from(f"<{count}{code}", self.buf, self.pos))
            self.pos += count * width
            return vals
        return self.scalar(*_SCALAR[vtype])


def read_header(source) -> Header:
    """Parse a GGUF header from a path or a bytes object.

    Only the front of the file is needed; passing a truncated download is
    fine as long as it covers the metadata and tensor index.
    """
    buf = source if isinstance(source, (bytes, bytearray)) else open(source, "rb").read()
    if buf[:4] != b"GGUF":
        raise ValueError("not a GGUF file (bad magic)")

    r = _Reader(buf)
    r.pos = 4
    version = r.scalar("I", 4)
    n_tensors = r.scalar("Q", 8)
    n_kv = r.scalar("Q", 8)

    kv, spans = {}, []
    for _ in range(n_kv):
        start = r.pos
        key = r.string()
        kv[key] = r.value(r.scalar("I", 4))
        spans.append((key, r.pos - start))
    meta_end = r.pos

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        shape = [r.scalar("Q", 8) for _ in range(r.scalar("I", 4))]
        tid = r.scalar("I", 4)
        type_name = GGML_TYPES.get(tid, f"?{tid}")
        offset = r.scalar("Q", 8)
        tensors.append(Tensor(name, shape, type_name, offset,
                              tensor_nbytes(type_name, shape)))
    index_end = r.pos

    align = kv.get("general.alignment", 32)
    data_start = (index_end + align - 1) // align * align

    return Header(version, n_tensors, n_kv, kv, tensors, spans,
                  meta_end, index_end, data_start)


def human(b: float) -> str:
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= scale:
            return f"{b / scale:.2f} {unit}"
    return f"{b:.0f} B"


def _main() -> None:
    import argparse
    import collections

    ap = argparse.ArgumentParser(description="Describe a GGUF file from its header.")
    ap.add_argument("path")
    ap.add_argument("--tensors", action="store_true", help="list every tensor")
    ap.add_argument("--kv", action="store_true", help="list every metadata key")
    args = ap.parse_args()

    g = read_header(args.path)
    print(f"GGUF v{g.version} · {g.n_tensors} tensors · {g.n_kv} keys")
    print(f"  metadata   {human(g.metadata_bytes):>10}")
    print(f"  index      {human(g.index_bytes):>10}")
    print(f"  data       {human(g.data_bytes):>10}")
    print(f"  total      {human(g.total_bytes):>10}  (data starts at 0x{g.data_start:X})")

    by_type = collections.Counter()
    for t in g.tensors:
        by_type[t.type] += t.nbytes
    print("\nstorage types:")
    for name, nbytes in by_type.most_common():
        count = sum(1 for t in g.tensors if t.type == name)
        print(f"  {name:<8} {count:>5} tensors  {human(nbytes):>10}")

    if args.kv:
        print("\nmetadata:")
        for key, nbytes in sorted(g.kv_spans, key=lambda kv: -kv[1]):
            val = g.kv[key]
            if isinstance(val, BigArray):
                shown = f"<{val.kind}[{val.count:,}]>"
            else:
                shown = str(val)
            print(f"  {key:<48} {human(nbytes):>9}  {shown[:60]}")

    if args.tensors:
        print("\ntensors:")
        for t in g.tensors:
            shape = "×".join(str(d) for d in t.shape)
            print(f"  {t.name:<44} {shape:>22} {t.type:>7} {human(t.nbytes):>10}  @0x{t.offset:X}")


if __name__ == "__main__":
    _main()
