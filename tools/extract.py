"""Fetch GGUF headers from Hugging Face and build data/files.json.

Only the front of each file is downloaded — a range request covering the
metadata and tensor index. That is enough to describe a 1.5 TB model without
pulling a single weight.

    python tools/extract.py            # fetch what is missing, then build
    python tools/extract.py --refetch  # re-download the headers first

Cached headers land in data/raw/ and are git-ignored; they are large and
reproducible.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import struct
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gguf import BigArray, read_header  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "files.json"

HF = "https://huggingface.co"

# How much of each file to pull. The metadata block is dominated by the
# tokenizer, so it can run to tens of MB on the first shard and almost nothing
# on the others.
SOURCES = {
    "gemma": [("unsloth/gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf", 25_000_000)],
    "k3": [
        ("bullerwins/Kimi-K3-GGUF", f"mxfp4/Kimi-K3-mxfp4-{i:05d}-of-00004.gguf",
         60_000_000 if i == 1 else 8_000_000)
        for i in range(1, 5)
    ],
}


# --------------------------------------------------------------------------- fetch

def fetch(repo: str, path: str, nbytes: int, dest: Path, refetch: bool = False) -> Path:
    if dest.exists() and not refetch:
        return dest
    url = f"{HF}/{repo}/resolve/main/{path}"
    print(f"  fetching {nbytes / 1e6:.0f} MB of {path}")
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def repo_file_sizes(repo: str, match: str) -> dict[str, int]:
    """Real on-disk sizes from the HF tree API, following pagination."""
    url = f"{HF}/api/models/{repo}/tree/main?recursive=true&expand=true"
    sizes: dict[str, int] = {}
    while url:
        r = urllib.request.urlopen(url)
        for entry in json.load(r):
            if match in entry["path"] and entry["path"].endswith(".gguf"):
                sizes[entry["path"]] = entry.get("size", 0)
        link = r.headers.get("Link", "")
        url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None
    return sizes


# ----------------------------------------------------------------- shared helpers

def entry_groups(path: Path, meta_end: int) -> tuple[list[dict], int]:
    """Decode the first tensor record byte by byte, for the walkthrough."""
    raw = open(path, "rb").read()[meta_end:meta_end + 256]
    hexes = lambda a, b: [f"{x:02x}" for x in raw[a:b]]  # noqa: E731

    nlen = struct.unpack_from("<Q", raw, 0)[0]
    name = raw[8:8 + nlen].decode()
    pos = 8 + nlen
    ndim = struct.unpack_from("<I", raw, pos)[0]
    groups = [
        {"l": "name length", "v": str(nlen), "g": "g1", "b": hexes(0, 8)},
        {"l": f"name ({nlen} bytes)", "v": name, "g": "", "b": hexes(8, 8 + nlen)},
        {"l": "dim count", "v": str(ndim), "g": "g2", "b": hexes(pos, pos + 4)},
    ]
    pos += 4
    for i in range(ndim):
        dim = struct.unpack_from("<Q", raw, pos)[0]
        groups.append({"l": f"dim {i}", "v": str(dim), "g": "g1", "b": hexes(pos, pos + 8)})
        pos += 8
    from gguf import GGML_TYPES
    tid = struct.unpack_from("<I", raw, pos)[0]
    groups.append({"l": "type", "v": f"{tid} = {GGML_TYPES.get(tid, tid)}",
                   "g": "g2", "b": hexes(pos, pos + 4)})
    pos += 4
    off = struct.unpack_from("<Q", raw, pos)[0]
    groups.append({"l": "offset into data", "v": str(off), "g": "g1",
                   "b": hexes(pos, pos + 8)})
    return groups, pos + 8


def top_kv_spans(header, n: int = 5) -> tuple[list[dict], int]:
    ranked = sorted(header.kv_spans, key=lambda kv: -kv[1])
    return ([{"k": k, "b": b} for k, b in ranked[:n]],
            sum(b for _, b in ranked[n:]))


def layer_of(name: str):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def quant_totals(tensors) -> list[dict]:
    count, byte = collections.Counter(), collections.Counter()
    for t in tensors:
        count[t.type] += 1
        byte[t.type] += t.nbytes
    return [{"t": k, "n": count[k], "b": byte[k]} for k in sorted(byte, key=lambda x: -byte[x])]


def tensor_rows(tensors, layer: int) -> list[dict]:
    rows = [t for t in tensors if t.name.startswith(f"blk.{layer}.")]
    rows.sort(key=lambda t: t.offset)
    return [{"n": t.name.split(".", 2)[2], "d": t.shape, "t": t.type, "b": t.nbytes}
            for t in rows]


# ----------------------------------------------------------------------- Gemma

def build_gemma(head: Path) -> dict:
    g = read_header(head)
    T = g.tensors
    by_layer = collections.defaultdict(list)
    for t in T:
        i = layer_of(t.name)
        if i is not None:
            by_layer[i].append(t)

    n_layers = g.kv["gemma4.block_count"]
    layers = []
    for i in range(n_layers):
        ts = by_layer[i]
        attn = sum(t.nbytes for t in ts if ".attn" in t.name)
        ffn = sum(t.nbytes for t in ts if ".ffn" in t.name)
        total = sum(t.nbytes for t in ts)
        layers.append({
            "i": i,
            # a global layer carries no value projection at all
            "k": "local" if any(t.name.endswith("attn_v.weight") for t in ts) else "global",
            "n": len(ts), "b": total, "o": min(t.offset for t in ts),
            "s": [attn, ffn, total - attn - ffn],
            "v6": any("attn_v" in t.name and t.type == "Q6_K" for t in ts),
            "d6": any("ffn_down" in t.name and t.type == "Q6_K" for t in ts),
        })

    local = next(l["i"] for l in layers if l["k"] == "local")
    glob = next(l["i"] for l in layers if l["k"] == "global")
    groups, elen = entry_groups(head, g.meta_end)
    meta, meta_rest = top_kv_spans(g)

    return {
        "id": "gemma", "tab": "Gemma 4 12B", "tabsub": "Q4_K_M · 7.12 GB · ordinary",
        "name": "gemma-4-12b-it-Q4_K_M.gguf", "arch": g.kv["general.architecture"],
        "ver": g.version, "nt": g.n_tensors, "nkv": g.n_kv, "shards": 1,
        "hex": [["47 47 55 46", "GGUF", "m"], ["03 00 00 00", "version 3", "h"],
                ["9b 02 00 00 00 00 00 00", f"{g.n_tensors} tensors", "n"],
                ["3a 00 00 00 00 00 00 00", f"{g.n_kv} keys", "h"]],
        "metaBytes": g.metadata_bytes, "indexBytes": g.index_bytes,
        "dataStart": g.data_start, "tensorBytes": g.data_bytes,
        "fileBytes": g.total_bytes,
        "meta": meta, "metaRest": meta_rest,
        "entry": {"g": groups, "len": elen},
        "stats": [["Architecture", g.kv["general.architecture"]],
                  ["Tensors", f"{g.n_tensors}"], ["Metadata keys", f"{g.n_kv}"],
                  ["Layers", f"{n_layers}"], ["Shards", "1"],
                  ["File size", "7.12 <small>GB</small>"]],
        "segs": [["attention", "var(--signal)", 1],
                 ["feed-forward", "var(--brass)", 1],
                 ["norms & scales", "var(--f32)", 1]],
        "layers": layers,
        "kindA": {"label": f"blk.{local} — local · window 1024", "t": tensor_rows(T, local)},
        "kindB": {"label": f"blk.{glob} — global · full 256K", "t": tensor_rows(T, glob)},
        "kindNote": (
            "The global layer <b>doubles the head width</b> — <code>attn_q</code> goes "
            "3840×4096 → 3840×8192 — but <b>cuts key-value heads from 8 to 1</b>: "
            "<code>attn_k</code> goes 3840×2048 → 3840×512. The layers that must remember "
            "everything are built to keep almost nothing per token."),
        "quant": quant_totals(T),
        "extra": [{"n": t.name, "d": t.shape, "t": t.type, "b": t.nbytes}
                  for t in T if layer_of(t.name) is None],
        "promo": True,
        "kv": {
            "fixed": 40 * 8192 * 1024, "per": 8 * (1 * 512 * 2 * 2),
            "ctx": [4096, 32768, 131072, 262144],
            "note": ("The 40 local layers cache <b>8 heads × 256</b> but only 1024 words — "
                     "their share is <b>fixed at 336 MB no matter how long the conversation "
                     "runs</b>. Only the 8 global layers grow, and they were cut to "
                     "<b>1 head × 512</b> so they could afford to."),
            "warn": ("Computed from the shapes above at 16-bit, assuming keys and values are "
                     "both stored at the global layers' width. The file records no value "
                     "projection on those layers, so their true cost may be half of what is shown."),
        },
        "close": ("<b>7.11 GB</b> of tensors + <b>15.82 MB</b> of header, metadata and index "
                  "= <b>7.122 GB</b>. Published size: <b>7.12 GB</b>."),
        "sections": ["regions", "meta", "entry", "layermap", "kinds", "promo", "totals", "ram"],
    }


# -------------------------------------------------------------------------- K3

K3_HOLDS = [
    "the 896-expert tensors for layers 1–29,<br>plus the tokenizer and all model settings",
    "the 896-expert tensors for layers 29–57",
    "the 896-expert tensors for layers 57–85",
    "everything else — attention, norms and the<br>vocabulary table, for all 93 layers",
]


def k3_role(name: str) -> str:
    n = name.split(".", 2)[2] if name.startswith("blk.") else name
    if "_exps" in n:
        return "exp"
    if "shexp" in n:
        return "shx"
    if n.startswith("ffn_routed"):
        return "rtd"
    if n.startswith("ffn_gate_inp") or "exp_probs" in n:
        return "rtr"
    if n.startswith("ssm_"):
        return "ssm"
    if n.startswith("attn"):
        return "att"
    # blk.0 is the one dense layer: plain ffn_gate/up/down, no experts.
    # ffn_norm and ffn_res_score are norms despite the prefix.
    if n.startswith("ffn") and "norm" not in n and "res_score" not in n:
        return "ffn"
    return "nrm"


def k3_ladder() -> list[list]:
    """Every published K3 build, smallest first. The original mxfp4 is the ceiling —
    even the 'Q8' build is the same size, because it kept the 4-bit tensors."""
    import collections as _c
    tot = _c.Counter()
    for path, size in repo_file_sizes("unsloth/Kimi-K3-GGUF", "").items():
        if "mmproj" in path:
            continue
        name = re.sub(r"-\d{5}-of-\d{5}", "", path.split("/")[-1]).removeprefix(
            "Kimi-K3-").removesuffix(".gguf")
        tot[name] += size
    rows = [[n, b / 1e9] for n, b in sorted(tot.items(), key=lambda kv: kv[1])]
    rows.append(["mxfp4 (original)", 1561.1578536])
    return rows


def build_k3(heads: list[Path], sizes: list[int]) -> dict:
    parsed = [read_header(h) for h in heads]
    first = parsed[0]
    all_t = [t for p in parsed for t in p.tensors]
    total_data = sum(t.nbytes for t in all_t)

    n_layers = first.kv["kimi-k3.block_count"]
    used = first.kv["kimi-k3.expert_used_count"]
    n_exp = first.kv["kimi-k3.expert_count"]
    kv_heads = first.kv["kimi-k3.attention.head_count_kv"]
    if isinstance(kv_heads, BigArray):
        raise RuntimeError("head_count_kv came back as a BigArray; raise BIG_ARRAY")

    shard_of = {}
    for si, p in enumerate(parsed, 1):
        for t in p.tensors:
            shard_of[t.name] = si

    seg_keys = ["att", "exp", "shx", "rtd", "nrm"]

    def seg_bucket(name: str) -> str:
        r = k3_role(name)
        if r == "ssm":
            return "att"      # linear-attention machinery is part of attention
        if r in ("rtr", "ffn"):
            return "rtd"      # router and the one dense layer share the plumbing column
        return r

    per_layer = collections.defaultdict(lambda: collections.Counter())
    counts = collections.Counter()
    for t in all_t:
        i = layer_of(t.name)
        if i is not None:
            per_layer[i][seg_bucket(t.name)] += t.nbytes
            counts[i] += 1

    layers = []
    for i in range(n_layers):
        d = per_layer[i]
        kind = "dense" if i == 0 else ("full" if kv_heads[i] > 0 else "kda")
        layers.append({"i": i, "k": kind, "n": counts[i], "b": sum(d.values()),
                       "sh": shard_of.get(f"blk.{i}.ffn_gate_exps.weight", 4),
                       "s": [d[k] for k in seg_keys]})

    # anatomy of one representative expert-bearing layer
    ANAT = {"exp": "The 896 experts", "att": "Attention",
            "shx": "Shared expert · always runs", "ssm": "Linear-attention machinery",
            "rtd": "Expert in / out projection", "rtr": "Router · picks the 16",
            "ffn": "Dense feed-forward · blk.0 only", "nrm": "Norms"}
    sample = [t for t in all_t if t.name.startswith("blk.1.")]
    agg, agg_n = collections.Counter(), collections.Counter()
    for t in sample:
        agg[ANAT[k3_role(t.name)]] += t.nbytes
        agg_n[ANAT[k3_role(t.name)]] += 1
    order = sorted(agg, key=lambda x: -agg[x])
    exp_b = agg["The 896 experts"]

    active = sum(t.nbytes * used / n_exp if "_exps" in t.name else t.nbytes for t in all_t)
    n_weights = sum(t.n_weights for t in all_t)
    exp_w = sum(t.n_weights for t in all_t if "_exps" in t.name)

    groups, elen = entry_groups(heads[0], first.meta_end)
    meta, meta_rest = top_kv_spans(first)

    shard_table = []
    for p, size, holds in zip(parsed, sizes, K3_HOLDS):
        front = 24 + p.metadata_bytes + p.index_bytes
        shard_table.append({"s": len(shard_table) + 1, "nt": p.n_tensors, "nkv": p.n_kv,
                            "hdr": 24, "meta": p.metadata_bytes, "index": p.index_bytes,
                            "front": front, "data": p.data_bytes, "file": size,
                            "holds": holds})

    full_n = sum(1 for l in layers if l["k"] == "full")
    kda_n = sum(1 for l in layers if l["k"] == "kda")

    return {
        "id": "k3", "tab": "Kimi K3", "tabsub": "mxfp4 · 1.56 TB · the strange one",
        "name": "Kimi-K3-mxfp4-0000N-of-00004.gguf",
        "arch": first.kv["general.architecture"], "ver": first.version,
        "nt": first.kv["split.tensors.count"], "nkv": first.n_kv, "shards": len(parsed),
        "hex": [["47 47 55 46", "GGUF", "m"], ["03 00 00 00", "version 3", "h"],
                ["55 00 00 00 00 00 00 00", f"{first.n_tensors} tensors in shard 1", "n"],
                ["32 00 00 00 00 00 00 00", f"{first.n_kv} keys", "h"]],
        "metaBytes": first.metadata_bytes, "indexBytes": first.index_bytes,
        "dataStart": first.data_start, "tensorBytes": total_data,
        "fileBytes": sum(sizes),
        "meta": meta, "metaRest": meta_rest,
        "entry": {"g": groups, "len": elen},
        "stats": [["Architecture", first.kv["general.architecture"]],
                  ["Tensors", f"{first.kv['split.tensors.count']:,}".replace(",", " ")],
                  ["Metadata keys", f"{first.n_kv}"], ["Layers", f"{n_layers}"],
                  ["Shards", f"{len(parsed)}"], ["Total size", "1.56 <small>TB</small>"]],
        "segs": [["attention", "var(--signal)", 1], ["experts (×896)", "var(--brass)", 1],
                 ["shared experts", "var(--brass)", 0.5],
                 ["feed-forward & routing", "var(--signal)", 0.5],
                 ["norms & misc", "var(--f32)", 1]],
        "layers": layers,
        "shardTable": shard_table, "shardTotal": sum(sizes),
        "anatomy": {"total": sum(agg.values()),
                    "rows": [{"l": x, "n": agg_n[x], "b": agg[x]} for x in order],
                    "expB": exp_b, "restB": sum(agg.values()) - exp_b},
        "expert": {"used": used, "total": n_exp, "stored": total_data, "read": active,
                   "params": n_weights, "activeParams": exp_w * used / n_exp + (n_weights - exp_w),
                   "expParams": exp_w},
        "strip": [l["k"] for l in layers],
        "ladder": k3_ladder(),
        "attncmp": [
            ["how many layers", f"{kda_n} of {n_layers}", f"{full_n} of {n_layers}"],
            ["what it remembers", "a fixed-size summary", "every word, compressed"],
            ["cache per word", "nothing — size never changes", "1 152 bytes"],
            ["at 1 million words", "0 GB", "29 GB"],
            ["cost per layer", "17.00 GB", "16.58 GB"],
        ],
        "kindA": {"label": "blk.1 — KDA linear attention + 896 experts", "t": tensor_rows(all_t, 1)},
        "kindB": {"label": "blk.3 — full attention (MLA) + 896 experts", "t": tensor_rows(all_t, 3)},
        "kindNote": (
            f"Only <b>{full_n} of {n_layers}</b> layers use ordinary attention; the other "
            f"{kda_n} use a linear variant with no growing cache at all. And the attention "
            "tensors were <b>never quantized</b> — they are BF16 in a file that is otherwise "
            "4-bit. Compare the two <code>attn</code> blocks: the KDA layer carries convolution "
            "and gate tensors instead of a key-value cache."),
        "quant": quant_totals(all_t),
        "extra": [{"n": t.name, "d": t.shape, "t": t.type, "b": t.nbytes}
                  for t in all_t if layer_of(t.name) is None],
        "promo": False,
        "roles": [{"r": {"exp": "experts (×896, stacked)", "att": "attention",
                         "shx": "shared experts", "rtd": "expert in / out projection",
                         "ffn": "dense feed-forward (blk.0)",
                         "rtr": "router", "ssm": "linear-attention machinery",
                         "nrm": "norms & misc"}[r],
                   "n": n, "b": b}
                  for r, n, b in sorted(
                      ((r, sum(1 for t in all_t if k3_role(t.name) == r),
                        sum(t.nbytes for t in all_t if k3_role(t.name) == r))
                       for r in {k3_role(t.name) for t in all_t}),
                      key=lambda x: -x[2])],
        "roleNote": (
            "Attention was <b>deliberately left alone</b>. The quantizer config excludes "
            "<code>self_attn</code> and the shared experts by name — so in a file that is "
            "92.65% 4-bit, the parts that route and attend stay at full 16-bit precision."),
        "kv": {
            "fixed": 217_000_000, "per": 24 * (576 * 2),
            "ctx": [4096, 32768, 262144, 1048576],
            "note": (f"Only the <b>{full_n} full-attention layers</b> cache anything that "
                     "grows, and they compress it to a <b>576-value latent</b> per token "
                     f"instead of storing full keys and values. The other {kda_n} layers hold "
                     "a fixed-size state that never grows with the conversation."),
            "warn": ("Growing cache computed from key_length 576 at 16-bit across the 24 "
                     "full-attention layers. The fixed 217 MB for the 69 linear layers is "
                     "derived from head count and head dimension, not read from the file."),
        },
        "close": ("<b>1.561 TB</b> of tensors across 4 shards. Published total: "
                  "<b>1.561 TB</b> — and the original safetensors release is "
                  "<b>the same 1.561 TB</b>."),
        "sections": ["zoom", "anatomy", "experts", "attncmp", "ram", "RAW",
                     "strip", "shards", "meta", "entry", "kinds", "promo", "totals"],
    }


# ------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true", help="re-download cached headers")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    print("Gemma 4 12B")
    repo, path, nbytes = SOURCES["gemma"][0]
    gemma_head = fetch(repo, path, nbytes, RAW / "gemma-4-12b-q4km.head", args.refetch)

    print("Kimi K3")
    k3_heads = [fetch(r, p, n, RAW / f"kimi-k3-mxfp4-{i}.head", args.refetch)
                for i, (r, p, n) in enumerate(SOURCES["k3"], 1)]
    print("  reading shard sizes from the HF tree API")
    sizes_by_path = repo_file_sizes("bullerwins/Kimi-K3-GGUF", "mxfp4")
    k3_sizes = [sizes_by_path[p] for _, p, _ in SOURCES["k3"]]

    files = {"gemma": build_gemma(gemma_head), "k3": build_k3(k3_heads, k3_sizes)}
    OUT.write_text(json.dumps(files, separators=(",", ":")))

    print(f"\nwrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1e3:.1f} KB)")
    for key, d in files.items():
        print(f"  {key:<6} {d['nt']:>5} tensors · {d['shards']} shard(s) · "
              f"{d['tensorBytes'] / 1e9:,.2f} GB of weights")

    # the whole point of the project: the parse must reconcile
    for row in files["k3"]["shardTable"]:
        pad = (-row["front"]) % 32
        computed = row["front"] + pad + row["data"]
        status = "OK" if computed == row["file"] else "MISMATCH"
        print(f"  shard {row['s']}: {computed:,} vs {row['file']:,}  {status}")


if __name__ == "__main__":
    main()
