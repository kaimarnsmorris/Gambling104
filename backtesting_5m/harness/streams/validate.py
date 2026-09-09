"""Read a stream's self-description, and refuse anything ambiguous.

Parquet key-value metadata lives in the footer and is parsed once per file at
open, never per row -- it costs nothing at read time. It is also robust across
writer versions: pyarrow 19 reads these files' metadata even where it cannot
read their data pages.
"""
import glob
import os

import pyarrow.parquet as pq

from harness.streams.spec import META_PREFIX, RESERVED, StreamInvalid


def _first_file(path):
    if os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "**", "*.parquet"),
                                recursive=True))
        if not hits:
            raise StreamInvalid(f"{path}: no parquet files found")
        return hits[0]
    return path


def validate_stream(path):
    """Return the parsed stream metadata, or raise StreamInvalid."""
    f = _first_file(path)
    md = pq.ParquetFile(f).schema_arrow.metadata or {}
    kv = {k.decode(): v.decode() for k, v in md.items()
          if k.decode().startswith(META_PREFIX)}
    if not kv:
        raise StreamInvalid(
            f"{path}: no stream.* metadata. Non-conforming files need an "
            f"explicit Stream(...) adapter at registration.")

    out = {k[len(META_PREFIX):]: v for k, v in kv.items()}
    if "causal" not in out:
        raise StreamInvalid(
            f"{path}: metadata has no stream.causal. Whether a series may be "
            f"carried forward onto a decision grid is not safe to guess.")
    out["causal"] = out["causal"].lower() == "true"

    cols = [c for c in pq.ParquetFile(f).schema_arrow.names]
    if "recv_ns" not in cols:
        raise StreamInvalid(f"{path}: a stream must carry recv_ns")
    out["values"] = tuple(c for c in cols if c not in RESERVED)
    out["path"] = path
    return out
