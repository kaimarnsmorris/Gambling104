"""The vendored clkit must never write into the read-only SRC checkout.

`common.ALIGNED` points at a 200 MB cache file inside SRC. If it is ever missing,
`build_aligned()` must fail loudly rather than rebuild it in place - that rebuild
would write straight into a tree this investigation does not own. This does not
touch the real cache file; it only points `ALIGNED` at a path that does not exist.
"""
from __future__ import annotations

import pytest


def test_build_aligned_refuses_to_rebuild_when_missing(monkeypatch, tmp_path):
    from fvmodel._vendor.clkit import common

    missing = tmp_path / "aligned_1s.parquet"
    assert not missing.exists()
    monkeypatch.setattr(common, "ALIGNED", missing)

    with pytest.raises(FileNotFoundError, match=r"aligned_1s\.parquet"):
        common.build_aligned()
