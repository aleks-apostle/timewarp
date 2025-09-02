from __future__ import annotations

import random

from timewarp.determinism import restore_rng, snapshot_rng


def test_rng_snapshot_restore() -> None:
    random.seed(123)
    _ = [random.random() for _ in range(5)]
    st = snapshot_rng()
    b = [random.random() for _ in range(5)]
    restore_rng(st)
    c = [random.random() for _ in range(5)]
    assert b == c
