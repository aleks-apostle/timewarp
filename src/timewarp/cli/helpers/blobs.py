from __future__ import annotations

from typing import cast

from ...codec import from_bytes
from ...events import BlobRef
from ...store import LocalStore


def read_json_blob(store: LocalStore, ref: BlobRef | None) -> object | None:
    """Read a JSON blob via the store and decode using canonical codec.

    Returns None when the reference is missing or decoding fails.
    """
    if ref is None:
        return None
    try:
        obj = from_bytes(store.get_blob(ref))
        return cast(object, obj)
    except Exception:
        return None
