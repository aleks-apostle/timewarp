from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .codec import from_bytes
from .events import Event, redact
from .store import LocalStore
from .utils.diffing import struct_diff as _struct_diff
from .utils.diffing import text_unified as _text_unified
from .utils.diffing import to_text as _to_text


@dataclass
class Divergence:
    step_a: int
    step_b: int
    reason: str
    diff_text: str | None = None
    diff_struct: dict[str, Any] | None = None


def make_anchor_key(ev: Event) -> tuple[Any, Any] | tuple[Any, Any, Any, Any, Any | None]:
    """Compute a comparable anchor key for an event.

    Keep aligned with `timewarp.langgraph.anchors.make_anchor_id`.
    Prefers explicit `labels["anchor_id"]` when present; otherwise falls back
    to a tuple of (action_type, actor, namespace, thread_id, prompt_hash?).
    """
    # Prefer explicit anchor_id when present for robust alignment
    if ev.labels and "anchor_id" in ev.labels:
        return ("ANCHOR", ev.labels["anchor_id"])  # compact comparable form
    # Fallback: include action type and actor, plus helpful labels and optional prompt hash
    ns = ev.labels.get("namespace") if ev.labels else None
    tid = ev.labels.get("thread_id") if ev.labels else None
    p_hash = None
    try:
        p_hash = ev.hashes.get("prompt") if ev.hashes else None
    except Exception:
        p_hash = None
    return (ev.action_type, ev.actor, ns, tid, p_hash)


def _first_adapter_version(evs: list[Event]) -> str | None:
    """Return the first non-empty adapter_version observed in model_meta."""
    for e in evs:
        mm = e.model_meta or {}
        v = mm.get("adapter_version") if isinstance(mm, dict) else None
        if isinstance(v, str):
            return v
    return None


def realign_by_anchor(
    evs_a: list[Event], evs_b: list[Event], *, start_a: int, start_b: int, window: int
) -> tuple[int | None, int | None]:
    """Search forward within a window to find matching anchors.

    Returns (match_i, match_j) where match_i is the index in A that matches
    B[start_b]'s anchor, and match_j is the index in B that matches A[start_a]'s
    anchor. Either may be None if not found within the window.
    """
    i = start_a
    j = start_b
    n_a = len(evs_a)
    n_b = len(evs_b)
    ak_a = make_anchor_key(evs_a[i]) if i < n_a else None
    ak_b = make_anchor_key(evs_b[j]) if j < n_b else None
    match_i: int | None = None
    match_j: int | None = None
    if ak_b is not None:
        for di in range(1, min(window, n_a - i)):
            if make_anchor_key(evs_a[i + di]) == ak_b:
                match_i = i + di
                break
    if ak_a is not None:
        for dj in range(1, min(window, n_b - j)):
            if make_anchor_key(evs_b[j + dj]) == ak_a:
                match_j = j + dj
                break
    return (match_i, match_j)


def first_divergence(
    store: LocalStore, run_a: UUID, run_b: UUID, *, window: int = 5
) -> Divergence | None:
    evs_a = store.list_events(run_a)
    evs_b = store.list_events(run_b)

    # Early schema/adapter checks
    if evs_a and evs_b and evs_a[0].schema_version != evs_b[0].schema_version:
        reason = (
            "adapter/schema mismatch: schema_version mismatch: "
            f"A={evs_a[0].schema_version} B={evs_b[0].schema_version}"
        )
        return Divergence(evs_a[0].step, evs_b[0].step, reason=reason)

    av_a = _first_adapter_version(evs_a)
    av_b = _first_adapter_version(evs_b)
    if av_a and av_b and av_a != av_b:
        return Divergence(
            0,
            0,
            reason=f"adapter/schema mismatch: adapter_version mismatch: A={av_a} B={av_b}",
        )

    def out_hash(ev: Event) -> str | None:
        if ev.hashes and "output" in ev.hashes:
            return ev.hashes["output"]
        if ev.hashes and "state" in ev.hashes:
            return ev.hashes["state"]
        return None

    # Two-pointer walk with small-window anchor realignment
    i = 0
    j = 0
    n_a = len(evs_a)
    n_b = len(evs_b)
    WINDOW = max(1, int(window))

    while i < n_a and j < n_b:
        a = evs_a[i]
        b = evs_b[j]
        if make_anchor_key(a) != make_anchor_key(b):
            # Try to realign within a small lookahead window
            match_i, match_j = realign_by_anchor(evs_a, evs_b, start_a=i, start_b=j, window=WINDOW)
            if match_i is not None or match_j is not None:
                # Skip over the shorter gap to re-align; treat as benign reordering
                if match_i is not None and (match_j is None or match_i - i <= match_j - j):
                    i = match_i
                else:
                    j = match_j if match_j is not None else j
                continue
            # No anchor alignment within window: actor/type mismatch is a real divergence
            return Divergence(a.step, b.step, reason="anchor mismatch")

        # Anchors aligned; compare hashes
        if out_hash(a) != out_hash(b):
            # Drill into structural/text diffs if possible
            text = None
            struct = None
            try:
                pa = _load_redacted_output(store, a, b)
                pb = _load_redacted_output(store, b, a)
                # Prefer structural diff when both sides are JSON-like
                struct = _struct_diff(pa, pb)
                if not struct:
                    ta = _to_text(pa)
                    tb = _to_text(pb)
                    text = _text_unified(ta, tb)
            except Exception:
                pass
            return Divergence(
                a.step, b.step, reason="output hash mismatch", diff_text=text, diff_struct=struct
            )
        i += 1
        j += 1

    if len(evs_a) != len(evs_b):
        # One run has extra trailing events; report as anchor mismatch
        # to align with the bisect cause taxonomy.
        longer = evs_a if len(evs_a) > len(evs_b) else evs_b
        last_idx = min(len(evs_a), len(evs_b))
        last = longer[last_idx]
        return Divergence(last.step, last.step, reason="anchor mismatch")

    return None


def _load_output(store: LocalStore, ev: Event) -> Any:
    if ev.output_ref:
        data = store.get_blob(ev.output_ref)
        return from_bytes(data)
    return None


def _load_redacted_output(store: LocalStore, ev: Event, other: Event | None = None) -> Any:
    obj = _load_output(store, ev)
    try:
        marks: dict[str, str] = {}
        if getattr(ev, "privacy_marks", None):
            marks.update(ev.privacy_marks)
        if other is not None and getattr(other, "privacy_marks", None):
            # Union privacy marks from both sides to be conservative
            marks.update(other.privacy_marks)
        if (marks and isinstance(obj, dict | list | str | int | float | bool)) or obj is None:
            # redact expects JSON-serializable structures; non-serializable fall back untouched
            return redact(obj, marks)
    except Exception:
        pass
    return obj


## Use to_text from utils.diffing via imported alias _to_text


def bisect_divergence(
    store: LocalStore, run_a: UUID, run_b: UUID, *, window: int = 5
) -> dict[str, int | str] | None:
    """Find the smallest contiguous mismatching window between two runs.

    - Anchors are aligned using a small lookahead window to skip benign reorders.
    - If adapter/schema mismatch is detected, returns a trivial window at (0,0).
    - If anchors cannot be realigned within the window, returns a single-step
      window at the boundary with cause "anchor mismatch".
    - Otherwise, returns a minimal window where at least one aligned pair differs
      by output/state hash with cause "output hash mismatch".

    Returns a dict with keys: start_a, end_a, start_b, end_b, cause; or None if
    the runs are equivalent by anchors and output/state hashes.
    """
    evs_a = store.list_events(run_a)
    evs_b = store.list_events(run_b)

    # Early schema/adapter checks: treat as immediate incompatibility
    if evs_a and evs_b and evs_a[0].schema_version != evs_b[0].schema_version:
        return {
            "start_a": 0,
            "end_a": 0,
            "start_b": 0,
            "end_b": 0,
            "cause": "adapter/schema mismatch",
        }

    av_a = _first_adapter_version(evs_a)
    av_b = _first_adapter_version(evs_b)
    if av_a and av_b and av_a != av_b:
        return {
            "start_a": 0,
            "end_a": 0,
            "start_b": 0,
            "end_b": 0,
            "cause": "adapter/schema mismatch",
        }

    def out_hash(ev: Event) -> str | None:
        if ev.hashes and "output" in ev.hashes:
            return ev.hashes["output"]
        if ev.hashes and "state" in ev.hashes:
            return ev.hashes["state"]
        return None

    i = 0
    j = 0
    n_a = len(evs_a)
    n_b = len(evs_b)
    WINDOW = max(1, int(window))
    pairs: list[tuple[int, int]] = []

    # Walk with realignment to gather comparable anchor pairs
    while i < n_a and j < n_b:
        a = evs_a[i]
        b = evs_b[j]
        if make_anchor_key(a) == make_anchor_key(b):
            pairs.append((i, j))
            i += 1
            j += 1
            continue
        match_i, match_j = realign_by_anchor(evs_a, evs_b, start_a=i, start_b=j, window=WINDOW)
        if match_i is not None or match_j is not None:
            if match_i is not None and (
                match_j is None or match_i - i <= (match_j - j if match_j is not None else 10**9)
            ):
                i = match_i
            else:
                j = match_j if match_j is not None else j
            continue
        # Could not align anchors within window: anchor mismatch
        return {
            "start_a": evs_a[i].step,
            "end_a": evs_a[i].step,
            "start_b": evs_b[j].step,
            "end_b": evs_b[j].step,
            "cause": "anchor mismatch",
        }

    # Equivalent by alignment if no pairs
    if not pairs:
        return None

    # Predicate over a window [lo, hi] in pair indices: True if any mismatch in window
    def mismatch(lo: int, hi: int) -> bool:
        for k in range(lo, hi + 1):
            ia, jb = pairs[k]
            if out_hash(evs_a[ia]) != out_hash(evs_b[jb]):
                return True
        return False

    # Quick scan: record all mismatching pair indices
    mismatches: list[int] = [
        idx for idx, (ia, jb) in enumerate(pairs) if out_hash(evs_a[ia]) != out_hash(evs_b[jb])
    ]
    if not mismatches:
        # Fully aligned and all output/state hashes match (within aligned region)
        # If lengths differ, surface the trailing delta as an anchor mismatch window
        if len(evs_a) != len(evs_b):
            if len(evs_a) > len(evs_b):
                step_a = evs_a[len(evs_b)].step if len(evs_b) < len(evs_a) else evs_a[-1].step
                step_b = evs_b[-1].step if evs_b else 0
            else:
                step_a = evs_a[-1].step if evs_a else 0
                step_b = evs_b[len(evs_a)].step if len(evs_a) < len(evs_b) else evs_b[-1].step
            return {
                "start_a": step_a,
                "end_a": step_a,
                "start_b": step_b,
                "end_b": step_b,
                "cause": "anchor mismatch",
            }
        return None

    # Minimal mismatching window is the single first mismatching aligned pair
    left = mismatches[0]
    right = left

    ia_l, jb_l = pairs[left]
    ia_r, jb_r = pairs[right]
    return {
        "start_a": evs_a[ia_l].step,
        "end_a": evs_a[ia_r].step,
        "start_b": evs_b[jb_l].step,
        "end_b": evs_b[jb_r].step,
        "cause": "output hash mismatch",
    }
