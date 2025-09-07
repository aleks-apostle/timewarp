from __future__ import annotations

import argparse
from typing import Any, cast
from uuid import UUID

from ...diff import bisect_divergence, first_divergence, make_anchor_key
from ...store import LocalStore
from ..helpers.jsonio import print_json


def _handler(args: argparse.Namespace, store: LocalStore) -> int:
    use_bisect = bool(getattr(args, "use_bisect", False))
    explain = bool(getattr(args, "explain", False))
    if use_bisect:
        b = bisect_divergence(store, UUID(args.run_a), UUID(args.run_b), window=args.window)
        if getattr(args, "as_json", False):
            exit_code = 0
            payload: object
            if b is None:
                payload = {"result": None}
            else:
                payload = cast(dict[str, Any], b)
                try:
                    if hasattr(args, "fail_on_divergence") and bool(args.fail_on_divergence):
                        exit_code = 1
                except Exception:
                    pass
            print_json(payload)
            return exit_code
        if b is None:
            print("No divergence: runs equivalent by step/order/hashes")
            return 0
        print(
            f"Minimal failing window: A[{b['start_a']}:{b['end_a']}], "
            f"B[{b['start_b']}:{b['end_b']}] — cause={b['cause']}"
        )
        if explain:
            try:
                evs_a = store.list_events(UUID(args.run_a))
                evs_b = store.list_events(UUID(args.run_b))
                # Show anchors and hashes at window start
                ea = next((e for e in evs_a if e.step == int(b["start_a"])), None)
                eb = next((e for e in evs_b if e.step == int(b["start_b"])), None)
                if ea and eb:
                    ak_a = make_anchor_key(ea)
                    ak_b = make_anchor_key(eb)
                    ha = (ea.hashes or {}).get("output") or (ea.hashes or {}).get("state")
                    hb = (eb.hashes or {}).get("output") or (eb.hashes or {}).get("state")
                    print("Anchor A:", ak_a)
                    print("Anchor B:", ak_b)
                    if ha or hb:
                        print("Hash A:", (ha or "-")[:32])
                        print("Hash B:", (hb or "-")[:32])
            except Exception:
                pass
        try:
            if hasattr(args, "fail_on_divergence") and bool(args.fail_on_divergence):
                return 1
        except Exception:
            pass
        return 0

    d = first_divergence(store, UUID(args.run_a), UUID(args.run_b), window=args.window)
    if getattr(args, "as_json", False):
        exit_code = 0
        diff_payload: dict[str, object]
        if d is None:
            diff_payload = {"result": None}
        else:
            diff_payload = {
                "step_a": d.step_a,
                "step_b": d.step_b,
                "reason": d.reason,
                "diff_struct": d.diff_struct,
                "diff_text": d.diff_text,
            }
            try:
                if hasattr(args, "fail_on_divergence") and bool(args.fail_on_divergence):
                    exit_code = 1
            except Exception:
                pass
        print_json(diff_payload)
        return exit_code
    if d is None:
        print("No divergence: runs equivalent by step/order/hashes")
        return 0
    print(f"First divergence at A:{d.step_a} B:{d.step_b}: {d.reason}")
    if explain:
        try:
            evs_a = store.list_events(UUID(args.run_a))
            evs_b = store.list_events(UUID(args.run_b))
            ea = next((e for e in evs_a if e.step == d.step_a), None)
            eb = next((e for e in evs_b if e.step == d.step_b), None)
            if ea and eb:
                ak_a = make_anchor_key(ea)
                ak_b = make_anchor_key(eb)
                ha = (ea.hashes or {}).get("output") or (ea.hashes or {}).get("state")
                hb = (eb.hashes or {}).get("output") or (eb.hashes or {}).get("state")
                print("Anchor A:", ak_a)
                print("Anchor B:", ak_b)
                if ha or hb:
                    print("Hash A:", (ha or "-")[:32])
                    print("Hash B:", (hb or "-")[:32])
        except Exception:
            pass
    if d.diff_struct:
        print("STRUCT DIFF:")
        print(d.diff_struct)
    if d.diff_text:
        print("TEXT DIFF:")
        print(d.diff_text)
    try:
        if hasattr(args, "fail_on_divergence") and bool(args.fail_on_divergence):
            return 1
    except Exception:
        pass
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ddf = sub.add_parser("diff")
    ddf.add_argument("run_a")
    ddf.add_argument("run_b")
    ddf.add_argument("--window", type=int, default=5, help="Anchor realignment window (default 5)")
    ddf.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON output")
    ddf.add_argument(
        "--bisect", dest="use_bisect", action="store_true", help="Find minimal failing window"
    )
    ddf.add_argument(
        "--explain",
        dest="explain",
        action="store_true",
        help="Explain anchors and hashes at the divergence",
    )
    ddf.add_argument(
        "--fail-on-divergence",
        dest="fail_on_divergence",
        action="store_true",
        help="Exit with non-zero status when a divergence is found",
    )
    ddf.set_defaults(func=_handler)
