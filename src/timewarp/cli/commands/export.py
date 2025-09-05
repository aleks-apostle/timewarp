from __future__ import annotations

import argparse
from uuid import UUID as _UUID

from ...store import LocalStore
from ..helpers.jsonio import print_json


def _handler(args: argparse.Namespace, store: LocalStore) -> int:
    if args.exporter == "langsmith":
        try:
            from ...exporters.langsmith import serialize_run as _serialize_run
        except Exception as exc:  # pragma: no cover - optional dependency path
            print("Export failed: missing dependencies:", exc)
            return 1
        payload = _serialize_run(
            store, _UUID(args.run_id), include_blobs=bool(getattr(args, "include_blobs", False))
        )
        print_json(payload)
        return 0
    print("Unknown exporter:", args.exporter)
    return 1


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    exp = sub.add_parser("export")
    exp_sub = exp.add_subparsers(dest="exporter", required=True)
    exp_ls = exp_sub.add_parser("langsmith")
    exp_ls.add_argument("run_id", help="Run ID to export")
    exp_ls.add_argument(
        "--include-blobs",
        dest="include_blobs",
        action="store_true",
        help="Inline small blobs as JSON where possible",
    )
    exp_ls.set_defaults(func=_handler)
