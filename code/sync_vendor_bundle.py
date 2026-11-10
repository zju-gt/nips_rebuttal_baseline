from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from vendor_bundle import sync_vendor_bundle, verify_vendor_bundle


CODE_DIR = Path(__file__).resolve().parent


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the transferable tau2 vendor bundle")
    parser.add_argument(
        "--source-workspace",
        type=Path,
        default=CODE_DIR.parents[1],
    )
    parser.add_argument("--code-dir", type=Path, default=CODE_DIR)
    parser.add_argument("--openai-test", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    code_dir = args.code_dir.expanduser().resolve()
    openai_test = (
        args.openai_test.expanduser().resolve()
        if args.openai_test is not None
        else code_dir / "openai_test.py"
    )
    manifest_path = sync_vendor_bundle(
        args.source_workspace,
        code_dir,
        openai_test,
    )
    manifest = verify_vendor_bundle(manifest_path.parent)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "file_count": manifest["total_files"],
                "total_bytes": manifest["total_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
