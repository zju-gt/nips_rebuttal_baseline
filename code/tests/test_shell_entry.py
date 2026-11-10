from __future__ import annotations

import os
import subprocess
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SHELL_ENTRY = CODE_DIR / "run_tau2_baselines.sh"


def test_shell_entry_resolves_directory_and_forwards_arguments(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "fake-python"
    argv_log = tmp_path / "argv.jsonl"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, os, sys",
                "from pathlib import Path",
                "Path(os.environ['ARGV_LOG']).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')",
                "raise SystemExit(23)",
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update({"PYTHON_BIN": str(fake_python), "ARGV_LOG": str(argv_log)})
    arguments = ["--dry-run", "--methods", "react,awm", "--run-id", "quoted value"]

    result = subprocess.run(
        ["bash", str(SHELL_ENTRY), *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    import json

    assert result.returncode == 23
    assert json.loads(argv_log.read_text(encoding="utf-8")) == [
        str(CODE_DIR / "run_tau2_baselines.py"),
        *arguments,
    ]
