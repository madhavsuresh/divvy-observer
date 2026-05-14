from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_probability_saturation.py"
    spec = importlib.util.spec_from_file_location("_divvy_probability_saturation_script", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load diagnostic script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    return int(_load_script_module().main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
