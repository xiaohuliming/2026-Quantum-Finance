from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CONFIGS = [
    "configs/dow30.yaml",
    "configs/nas100.yaml",
    "configs/olps.yaml",
]


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    for config in CONFIGS:
        run([PYTHON, "scripts/run_data_check.py", "--config", config])
        run([PYTHON, "scripts/run_classical_baselines.py", "--config", config])
        run([PYTHON, "scripts/run_opl_baselines.py", "--config", config])
    run([PYTHON, "scripts/run_plain_rl.py", "--config", "configs/dow30.yaml"])
    run([PYTHON, "scripts/summarize_results.py"])


if __name__ == "__main__":
    main()
