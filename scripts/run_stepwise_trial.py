from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "newcomb_wildflower_guide" / "experiment_repro"

sys.path.insert(0, str(ROOT / "scripts" / "adapters"))
from openrouter_models import resolve_model_alias  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a JM stepwise trial with concise options.")
    parser.add_argument("--model", default="openrouter/free", help="OpenRouter model id or local alias openrouter/free.")
    parser.add_argument("--sample-limit", default="1", help="Number of sample rows to run.")
    parser.add_argument("--features", default="key_flower_type", help="Comma-separated feature list.")
    parser.add_argument("--workers", default="1", help="Thread worker count.")
    parser.add_argument("--timeout", default="75", help="Seconds before one model call is treated as timed out.")
    parser.add_argument("--mode", choices=["command", "mock"], default="command", help="Model adapter mode.")
    parser.add_argument("--out-file", default="", help="Optional output CSV filename. Default is auto-generated.")
    parser.add_argument("--trial-id", default="", help="Optional trial id. Default is timestamp-based.")
    args = parser.parse_args()

    model = args.model
    if args.mode == "command":
        model = resolve_model_alias(model, require_image=True)

    env = os.environ.copy()
    env["EXPERIMENT_MODEL_MODE"] = args.mode
    env["EXPERIMENT_MODELS"] = model
    env["EXPERIMENT_SAMPLE_LIMIT"] = args.sample_limit
    env["EXPERIMENT_FEATURES"] = args.features
    env["EXPERIMENT_NUM_WORKERS"] = args.workers
    env["EXPERIMENT_MODEL_COMMAND_TIMEOUT"] = args.timeout
    env["OPENROUTER_TIMEOUT"] = args.timeout

    if args.mode == "command":
        env["EXPERIMENT_MODEL_COMMAND"] = "uv run python ../../scripts/adapters/openrouter_command_adapter.py"
    if args.out_file:
        env["EXPERIMENT_OUT_FILE"] = args.out_file
    if args.trial_id:
        env["EXPERIMENT_TRIAL_ID"] = args.trial_id

    print("Running JM stepwise trial")
    print(f"  model: {model}")
    print(f"  sample limit: {args.sample_limit}")
    print(f"  features: {args.features}")
    print(f"  workers: {args.workers}")
    print(f"  timeout: {args.timeout}s per model call")
    print(f"  output: {'auto-generated' if not args.out_file else args.out_file}")
    sys.stdout.flush()

    return subprocess.run(
        ["uv", "run", "python", "run_stepwise_local.py"],
        cwd=EXPERIMENT_DIR,
        env=env,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
