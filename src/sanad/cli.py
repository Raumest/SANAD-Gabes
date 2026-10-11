"""Command-line entry point.

    python -m sanad.cli generate [--config configs/default.yaml]
    python -m sanad.cli run      [--config ...]
    python -m sanad.cli evaluate [--config ...]
    python -m sanad.cli demo
"""

from __future__ import annotations

import argparse
import sys
import time

from sanad.config import load_config


def cmd_generate(cfg: dict) -> None:
    from sanad.sim.generator import generate

    t0 = time.perf_counter()
    out = generate(cfg)
    print(f"[generate] wrote {out} in {time.perf_counter() - t0:.1f}s")


def cmd_run(cfg: dict) -> None:
    from sanad.pipeline import run_pipeline

    t0 = time.perf_counter()
    out = run_pipeline(cfg)
    print(f"[run] wrote {out} in {time.perf_counter() - t0:.1f}s")


def cmd_evaluate(cfg: dict) -> None:
    from sanad.evaluate.report import evaluate_run

    out = evaluate_run(cfg)
    print(f"[evaluate] wrote {out}")


def cmd_demo(cfg: dict) -> None:
    from sanad.demo.scenarios import build_demo

    out = build_demo(cfg)
    print(f"[demo] wrote {out}")


COMMANDS = {"generate": cmd_generate, "run": cmd_run, "evaluate": cmd_evaluate, "demo": cmd_demo}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sanad", description="SANAD-Gabès trust layer")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if cfg.get("data_source", "synthetic") != "synthetic" and args.command == "generate":
        print("data_source=real is a Phase 2 stub; see src/sanad/adapters/.", file=sys.stderr)
        return 2
    COMMANDS[args.command](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
