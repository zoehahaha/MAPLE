from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import dataset_path, validate_data
from .io import atomic_json
from .replay import run_replay
from .reporting import regenerate, validate_reference


def _repo(value: str | None) -> Path:
    return Path(value).resolve() if value else Path.cwd().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(prog="prep", description="PREP offline replay and reporting")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="run the credential-free CPU demo")
    smoke.add_argument("--repo")
    smoke.add_argument("--output", required=True)
    replay = sub.add_parser("replay", help="run an offline replay from a frozen answer bank")
    inputs = replay.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--workload")
    inputs.add_argument("--dataset", help="bundled dataset ID (see data/datasets.json)")
    replay.add_argument("--repo")
    replay.add_argument("--config", help="defaults to experiments/configs/paper.json")
    replay.add_argument("--limit-rows", type=int)
    replay.add_argument("--limit-queries", type=int)
    replay.add_argument("--output", required=True)
    data = sub.add_parser("check-data", help="check all bundled inputs and oracle answer banks")
    data.add_argument("--repo")
    data.add_argument("--output", help="optional JSON check report")
    report = sub.add_parser("report", help="export paper tables and original figures")
    report.add_argument("--repo")
    report.add_argument("--output", required=True)
    validate = sub.add_parser("validate", help="check printed values, signs, units, and footnotes")
    validate.add_argument("--repo")
    args = parser.parse_args()
    if args.command == "smoke":
        repo = _repo(args.repo)
        result = run_replay(
            repo / "data" / "tiny_demo.json",
            repo / "experiments" / "configs" / "tiny_demo.json",
            Path(args.output),
        )
    elif args.command == "replay":
        repo = _repo(args.repo)
        try:
            workload = dataset_path(repo, args.dataset) if args.dataset else Path(args.workload)
            config = Path(args.config) if args.config else repo / "experiments/configs/paper.json"
            result = run_replay(workload, config, Path(args.output),
                                row_limit=args.limit_rows, query_limit=args.limit_queries)
        except ValueError as error:
            parser.error(str(error))
    elif args.command == "check-data":
        result = validate_data(_repo(args.repo))
        if args.output:
            atomic_json(Path(args.output), result)
    elif args.command == "report":
        result = regenerate(_repo(args.repo), Path(args.output))
    else:
        result = validate_reference(_repo(args.repo))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

