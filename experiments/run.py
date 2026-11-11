"""Config-only PREP replay entry point.

Scientific settings are variables loaded from the selected JSON file. This
entry point intentionally contains no paper-specific parameter values.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from prep.replay import run_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--limit-queries", type=int)
    arguments = parser.parse_args()
    run_replay(arguments.workload, arguments.config, arguments.output,
               row_limit=arguments.limit_rows, query_limit=arguments.limit_queries)


if __name__ == "__main__":
    main()

