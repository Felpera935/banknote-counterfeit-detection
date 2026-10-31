"""Command-line interface: ``banknote <command>`` (or ``python -m banknote``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import report
from .config import CV_FOLDS, DEFAULT_DATA_DIR, DEFAULT_REPORTS_DIR
from .data import load_banknotes, make_split

COMMANDS = {
    "all": "run every experiment and write reports/summary.json",
    "compare": "cross-validate and test all models",
    "explain": "logistic regression coefficients",
    "depth": "decision-tree depth trade-off and the ATM rules",
    "scaling": "k-NN with and without standardization",
    "review": "bills inside the SVM margin, for human review",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="banknote",
        description="Counterfeit banknote detection: train, compare and explain classifiers.",
    )
    parser.add_argument(
        "command", choices=COMMANDS, nargs="?", default="all", help="; ".join(f"{k}: {v}" for k, v in COMMANDS.items())
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="where the CSV is cached")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="where outputs are written")
    parser.add_argument("--cv", type=int, default=CV_FOLDS, help="number of cross-validation folds")
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading the data")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "all":
        summary = report.run_all(args.data_dir, args.out_dir, k=args.cv, download=not args.no_download)
    else:
        split = make_split(load_banknotes(args.data_dir, download=not args.no_download))
        runners = {
            "compare": lambda: report.run_compare(split, args.out_dir, args.cv),
            "explain": lambda: report.run_explain(split, args.out_dir),
            "depth": lambda: report.run_depth(split, args.out_dir, args.cv),
            "scaling": lambda: report.run_scaling(split, args.out_dir, args.cv),
            "review": lambda: report.run_review(split, args.out_dir),
        }
        summary = runners[args.command]()
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs written to {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
