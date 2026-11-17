"""Run the binary validation workflow from the command line."""
import argparse
import json
from pathlib import Path

from cmvs.core import evaluate_predictions, read_csv, validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate binary risks or fit a logistic baseline on a stratified holdout.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--outcome", required=True, help="Binary outcome column")
    parser.add_argument("--event", required=True, help="Exact text label of the event")
    parser.add_argument("--numeric", nargs="+", default=[])
    parser.add_argument("--categorical", nargs="+", default=[])
    parser.add_argument("--probability", help="Existing event-probability column; no model is fitted")
    parser.add_argument("--test-size", type=float, help="Holdout fraction for model fitting (default 0.25)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of evaluation-row resamples")
    parser.add_argument("--output", type=Path, help="Write an audit report instead of printing it")
    args = parser.parse_args()
    try:
        if args.probability is not None and (args.numeric or args.categorical or args.test_size is not None):
            parser.error("--probability cannot be combined with predictors or --test-size")
        data = read_csv(args.csv)
        if args.probability is not None:
            result = evaluate_predictions(data, args.outcome, args.event, args.probability,
                                          seed=args.seed, bootstrap=args.bootstrap)
        else:
            result = validate(data, args.outcome, args.event, args.numeric, args.categorical,
                              test_size=.25 if args.test_size is None else args.test_size,
                              seed=args.seed, bootstrap=args.bootstrap)
        report = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        if args.output is None:
            print(report)
        else:
            if args.output.resolve() == args.csv.resolve():
                parser.error("--output must not replace the input CSV")
            # Exclusive creation also protects earlier reports from accidental overwrite.
            with args.output.open("x", encoding="utf-8") as f:
                f.write(report + "\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
