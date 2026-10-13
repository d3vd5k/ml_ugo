"""Descriptive, protocol-compatible summaries; setup checks are excluded by default."""
import argparse
import json

from benchmarks.results import paired_summary, read_records, summarize_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="?", default="results/benchmark_runs.csv")
    parser.add_argument("--run-id", action="append", help="Select invocations explicitly; duplicate cells are refused")
    parser.add_argument("--metric", default="average_precision")
    parser.add_argument("--reference", help="Declared variant ID for per-split/seed paired differences")
    parser.add_argument("--include-smoke", action="store_true", help="Inspect setup output, never study evidence")
    args = parser.parse_args(argv)
    rows = read_records(args.records)
    if args.run_id:
        rows = [row for row in rows if row["run_id"] in args.run_id]
    output = {"interpretation": "descriptive only; folds are not independent datasets",
              "summaries": summarize_records(rows, args.metric, args.include_smoke)}
    if args.reference:
        output["paired_differences"] = paired_summary(rows, args.reference, args.metric, args.include_smoke)
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    main()
