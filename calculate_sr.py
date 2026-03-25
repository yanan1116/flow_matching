#!/usr/bin/env python3

import argparse
import re
import sys
from collections import defaultdict


EXPECTED_SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
]

TASK_SUMMARY_PATTERN = re.compile(
    r"^task summary.*?suite:\s*(?P<suite>\S+).*?success rate:(?P<success_rate>\d+(?:\.\d+)?)\s+n_test:(?P<n_test>\d+)\s*$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate average success rates for LIBERO suites from a log file."
    )
    parser.add_argument("log_file", help="Path to the log file, e.g. cp-frozen_text_model-1800.log")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        with open(args.log_file, "r", encoding="utf-8") as f:
            task_summary_lines = [line.strip() for line in f if line.startswith("task summary")]
    except OSError as exc:
        print(f"Failed to read log file '{args.log_file}': {exc}", file=sys.stderr)
        return 1

    if len(task_summary_lines) != 40:
        print(
            f"Expected exactly 40 lines starting with 'task summary', but found {len(task_summary_lines)}.",
            file=sys.stderr,
        )
        return 1

    suite_to_rates = defaultdict(list)

    for index, line in enumerate(task_summary_lines, start=1):
        match = TASK_SUMMARY_PATTERN.match(line)
        if not match:
            print(f"Failed to parse task summary line {index}: {line}", file=sys.stderr)
            return 1

        suite = match.group("suite")
        success_rate = float(match.group("success_rate"))
        n_test = int(match.group("n_test"))

        if suite not in EXPECTED_SUITES:
            print(f"Unexpected suite '{suite}' on line {index}.", file=sys.stderr)
            return 1

        if n_test != 50:
            print(f"Expected n_test to be 50 on line {index}, but found {n_test}.", file=sys.stderr)
            return 1

        suite_to_rates[suite].append(success_rate)

    missing_suites = [suite for suite in EXPECTED_SUITES if suite not in suite_to_rates]
    if missing_suites:
        print(f"Missing suites: {', '.join(missing_suites)}.", file=sys.stderr)
        return 1

    for suite in EXPECTED_SUITES:
        count = len(suite_to_rates[suite])
        if count != 10:
            print(f"Expected 10 task summary lines for suite '{suite}', but found {count}.", file=sys.stderr)
            return 1

    suite_averages = {}
    for suite in EXPECTED_SUITES:
        suite_averages[suite] = sum(suite_to_rates[suite]) / len(suite_to_rates[suite])
        print(f"{suite}: {suite_averages[suite]:.4f}")

    overall_success_rate = sum(suite_averages[suite] for suite in EXPECTED_SUITES) / len(EXPECTED_SUITES)
    print(f"overall: {overall_success_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
