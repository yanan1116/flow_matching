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
    parser.add_argument(
        "--mode",
        choices=("strict", "general"),
        default="strict",
        help="Calculation mode. 'strict' preserves the current 40-line LIBERO suite checks; "
        "'general' averages all parsed task summary success rates.",
    )
    return parser.parse_args()


def read_task_summary_lines(log_file):
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.startswith("task summary")]
    except OSError as exc:
        print(f"Failed to read log file '{log_file}': {exc}", file=sys.stderr)
        return None


def print_suite_counts(suite_to_rates):
    for suite in sorted(suite_to_rates):
        print(f"{suite} valid_lines: {len(suite_to_rates[suite])}")


def run_strict(task_summary_lines):
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

    print_suite_counts(suite_to_rates)

    suite_averages = {}
    for suite in EXPECTED_SUITES:
        suite_averages[suite] = sum(suite_to_rates[suite]) / len(suite_to_rates[suite])
        print(f"{suite}: {suite_averages[suite]:.4f}")

    overall_success_rate = sum(suite_averages[suite] for suite in EXPECTED_SUITES) / len(EXPECTED_SUITES)
    print(f"overall: {overall_success_rate:.4f}")
    return 0


def run_general(task_summary_lines):
    suite_to_rates = defaultdict(list)

    for index, line in enumerate(task_summary_lines, start=1):
        match = TASK_SUMMARY_PATTERN.match(line)
        if not match:
            continue
        suite = match.group("suite")
        suite_to_rates[suite].append(float(match.group("success_rate")))

    if not suite_to_rates:
        print("No task summary lines matched TASK_SUMMARY_PATTERN.", file=sys.stderr)
        return 1

    print_suite_counts(suite_to_rates)

    success_rates = [rate for rates in suite_to_rates.values() for rate in rates]
    overall_success_rate = sum(success_rates) / len(success_rates)
    print(f"overall: {overall_success_rate:.4f}")
    return 0


def main():
    args = parse_args()
    task_summary_lines = read_task_summary_lines(args.log_file)
    if task_summary_lines is None:
        return 1

    if args.mode == "general":
        return run_general(task_summary_lines)

    return run_strict(task_summary_lines)


if __name__ == "__main__":
    raise SystemExit(main())
