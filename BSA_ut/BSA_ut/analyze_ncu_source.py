# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize the SASS CSV exported by profile_b300_ncu.sh."""

import argparse
import csv


def as_int(value):
    return int(float(value)) if value and value != "-" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    with open(args.path, encoding="utf-8", newline="") as report:
        header = next(line for line in report if line.startswith('"Address","Source"'))
        rows = list(csv.DictReader([header, *report]))

    for metric in (
        "L2 Theoretical Sectors Global Excessive",
        "L2 Theoretical Sectors Global",
        "L1 Wavefronts Shared Excessive",
        "L1 Wavefronts Shared",
        "stall_long_sb",
        "stall_wait",
        "stall_mio",
        "Warp Stall Sampling (All Samples)",
        "Instructions Executed",
    ):
        total = sum(as_int(row[metric]) for row in rows)
        print(f"\n{metric}: {total:,}")
        ranked = sorted(rows, key=lambda row: as_int(row[metric]), reverse=True)
        for row in ranked[: args.top]:
            count = as_int(row[metric])
            if count == 0:
                break
            source = row["Source"].strip()
            print(f"  {count:>12,}  {row['Address']:>18s}  {source}")


if __name__ == "__main__":
    main()
