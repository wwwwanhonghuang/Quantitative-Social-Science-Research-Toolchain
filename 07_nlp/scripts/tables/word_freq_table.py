#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
word_freq_table.py
-----------------
Convert word frequency CSV into a summary table suitable for social science analysis.
Can filter by frequency, normalize counts, or create relative frequency columns.
"""

import argparse
import pandas as pd
import os

def create_word_freq_table(freq_csv, output_csv, min_freq=1, normalize=False):
    df = pd.read_csv(freq_csv)
    df = df[df["frequency"] >= min_freq].copy()

    if normalize:
        total = df["frequency"].sum()
        df["relative_frequency"] = df["frequency"] / total

    df = df.sort_values("frequency", ascending=False)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✓ Word frequency table saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Generate word frequency table for analysis")
    parser.add_argument("--input", required=True, help="Path to word frequency CSV")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum frequency to include")
    parser.add_argument("--normalize", action="store_true", help="Add relative frequency column")
    args = parser.parse_args()

    create_word_freq_table(args.input, args.output, args.min_freq, args.normalize)

if __name__ == "__main__":
    main()
