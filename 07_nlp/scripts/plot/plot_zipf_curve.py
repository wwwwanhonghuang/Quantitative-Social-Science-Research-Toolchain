#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_zipf_curve.py
------------------
Plot Zipf’s law (rank-frequency distribution) for text data.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_zipf(freq_csv, output_path):
    df = pd.read_csv(freq_csv)
    df = df.sort_values("frequency", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    plt.figure(figsize=(8, 6))
    plt.loglog(df["rank"], df["frequency"], marker="o", linestyle="", color="darkred", alpha=0.7)
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    plt.title("Zipf's Law")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✓ Zipf’s law plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot Zipf’s law from word frequency CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plot_zipf(args.input, args.output)

if __name__ == "__main__":
    main()
