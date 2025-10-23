#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_freq_bar.py
----------------
Plot a bar chart for top-N word frequencies.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_freq_bar(freq_csv, output_path, top_n=20):
    df = pd.read_csv(freq_csv)
    df = df.sort_values("frequency", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(df["word"][::-1], df["frequency"][::-1], color="steelblue")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.title(f"Top {top_n} Words")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✓ Frequency bar chart saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot top-N word frequency bar chart.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    plot_freq_bar(args.input, args.output, args.top_n)

if __name__ == "__main__":
    main()
