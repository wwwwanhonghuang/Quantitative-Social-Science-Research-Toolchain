#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_wordcloud.py
-----------------
Generate a word cloud from word frequency CSV.

Example:
  python plot_wordcloud.py \
      --input results/statistics/interview_word_freq.csv \
      --output results/figures/interview_wordcloud.png \
      --max-words 200
"""

import argparse
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

def generate_wordcloud(freq_csv, output_path, max_words=200, width=1200, height=800):
    """Generate and save a word cloud image."""
    df = pd.read_csv(freq_csv)
    if "word" not in df.columns or "frequency" not in df.columns:
        raise ValueError("CSV must have 'word' and 'frequency' columns.")

    # Convert to dict
    freq_dict = dict(zip(df["word"], df["frequency"]))

    # Generate word cloud
    wc = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        background_color="white",
        colormap="viridis"
    ).generate_from_frequencies(freq_dict)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wc.to_file(output_path)
    print(f"✓ Word cloud saved to {output_path}")

    # Optionally show inline (if running interactively)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate a word cloud from word frequency CSV.")
    parser.add_argument("--input", required=True, help="Path to word frequency CSV file.")
    parser.add_argument("--output", required=True, help="Output path for word cloud image (e.g., .png)")
    parser.add_argument("--max-words", type=int, default=200, help="Maximum number of words to display")
    parser.add_argument("--width", type=int, default=1200, help="Width of image")
    parser.add_argument("--height", type=int, default=800, help="Height of image")
    args = parser.parse_args()

    generate_wordcloud(args.input, args.output, args.max_words, args.width, args.height)

if __name__ == "__main__":
    main()
