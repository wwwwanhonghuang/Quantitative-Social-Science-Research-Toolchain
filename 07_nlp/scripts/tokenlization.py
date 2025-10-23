#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tokenization.py
---------------
Low-level tokenizer for the NLP toolchain.
Outputs:
  - tokenized text (.txt)
  - word frequency statistics (.csv)

Example usage:
  python tokenization.py \
      --input data/raw/sample.txt \
      --output-tokenized data/processed/sample_tokenized.txt \
      --output-freq data/processed/sample_freq.csv \
      --lang en
"""

import argparse
import os
import re
import pandas as pd
from collections import Counter
import spacy

# -----------------------------
# Utility functions
# -----------------------------

def clean_text(text: str) -> str:
    """Basic normalization: remove redundant spaces, control chars, etc."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: str, lang: str = "en"):
    """Tokenize text using spaCy (supports multiple languages)."""
    try:
        nlp = spacy.load(f"{lang}_core_web_sm")
    except OSError:
        raise ValueError(f"Language model '{lang}_core_web_sm' not installed. Run: python -m spacy download {lang}_core_web_sm")
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    return tokens

def write_outputs(tokens, txt_out: str, csv_out: str):
    """Write tokenized text and frequency table to disk."""
    os.makedirs(os.path.dirname(txt_out), exist_ok=True)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    # Tokenized text
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(" ".join(tokens))

    # Word frequencies
    freq = Counter(tokens)
    df = pd.DataFrame(freq.items(), columns=["word", "frequency"])
    df = df.sort_values(by="frequency", ascending=False)
    df.to_csv(csv_out, index=False, encoding="utf-8")

    print(f"✓ Tokenized text saved to: {txt_out}")
    print(f"✓ Word frequency CSV saved to: {csv_out}")

# -----------------------------
# Main CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Tokenize text and compute word frequencies.")
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--output-tokenized", required=True, help="Output path for tokenized text (.txt)")
    parser.add_argument("--output-freq", required=True, help="Output path for word frequency table (.csv)")
    parser.add_argument("--lang", default="en", help="Language code (e.g., en, ja, zh)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_text(text)
    tokens = tokenize(text, args.lang)
    write_outputs(tokens, args.output_tokenized, args.output_freq)

if __name__ == "__main__":
    main()
