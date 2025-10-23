#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
classification_table.py
----------------------
Summarize classifier predictions into CSV table for analysis.
"""

import argparse
import pandas as pd
import json
import os

def create_classification_table(class_json, output_csv):
    with open(class_json, "r", encoding="utf-8") as f:
        classes = json.load(f)

    df = pd.json_normalize(classes)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✓ Classification summary table saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Summarize classifier predictions")
    parser.add_argument("--input", required=True, help="Classification JSON path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    create_classification_table(args.input, args.output)

if __name__ == "__main__":
    main()
