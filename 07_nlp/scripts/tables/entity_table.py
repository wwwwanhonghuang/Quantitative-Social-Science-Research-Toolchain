#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
entity_table.py
---------------
Aggregate extracted entities into a dataframe:
- Counts per entity type
- Frequency per document
"""

import argparse
import pandas as pd
import json
import os

def create_entity_table(json_path, output_csv):
    with open(json_path, "r", encoding="utf-8") as f:
        entities = json.load(f)

    # Flatten JSON to dataframe
    df = pd.json_normalize(entities)
    if df.empty:
        print("No entities found in JSON.")
        return

    # Example: counts per entity type
    summary = df.groupby("entity").size().reset_index(name="count")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f"✓ Entity summary table saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate extracted entities into table")
    parser.add_argument("--input", required=True, help="Path to extracted entities JSON")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    create_entity_table(args.input, args.output)

if __name__ == "__main__":
    main()
