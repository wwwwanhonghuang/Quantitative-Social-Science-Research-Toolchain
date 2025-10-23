#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
topic_table.py
--------------
Convert BERTopic output into table:
- Document-topic assignments
- Topic keywords summary
"""

import argparse
import pandas as pd
import json
import os

def create_topic_table(topic_json, output_csv):
    with open(topic_json, "r", encoding="utf-8") as f:
        topics = json.load(f)

    # Example: topics should contain a list of dicts: {"topic": 0, "keywords": ["a","b"], "docs": [...]}
    rows = []
    for t in topics:
        topic_id = t["topic"]
        keywords = ", ".join(t.get("keywords", []))
        doc_count = len(t.get("docs", []))
        rows.append({"topic": topic_id, "keywords": keywords, "doc_count": doc_count})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✓ Topic summary table saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Summarize topic modeling output to table")
    parser.add_argument("--input", required=True, help="Topic JSON path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    create_topic_table(args.input, args.output)

if __name__ == "__main__":
    main()
