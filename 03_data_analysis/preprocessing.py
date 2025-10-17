#!/usr/bin/env python3
import argparse
import json
import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder

# -----------------------------
# Preprocessing functions
# -----------------------------
def handle_missing(df, schema):
    for var, info in schema["variables"].items():
        if "missing" not in info:
            continue
        if info["missing"] == "mean":
            df[var].fillna(df[var].mean(), inplace=True)
        elif info["missing"] == "median":
            df[var].fillna(df[var].median(), inplace=True)
        elif info["missing"] == "most_frequent":
            df[var].fillna(df[var].mode()[0], inplace=True)
        elif info["missing"] == "drop":
            df.dropna(subset=[var], inplace=True)
    return df

def encode_categorical(df, schema):
    for var, info in schema["variables"].items():
        if info["type"] == "nominal":
            enc = OneHotEncoder(sparse=False, drop='first')
            encoded = enc.fit_transform(df[[var]])
            cols = [f"{var}_{cat}" for cat in enc.categories_[0][1:]]
            df = df.drop(columns=[var])
            df[cols] = encoded
        elif info["type"] == "ordinal":
            order = info["order"]
            enc = OrdinalEncoder(categories=[order])
            df[var] = enc.fit_transform(df[[var]])
    return df

def scale_numeric(df, schema):
    for var, info in schema["variables"].items():
        if info["type"] in ["interval", "ratio"]:
            if info.get("scaling") == "standard":
                df[var] = StandardScaler().fit_transform(df[[var]])
            elif info.get("scaling") == "minmax":
                df[var] = MinMaxScaler().fit_transform(df[[var]])
    return df

def extract_text_features(df, schema):
    for var, info in schema["variables"].items():
        if info["type"] == "descriptive":
            df[var + "_len"] = df[var].astype(str).apply(len)
    return df

# -----------------------------
# Main preprocessing
# -----------------------------
def preprocess_project(project_path, select_datasets=None):
    raw_dir = os.path.join(project_path, "data", "raw")
    processed_dir = os.path.join(project_path, "data", "processed")
    config_dir = os.path.join(project_path, "data", "config")
    os.makedirs(processed_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
        return

    for csv_file in csv_files:
        dataset_name = os.path.splitext(os.path.basename(csv_file))[0]

        # Skip if not in select_datasets
        if select_datasets and dataset_name not in select_datasets:
            print(f"Skipping {dataset_name}: not in selected datasets")
            continue

        schema_file = os.path.join(config_dir, f"{dataset_name}_schema.json")
        pipeline_file = os.path.join(config_dir, f"{dataset_name}_pipeline.json")

        if not os.path.exists(schema_file) or not os.path.exists(pipeline_file):
            print(f"Skipping {dataset_name}: missing schema or pipeline JSON.")
            continue

        # Load configs
        with open(schema_file) as f:
            schema = json.load(f)
        with open(pipeline_file) as f:
            pipeline = json.load(f)

        # Load dataset
        df = pd.read_csv(csv_file)
        print(f"Processing dataset: {dataset_name} ...")

        # Run pipeline
        for step in pipeline["steps"]:
            if step == "handle_missing":
                df = handle_missing(df, schema)
            elif step == "encode_categorical":
                df = encode_categorical(df, schema)
            elif step == "scale_numeric":
                df = scale_numeric(df, schema)
            elif step == "extract_text_features":
                df = extract_text_features(df, schema)
            elif step == "save_processed":
                output_file = os.path.join(processed_dir, f"{dataset_name}_processed.csv")
                df.to_csv(output_file, index=False)
                print(f"Saved processed dataset to {output_file}")

# -----------------------------
# CLI interface
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess all datasets in a project using per-dataset JSON configs.")
    parser.add_argument("project_path", type=str, help="Path to the project folder")
    parser.add_argument("--select-dataset", type=str, default=None,
                        help="Comma-separated list of datasets to preprocess (without .csv extension)")
    args = parser.parse_args()

    select_datasets = args.select_dataset.split(",") if args.select_dataset else None
    preprocess_project(args.project_path, select_datasets)
