#!/usr/bin/env python3
import argparse
import os
import glob
import json
import pandas as pd
import statsmodels.api as sm
import numpy as np

# -----------------------------
# Descriptive stats
# -----------------------------
def run_descriptive(df, config):
    groupby_vars = config.get("groupby", [])
    vars_to_use = config.get("variables", df.columns.tolist())

    # Keep only columns that exist
    existing_vars = [v for v in vars_to_use if v in df.columns]
    existing_groupby = [v for v in groupby_vars if v in df.columns]

    # Ensure groupby columns are 1D
    existing_groupby_series = [df[v].squeeze() if isinstance(df[v], pd.DataFrame) else df[v] for v in existing_groupby]

    if existing_groupby_series:
        desc = df[existing_vars].groupby(existing_groupby_series).describe()
    else:
        desc = df[existing_vars].describe()

    return desc

# -----------------------------
# Regression
# -----------------------------
def run_regression(df, config):
    dependent = config["dependent"]
    independents = config["independents"]

    # Select independent variables and encode categorical
    X = pd.get_dummies(df[independents], drop_first=True).astype(float)
    X = sm.add_constant(X)

    # Dependent variable
    y = pd.to_numeric(df[dependent], errors='coerce')

    # Drop rows with missing values
    data = pd.concat([y, X], axis=1).dropna()
    y = data[dependent]
    X = data.drop(columns=[dependent])

    model = sm.OLS(y, X).fit()
   
    return model.summary().as_text()


# -----------------------------
# Main statistics runner
# -----------------------------
def run_statistics(project_path, select_plans=None):
    processed_dir = os.path.join(project_path, "data", "processed")
    results_dir = os.path.join(project_path, "results", "statistics")
    config_dir = os.path.join(project_path, "statistics", "config")
    os.makedirs(results_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(config_dir, "*.json"))
    if not json_files:
        print(f"No table configuration JSONs found in {config_dir}")
        return

    for json_file in json_files:
        print(f'Use configuration {json_file}.')
        print(os.path.basename(json_file))
        dataset_name = os.path.splitext(os.path.basename(json_file))[0]
        with open(json_file) as f:
            plans = json.load(f)

        dataset_file = os.path.join(processed_dir, f"{dataset_name}_processed.csv")
        if not os.path.exists(dataset_file):
            print(f"Dataset {dataset_name}_processed.csv not found. Skipping.")
            continue

        df = pd.read_csv(dataset_file)

        for plan_name, config in plans.items():
            if select_plans and plan_name not in select_plans:
                print(f"Skipping plan {plan_name}")
                continue

            output_path = os.path.join(results_dir, config["output_file"])
            print(f"Running plan {plan_name} on dataset {dataset_name} ...")

            if config["type"] == "descriptive":
                result = run_descriptive(df, config)
                print(result)
                result.to_csv(output_path)
                print(f"Saved descriptive table to {output_path}")
            elif config["type"] == "regression":
                result = run_regression(df, config)
                with open(output_path, "w") as f:
                    f.write(result)
                print(result)
                print(f"Saved regression summary to {output_path}")

# -----------------------------
# CLI interface
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistics for project according to JSON configs.")
    parser.add_argument("--project_path", type=str, help="Path to project folder")
    parser.add_argument("--select-plans", type=str, default=None,
                        help="Comma-separated list of plan names to run")
    args = parser.parse_args()

    select_plans = args.select_plans.split(",") if args.select_plans else None
    run_statistics(args.project_path, select_plans)
