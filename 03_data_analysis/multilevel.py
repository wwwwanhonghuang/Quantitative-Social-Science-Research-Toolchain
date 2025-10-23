#!/usr/bin/env python3
import argparse
import os
import glob
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings

warnings.filterwarnings('ignore')

# ----------------------------- 
# Random Intercept Model
# ----------------------------- 
def run_random_intercept(df, config):
    """
    Random intercept model (simplest multilevel model)
    Config should have:
    - dependent: outcome variable
    - fixed_effects: list of fixed effect predictors
    - group: grouping/clustering variable (e.g., school_id, country_id)
    - vcov: variance-covariance structure (default: 'independent')
    """
    dependent = config["dependent"]
    fixed_effects = config["fixed_effects"]
    group = config["group"]
    vcov = config.get("vcov", "independent")
    
    # Prepare fixed effects with dummy encoding
    X = pd.get_dummies(df[fixed_effects], drop_first=True).astype(float)
    y = pd.to_numeric(df[dependent], errors='coerce')
    groups = df[group]
    
    # Drop missing values
    data = pd.concat([y, X, groups], axis=1).dropna()
    y = data[dependent]
    X = data[fixed_effects].apply(lambda col: pd.get_dummies(col, drop_first=True) if col.dtype == 'object' else col)
    X = pd.get_dummies(X, drop_first=True).astype(float)
    groups = data[group]
    
    # Fit model
    model = MixedLM(y, X, groups=groups, missing='drop')
    result = model.fit(reml=True)
    
    # Extract results
    output = {
        "model_type": "Random Intercept Model",
        "dependent": dependent,
        "fixed_effects": fixed_effects,
        "group": group,
        "n_observations": int(result.nobs),
        "n_groups": int(result.k_re),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "log_likelihood": float(result.llf),
        "group_variance": float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, 'iloc') else float(result.cov_re[0, 0]),
        "residual_variance": float(result.scale),
        "icc": None,  # Will calculate below
        "fixed_effects_coefficients": {},
        "fixed_effects_pvalues": {},
        "fixed_effects_std_err": {},
        "converged": result.converged
    }
    
    # Calculate ICC (Intraclass Correlation Coefficient)
    group_var = output["group_variance"]
    resid_var = output["residual_variance"]
    output["icc"] = float(group_var / (group_var + resid_var))
    
    # Extract fixed effects
    for i, param in enumerate(result.params.index):
        output["fixed_effects_coefficients"][param] = float(result.params.iloc[i])
        output["fixed_effects_pvalues"][param] = float(result.pvalues.iloc[i])
        output["fixed_effects_std_err"][param] = float(result.bse.iloc[i])
    
    # Summary text
    summary_text = result.summary().as_text()
    
    return output, summary_text

# ----------------------------- 
# Random Slope Model
# ----------------------------- 
def run_random_slope(df, config):
    """
    Random slope model (allows slopes to vary by group)
    Config should have:
    - dependent: outcome variable
    - fixed_effects: list of fixed effect predictors
    - random_slope: variable(s) with random slope
    - group: grouping/clustering variable
    """
    dependent = config["dependent"]
    fixed_effects = config["fixed_effects"]
    random_slope = config["random_slope"]  # Can be string or list
    group = config["group"]
    
    if isinstance(random_slope, str):
        random_slope = [random_slope]
    
    # Prepare data
    X = pd.get_dummies(df[fixed_effects], drop_first=True).astype(float)
    y = pd.to_numeric(df[dependent], errors='coerce')
    groups = df[group]
    
    # Random effects formula
    re_formula = " + ".join(random_slope)
    
    # Drop missing values
    all_vars = [dependent] + fixed_effects + random_slope + [group]
    data = df[all_vars].copy()
    
    # Convert to numeric where needed
    for var in fixed_effects + random_slope:
        if var in data.columns and data[var].dtype == 'object':
            data[var] = pd.Categorical(data[var]).codes
    
    data = data.dropna()
    
    y = pd.to_numeric(data[dependent], errors='coerce')
    X = data[fixed_effects].astype(float)
    groups = data[group]
    
    # Prepare random effects design
    exog_re = data[random_slope].astype(float)
    
    # Fit model
    model = MixedLM(y, X, groups=groups, exog_re=exog_re, missing='drop')
    result = model.fit(reml=True)
    
    # Extract results
    output = {
        "model_type": "Random Slope Model",
        "dependent": dependent,
        "fixed_effects": fixed_effects,
        "random_slope": random_slope,
        "group": group,
        "n_observations": int(result.nobs),
        "n_groups": int(result.k_re),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "log_likelihood": float(result.llf),
        "random_effects_covariance": result.cov_re.tolist() if hasattr(result.cov_re, 'tolist') else result.cov_re,
        "residual_variance": float(result.scale),
        "fixed_effects_coefficients": {},
        "fixed_effects_pvalues": {},
        "fixed_effects_std_err": {},
        "converged": result.converged
    }
    
    # Extract fixed effects
    for i, param in enumerate(result.params.index):
        output["fixed_effects_coefficients"][param] = float(result.params.iloc[i])
        output["fixed_effects_pvalues"][param] = float(result.pvalues.iloc[i])
        output["fixed_effects_std_err"][param] = float(result.bse.iloc[i])
    
    summary_text = result.summary().as_text()
    
    return output, summary_text

# ----------------------------- 
# Cross-Classified Model
# ----------------------------- 
def run_cross_classified(df, config):
    """
    Cross-classified model (multiple grouping factors)
    Config should have:
    - dependent: outcome variable
    - fixed_effects: list of fixed effect predictors
    - groups: list of grouping variables [group1, group2]
    
    Note: Implements using composite grouping strategy
    """
    dependent = config["dependent"]
    fixed_effects = config["fixed_effects"]
    groups = config["groups"]  # List of 2+ grouping variables
    
    if len(groups) < 2:
        raise ValueError("Cross-classified model requires at least 2 grouping variables")
    
    # Create composite group
    df_copy = df.copy()
    df_copy['_composite_group'] = df_copy[groups].astype(str).agg('_'.join, axis=1)
    
    # Prepare data
    X = pd.get_dummies(df_copy[fixed_effects], drop_first=True).astype(float)
    y = pd.to_numeric(df_copy[dependent], errors='coerce')
    composite_groups = df_copy['_composite_group']
    
    # Drop missing values
    data = pd.concat([y, X, composite_groups], axis=1).dropna()
    y = data[dependent]
    X = data[fixed_effects].apply(lambda col: pd.get_dummies(col, drop_first=True) if col.dtype == 'object' else col)
    X = pd.get_dummies(X, drop_first=True).astype(float)
    composite_groups = data['_composite_group']
    
    # Fit model
    model = MixedLM(y, X, groups=composite_groups, missing='drop')
    result = model.fit(reml=True)
    
    output = {
        "model_type": "Cross-Classified Model",
        "dependent": dependent,
        "fixed_effects": fixed_effects,
        "grouping_factors": groups,
        "n_observations": int(result.nobs),
        "n_composite_groups": int(len(composite_groups.unique())),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "log_likelihood": float(result.llf),
        "group_variance": float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, 'iloc') else float(result.cov_re[0, 0]),
        "residual_variance": float(result.scale),
        "fixed_effects_coefficients": {},
        "fixed_effects_pvalues": {},
        "fixed_effects_std_err": {},
        "converged": result.converged
    }
    
    # Extract fixed effects
    for i, param in enumerate(result.params.index):
        output["fixed_effects_coefficients"][param] = float(result.params.iloc[i])
        output["fixed_effects_pvalues"][param] = float(result.pvalues.iloc[i])
        output["fixed_effects_std_err"][param] = float(result.bse.iloc[i])
    
    summary_text = result.summary().as_text()
    
    return output, summary_text

# ----------------------------- 
# Longitudinal/Growth Curve Model
# ----------------------------- 
def run_growth_curve(df, config):
    """
    Growth curve model (repeated measures over time)
    Config should have:
    - dependent: outcome variable
    - time: time variable
    - fixed_effects: other fixed effects (optional)
    - group: subject/individual identifier
    - time_as_random: whether time has random slope (default: True)
    """
    dependent = config["dependent"]
    time = config["time"]
    fixed_effects = config.get("fixed_effects", [])
    group = config["group"]
    time_as_random = config.get("time_as_random", True)
    
    # Prepare fixed effects (include time)
    all_fixed = [time] + fixed_effects
    X = df[all_fixed].copy()
    
    # Convert to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    X = X.astype(float)
    
    y = pd.to_numeric(df[dependent], errors='coerce')
    groups = df[group]
    
    # Drop missing
    data = pd.concat([y, X, groups], axis=1).dropna()
    y = data[dependent]
    X = data[all_fixed]
    groups = data[group]
    
    # Fit model
    if time_as_random:
        # Random intercept and slope for time
        exog_re = data[[time]]
        model = MixedLM(y, X, groups=groups, exog_re=exog_re, missing='drop')
    else:
        # Random intercept only
        model = MixedLM(y, X, groups=groups, missing='drop')
    
    result = model.fit(reml=True)
    
    output = {
        "model_type": "Growth Curve Model",
        "dependent": dependent,
        "time": time,
        "fixed_effects": fixed_effects,
        "group": group,
        "time_as_random_slope": time_as_random,
        "n_observations": int(result.nobs),
        "n_subjects": int(len(groups.unique())),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "log_likelihood": float(result.llf),
        "residual_variance": float(result.scale),
        "fixed_effects_coefficients": {},
        "fixed_effects_pvalues": {},
        "fixed_effects_std_err": {},
        "converged": result.converged
    }
    
    # Extract random effects covariance
    if time_as_random:
        output["random_effects_covariance"] = result.cov_re.tolist() if hasattr(result.cov_re, 'tolist') else result.cov_re
    else:
        output["group_variance"] = float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, 'iloc') else float(result.cov_re[0, 0])
    
    # Extract fixed effects
    for i, param in enumerate(result.params.index):
        output["fixed_effects_coefficients"][param] = float(result.params.iloc[i])
        output["fixed_effects_pvalues"][param] = float(result.pvalues.iloc[i])
        output["fixed_effects_std_err"][param] = float(result.bse.iloc[i])
    
    summary_text = result.summary().as_text()
    
    return output, summary_text

# ----------------------------- 
# Model Comparison
# ----------------------------- 
def run_model_comparison(df, config):
    """
    Compare multiple multilevel models
    Config should have:
    - models: list of model configurations to compare
    Each model config should have same structure as individual models
    """
    models_config = config["models"]
    
    results = []
    comparison = {
        "model_type": "Model Comparison",
        "models": []
    }
    
    # Map model types to functions
    model_functions = {
        "random_intercept": run_random_intercept,
        "random_slope": run_random_slope,
        "growth_curve": run_growth_curve
    }
    
    for model_name, model_config in models_config.items():
        model_type = model_config["type"]
        if model_type in model_functions:
            output, _ = model_functions[model_type](df, model_config)
            
            comparison["models"].append({
                "name": model_name,
                "type": model_type,
                "aic": output["aic"],
                "bic": output["bic"],
                "log_likelihood": output["log_likelihood"],
                "n_observations": output["n_observations"]
            })
    
    # Rank by AIC and BIC
    comparison["best_aic"] = min(comparison["models"], key=lambda x: x["aic"])["name"]
    comparison["best_bic"] = min(comparison["models"], key=lambda x: x["bic"])["name"]
    
    return comparison, json.dumps(comparison, indent=2)

# ----------------------------- 
# Main multilevel runner
# ----------------------------- 
def run_multilevel(project_path, select_plans=None):
    processed_dir = os.path.join(project_path, "data", "processed")
    results_dir = os.path.join(project_path, "results", "multilevel")
    config_dir = os.path.join(project_path, "multilevel", "config")
    
    os.makedirs(results_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(config_dir, "*.json"))
    if not json_files:
        print(f"No multilevel model configuration JSONs found in {config_dir}")
        return
    
    # Map model types to functions
    model_functions = {
        "random_intercept": run_random_intercept,
        "random_slope": run_random_slope,
        "cross_classified": run_cross_classified,
        "growth_curve": run_growth_curve,
        "model_comparison": run_model_comparison
    }
    
    for json_file in json_files:
        print(f'Use configuration {json_file}.')
        dataset_name = os.path.splitext(os.path.basename(json_file))[0]
        
        with open(json_file) as f:
            plans = json.load(f)
        
        dataset_file = os.path.join(processed_dir, f"{dataset_name}_processed.csv")
        if not os.path.exists(dataset_file):
            print(f"Dataset {dataset_name}_processed.csv not found. Skipping.")
            continue
        
        df = pd.read_csv(dataset_file)
        
        all_results = {}
        
        for plan_name, config in plans.items():
            if select_plans and plan_name not in select_plans:
                print(f"Skipping plan {plan_name}")
                continue
            
            model_type = config["type"]
            output_file = config["output_file"]
            output_path = os.path.join(results_dir, output_file)
            
            print(f"Running plan {plan_name} ({model_type}) on dataset {dataset_name} ...")
            
            if model_type not in model_functions:
                print(f"Unknown model type: {model_type}")
                continue
            
            try:
                result, summary_text = model_functions[model_type](df, config)
                all_results[plan_name] = result
                
                # Print key results
                print(f"\n--- Results for {plan_name} ---")
                print(f"Model: {result.get('model_type', 'N/A')}")
                print(f"N observations: {result.get('n_observations', 'N/A')}")
                print(f"AIC: {result.get('aic', 'N/A'):.2f}")
                print(f"BIC: {result.get('bic', 'N/A'):.2f}")
                if 'icc' in result:
                    print(f"ICC: {result['icc']:.4f}")
                print(f"Converged: {result.get('converged', 'N/A')}")
                
                # Save JSON result
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                
                # Save text summary
                summary_path = output_path.replace('.json', '_summary.txt')
                with open(summary_path, "w") as f:
                    f.write(summary_text)
                
                print(f"Saved results to {output_path}")
                print(f"Saved summary to {summary_path}\n")
                
            except Exception as e:
                print(f"Error running {plan_name}: {str(e)}\n")
                import traceback
                traceback.print_exc()
                all_results[plan_name] = {"error": str(e)}
        
        # Save all results summary
        summary_path = os.path.join(results_dir, f"{dataset_name}_multilevel_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved multilevel summary to {summary_path}")

# ----------------------------- 
# CLI interface
# ----------------------------- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multilevel models for project according to JSON configs."
    )
    parser.add_argument("--project_path", type=str, 
                        help="Path to project folder")
    parser.add_argument("--select-plans", type=str, default=None, 
                        help="Comma-separated list of plan names to run")
    
    args = parser.parse_args()
    
    select_plans = args.select_plans.split(",") if args.select_plans else None
    run_multilevel(args.project_path, select_plans)