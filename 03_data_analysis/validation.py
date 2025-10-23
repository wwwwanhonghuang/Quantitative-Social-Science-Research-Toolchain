#!/usr/bin/env python3
import argparse
import os
import glob
import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson, jarque_bera, kstest
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# ----------------------------- 
# Outlier Detection
# ----------------------------- 
def run_outlier_detection(df, config):
    """
    Detect outliers using multiple methods
    Config should have:
    - variables: list of variables to check
    - methods: list of methods ['zscore', 'iqr', 'isolation_forest', 'mahalanobis']
    - threshold: threshold for z-score (default: 3)
    """
    variables = config["variables"]
    methods = config.get("methods", ["zscore", "iqr"])
    zscore_threshold = config.get("threshold", 3)
    
    results = {
        "test": "Outlier Detection",
        "variables": variables,
        "methods": methods,
        "outliers_by_method": {},
        "outlier_indices": {}
    }
    
    for method in methods:
        if method == "zscore":
            # Z-score method
            outliers = pd.DataFrame()
            for var in variables:
                if var in df.columns:
                    z_scores = np.abs(stats.zscore(df[var].dropna()))
                    outliers[var] = z_scores > zscore_threshold
            
            results["outliers_by_method"]["zscore"] = {
                "threshold": zscore_threshold,
                "count_per_variable": {var: int(outliers[var].sum()) for var in outliers.columns},
                "total_unique_outliers": int(outliers.any(axis=1).sum())
            }
            results["outlier_indices"]["zscore"] = outliers.any(axis=1)[outliers.any(axis=1)].index.tolist()
        
        elif method == "iqr":
            # IQR method
            outliers = pd.DataFrame()
            for var in variables:
                if var in df.columns:
                    Q1 = df[var].quantile(0.25)
                    Q3 = df[var].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outliers[var] = (df[var] < lower) | (df[var] > upper)
            
            results["outliers_by_method"]["iqr"] = {
                "count_per_variable": {var: int(outliers[var].sum()) for var in outliers.columns},
                "total_unique_outliers": int(outliers.any(axis=1).sum())
            }
            results["outlier_indices"]["iqr"] = outliers.any(axis=1)[outliers.any(axis=1)].index.tolist()
        
        elif method == "mahalanobis":
            # Mahalanobis distance for multivariate outliers
            data = df[variables].dropna()
            if len(data) > 0:
                mean = data.mean()
                cov = data.cov()
                try:
                    inv_cov = np.linalg.inv(cov)
                    mahal_dist = data.apply(lambda x: np.sqrt((x - mean).T @ inv_cov @ (x - mean)), axis=1)
                    
                    # Chi-square critical value
                    chi2_critical = stats.chi2.ppf(0.975, df=len(variables))
                    outliers_mask = mahal_dist > chi2_critical
                    
                    results["outliers_by_method"]["mahalanobis"] = {
                        "critical_value": float(chi2_critical),
                        "total_outliers": int(outliers_mask.sum()),
                        "max_distance": float(mahal_dist.max()),
                        "mean_distance": float(mahal_dist.mean())
                    }
                    results["outlier_indices"]["mahalanobis"] = data.index[outliers_mask].tolist()
                except:
                    results["outliers_by_method"]["mahalanobis"] = {"error": "Could not compute (singular covariance matrix)"}
    
    return results

# ----------------------------- 
# Normality Tests
# ----------------------------- 
def run_normality_tests(df, config):
    """
    Test normality using multiple methods
    Config should have:
    - variables: list of variables to test
    - tests: list of tests ['shapiro', 'anderson', 'jarque_bera', 'ks']
    """
    variables = config["variables"]
    tests = config.get("tests", ["shapiro", "anderson"])
    
    results = {
        "test": "Normality Tests",
        "variables": variables,
        "results_by_variable": {}
    }
    
    for var in variables:
        if var not in df.columns:
            continue
        
        data = df[var].dropna()
        var_results = {
            "n": len(data),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "skewness": float(data.skew()),
            "kurtosis": float(data.kurtosis()),
            "tests": {}
        }
        
        for test_name in tests:
            if test_name == "shapiro" and len(data) >= 3:
                # Shapiro-Wilk test (best for n < 5000)
                if len(data) <= 5000:
                    stat, pvalue = shapiro(data)
                    var_results["tests"]["shapiro"] = {
                        "statistic": float(stat),
                        "p_value": float(pvalue),
                        "normal": bool(pvalue > 0.05)
                    }
                else:
                    var_results["tests"]["shapiro"] = {"note": "Sample too large (n>5000), use other tests"}
            
            elif test_name == "anderson":
                # Anderson-Darling test
                result = anderson(data, dist='norm')
                var_results["tests"]["anderson"] = {
                    "statistic": float(result.statistic),
                    "critical_values": result.critical_values.tolist(),
                    "significance_levels": result.significance_level.tolist(),
                    "normal_at_5pct": bool(result.statistic < result.critical_values[2])
                }
            
            elif test_name == "jarque_bera":
                # Jarque-Bera test
                stat, pvalue = jarque_bera(data)
                var_results["tests"]["jarque_bera"] = {
                    "statistic": float(stat),
                    "p_value": float(pvalue),
                    "normal": bool(pvalue > 0.05)
                }
            
            elif test_name == "ks":
                # Kolmogorov-Smirnov test
                stat, pvalue = kstest(data, 'norm', args=(data.mean(), data.std()))
                var_results["tests"]["ks"] = {
                    "statistic": float(stat),
                    "p_value": float(pvalue),
                    "normal": bool(pvalue > 0.05)
                }
        
        results["results_by_variable"][var] = var_results
    
    return results

# ----------------------------- 
# Homoscedasticity Tests
# ----------------------------- 
def run_homoscedasticity(df, config):
    """
    Test for homoscedasticity (constant variance)
    Config should have:
    - dependent: outcome variable
    - independents: predictor variables
    - tests: ['breusch_pagan', 'white'] (default: both)
    """
    dependent = config["dependent"]
    independents = config["independents"]
    tests = config.get("tests", ["breusch_pagan", "white"])
    
    # Prepare data
    X = pd.get_dummies(df[independents], drop_first=True).astype(float)
    X = sm.add_constant(X)
    y = pd.to_numeric(df[dependent], errors='coerce')
    
    # Drop missing
    data = pd.concat([y, X], axis=1).dropna()
    y = data[dependent]
    X = data.drop(columns=[dependent])
    
    # Fit OLS model
    model = sm.OLS(y, X).fit()
    
    results = {
        "test": "Homoscedasticity Tests",
        "dependent": dependent,
        "independents": independents,
        "n_observations": len(y),
        "tests": {}
    }
    
    if "breusch_pagan" in tests:
        # Breusch-Pagan test
        lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid, X)
        results["tests"]["breusch_pagan"] = {
            "lm_statistic": float(lm),
            "lm_pvalue": float(lm_pvalue),
            "f_statistic": float(fvalue),
            "f_pvalue": float(f_pvalue),
            "homoscedastic": bool(lm_pvalue > 0.05),
            "interpretation": "Homoscedastic (constant variance)" if lm_pvalue > 0.05 else "Heteroscedastic (non-constant variance)"
        }
    
    if "white" in tests:
        # White's test
        lm, lm_pvalue, fvalue, f_pvalue = het_white(model.resid, X)
        results["tests"]["white"] = {
            "lm_statistic": float(lm),
            "lm_pvalue": float(lm_pvalue),
            "f_statistic": float(fvalue),
            "f_pvalue": float(f_pvalue),
            "homoscedastic": bool(lm_pvalue > 0.05),
            "interpretation": "Homoscedastic (constant variance)" if lm_pvalue > 0.05 else "Heteroscedastic (non-constant variance)"
        }
    
    return results

# ----------------------------- 
# Multicollinearity Diagnostics
# ----------------------------- 
def run_multicollinearity(df, config):
    """
    Check for multicollinearity using VIF
    Config should have:
    - variables: list of predictor variables
    - threshold: VIF threshold (default: 10, some use 5)
    """
    variables = config["variables"]
    threshold = config.get("threshold", 10)
    
    # Prepare data
    X = df[variables].copy()
    
    # Convert categorical to dummies
    X = pd.get_dummies(X, drop_first=True).astype(float)
    X = X.dropna()
    
    # Calculate VIF for each variable
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    results = {
        "test": "Multicollinearity Diagnostics (VIF)",
        "variables": variables,
        "threshold": threshold,
        "n_observations": len(X),
        "vif_values": {},
        "problematic_variables": [],
        "interpretation": {}
    }
    
    for idx, row in vif_data.iterrows():
        var = row["Variable"]
        vif = row["VIF"]
        results["vif_values"][var] = float(vif)
        
        if vif > threshold:
            results["problematic_variables"].append(var)
        
        # Interpretation
        if vif < 5:
            interpretation = "No multicollinearity"
        elif vif < 10:
            interpretation = "Moderate multicollinearity"
        else:
            interpretation = "High multicollinearity (problematic)"
        
        results["interpretation"][var] = interpretation
    
    # Correlation matrix
    corr_matrix = X.corr()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    "var1": corr_matrix.columns[i],
                    "var2": corr_matrix.columns[j],
                    "correlation": float(corr_matrix.iloc[i, j])
                })
    
    results["high_correlation_pairs"] = high_corr_pairs
    results["has_multicollinearity"] = len(results["problematic_variables"]) > 0
    
    return results

# ----------------------------- 
# Data Quality Report
# ----------------------------- 
def run_quality_report(df, config):
    """
    Comprehensive data quality report
    Config should have:
    - variables: list of variables to check (optional, defaults to all)
    """
    variables = config.get("variables", df.columns.tolist())
    
    results = {
        "test": "Data Quality Report",
        "dataset_shape": {
            "n_rows": len(df),
            "n_columns": len(df.columns)
        },
        "variables": {}
    }
    
    for var in variables:
        if var not in df.columns:
            continue
        
        var_data = df[var]
        var_results = {
            "dtype": str(var_data.dtype),
            "n_total": len(var_data),
            "n_missing": int(var_data.isna().sum()),
            "pct_missing": float(var_data.isna().sum() / len(var_data) * 100),
            "n_unique": int(var_data.nunique())
        }
        
        # Numeric variables
        if pd.api.types.is_numeric_dtype(var_data):
            var_results["type"] = "numeric"
            non_missing = var_data.dropna()
            if len(non_missing) > 0:
                var_results["stats"] = {
                    "mean": float(non_missing.mean()),
                    "median": float(non_missing.median()),
                    "std": float(non_missing.std()),
                    "min": float(non_missing.min()),
                    "max": float(non_missing.max()),
                    "q25": float(non_missing.quantile(0.25)),
                    "q75": float(non_missing.quantile(0.75)),
                    "skewness": float(non_missing.skew()),
                    "kurtosis": float(non_missing.kurtosis())
                }
                
                # Check for zeros
                var_results["n_zeros"] = int((non_missing == 0).sum())
                
                # Check for negative values
                var_results["n_negative"] = int((non_missing < 0).sum())
        
        # Categorical variables
        else:
            var_results["type"] = "categorical"
            value_counts = var_data.value_counts()
            var_results["top_5_values"] = {
                str(k): int(v) for k, v in value_counts.head(5).items()
            }
            var_results["most_common"] = str(value_counts.index[0]) if len(value_counts) > 0 else None
            var_results["most_common_count"] = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        
        results["variables"][var] = var_results
    
    # Overall summary
    results["summary"] = {
        "total_missing_cells": int(df[variables].isna().sum().sum()),
        "pct_complete": float((1 - df[variables].isna().sum().sum() / (len(df) * len(variables))) * 100),
        "variables_with_missing": [v for v in variables if df[v].isna().sum() > 0]
    }
    
    return results

# ----------------------------- 
# Linearity Check
# ----------------------------- 
def run_linearity_check(df, config):
    """
    Check linearity assumption for regression
    Config should have:
    - dependent: outcome variable
    - independent: predictor variable to check
    """
    dependent = config["dependent"]
    independent = config["independent"]
    
    # Prepare data
    data = df[[dependent, independent]].dropna()
    x = pd.to_numeric(data[independent], errors='coerce')
    y = pd.to_numeric(data[dependent], errors='coerce')
    data = pd.concat([y, x], axis=1).dropna()
    
    # Fit linear model
    X = sm.add_constant(data[independent])
    model = sm.OLS(data[dependent], X).fit()
    
    # Fit quadratic model
    data['x_squared'] = data[independent] ** 2
    X_quad = sm.add_constant(data[[independent, 'x_squared']])
    model_quad = sm.OLS(data[dependent], X_quad).fit()
    
    results = {
        "test": "Linearity Check",
        "dependent": dependent,
        "independent": independent,
        "n_observations": len(data),
        "linear_model": {
            "r_squared": float(model.rsquared),
            "aic": float(model.aic),
            "bic": float(model.bic)
        },
        "quadratic_model": {
            "r_squared": float(model_quad.rsquared),
            "aic": float(model_quad.aic),
            "bic": float(model_quad.bic),
            "quadratic_term_pvalue": float(model_quad.pvalues['x_squared'])
        },
        "interpretation": ""
    }
    
    # Interpretation
    if model_quad.pvalues['x_squared'] < 0.05:
        results["interpretation"] = "Non-linear relationship detected (quadratic term significant). Consider transformation or non-linear model."
    else:
        results["interpretation"] = "Linear relationship appears adequate."
    
    results["better_model"] = "quadratic" if model_quad.aic < model.aic else "linear"
    
    return results

# ----------------------------- 
# Main validation runner
# ----------------------------- 
def run_validation(project_path, select_plans=None):
    processed_dir = os.path.join(project_path, "data", "processed")
    results_dir = os.path.join(project_path, "results", "validation")
    config_dir = os.path.join(project_path, "validation", "config")
    
    os.makedirs(results_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(config_dir, "*.json"))
    if not json_files:
        print(f"No validation configuration JSONs found in {config_dir}")
        return
    
    # Map test types to functions
    test_functions = {
        "outlier_detection": run_outlier_detection,
        "normality": run_normality_tests,
        "homoscedasticity": run_homoscedasticity,
        "multicollinearity": run_multicollinearity,
        "quality_report": run_quality_report,
        "linearity": run_linearity_check
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
            
            test_type = config["type"]
            output_file = config["output_file"]
            output_path = os.path.join(results_dir, output_file)
            
            print(f"Running plan {plan_name} ({test_type}) on dataset {dataset_name} ...")
            
            if test_type not in test_functions:
                print(f"Unknown test type: {test_type}")
                continue
            
            try:
                result = test_functions[test_type](df, config)
                all_results[plan_name] = result
                
                # Print summary
                print(f"\n--- Results for {plan_name} ---")
                print(f"Test: {result.get('test', 'N/A')}")
                
                # Save result
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"Saved validation result to {output_path}\n")
                
            except Exception as e:
                print(f"Error running {plan_name}: {str(e)}\n")
                import traceback
                traceback.print_exc()
                all_results[plan_name] = {"error": str(e)}
        
        # Save all results summary
        summary_path = os.path.join(results_dir, f"{dataset_name}_validation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved validation summary to {summary_path}")

# ----------------------------- 
# CLI interface
# ----------------------------- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run data validation checks for project according to JSON configs."
    )
    parser.add_argument("--project_path", type=str, 
                        help="Path to project folder")
    parser.add_argument("--select-plans", type=str, default=None, 
                        help="Comma-separated list of plan names to run")
    
    args = parser.parse_args()
    
    select_plans = args.select_plans.split(",") if args.select_plans else None
    run_validation(args.project_path, select_plans)