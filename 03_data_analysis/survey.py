#!/usr/bin/env python3
import argparse
import os
import glob
import json
import pandas as pd
import numpy as np
from scipy import stats
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer.factor_analyzer import calculate_kmo as kmo_test
import warnings

warnings.filterwarnings('ignore')

# ----------------------------- 
# Scale Reliability (Cronbach's Alpha)
# ----------------------------- 
def cronbach_alpha(df, items):
    """Calculate Cronbach's Alpha"""
    item_data = df[items].dropna()
    
    if len(item_data) == 0:
        return None
    
    # Number of items
    k = len(items)
    
    # Variance of each item
    item_variances = item_data.var(axis=0, ddof=1)
    
    # Total variance
    total_variance = item_data.sum(axis=1).var(ddof=1)
    
    # Cronbach's alpha
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    
    return alpha

def run_reliability_analysis(df, config):
    """
    Scale reliability analysis (Cronbach's Alpha)
    Config should have:
    - scales: dict of scale_name: [item1, item2, ...]
    - calculate_if_deleted: whether to calculate alpha if item deleted (default: True)
    """
    scales = config["scales"]
    calc_if_deleted = config.get("calculate_if_deleted", True)
    
    results = {
        "test": "Scale Reliability Analysis",
        "scales": {}
    }
    
    for scale_name, items in scales.items():
        # Filter items that exist in dataframe
        existing_items = [item for item in items if item in df.columns]
        
        if len(existing_items) < 2:
            results["scales"][scale_name] = {
                "error": f"Need at least 2 items, found {len(existing_items)}"
            }
            continue
        
        # Calculate overall alpha
        alpha = cronbach_alpha(df, existing_items)
        
        scale_results = {
            "items": existing_items,
            "n_items": len(existing_items),
            "cronbach_alpha": float(alpha) if alpha is not None else None,
            "interpretation": "",
            "item_statistics": {}
        }
        
        # Interpretation
        if alpha is not None:
            if alpha >= 0.9:
                interpretation = "Excellent"
            elif alpha >= 0.8:
                interpretation = "Good"
            elif alpha >= 0.7:
                interpretation = "Acceptable"
            elif alpha >= 0.6:
                interpretation = "Questionable"
            elif alpha >= 0.5:
                interpretation = "Poor"
            else:
                interpretation = "Unacceptable"
            scale_results["interpretation"] = interpretation
        
        # Item-total correlations and alpha if deleted
        item_data = df[existing_items].dropna()
        scale_total = item_data.sum(axis=1)
        
        for item in existing_items:
            # Item-total correlation (corrected)
            item_rest_total = scale_total - item_data[item]
            corr = item_data[item].corr(item_rest_total)
            
            item_stats = {
                "mean": float(item_data[item].mean()),
                "std": float(item_data[item].std()),
                "corrected_item_total_correlation": float(corr) if not pd.isna(corr) else None
            }
            
            # Alpha if deleted
            if calc_if_deleted and len(existing_items) > 2:
                items_without = [i for i in existing_items if i != item]
                alpha_if_deleted = cronbach_alpha(df, items_without)
                item_stats["alpha_if_deleted"] = float(alpha_if_deleted) if alpha_if_deleted is not None else None
            
            scale_results["item_statistics"][item] = item_stats
        
        results["scales"][scale_name] = scale_results
    
    return results

# ----------------------------- 
# Factor Analysis (Exploratory)
# ----------------------------- 
def run_exploratory_factor_analysis(df, config):
    """
    Exploratory Factor Analysis (EFA)
    Config should have:
    - variables: list of variables to factor analyze
    - n_factors: number of factors to extract (optional, will use Kaiser criterion if not specified)
    - rotation: rotation method ('varimax', 'promax', 'oblimin', None)
    - method: extraction method ('minres', 'ml', 'principal')
    """
    variables = config["variables"]
    n_factors = config.get("n_factors", None)
    rotation = config.get("rotation", "varimax")
    method = config.get("method", "minres")
    
    # Prepare data
    data = df[variables].dropna()
    
    if len(data) < 50:
        return {
            "error": "Sample size too small (n < 50) for reliable factor analysis"
        }
    
    results = {
        "test": "Exploratory Factor Analysis",
        "variables": variables,
        "n_observations": len(data),
        "rotation": rotation,
        "method": method
    }
    
    # KMO test (sampling adequacy)
    try:
        kmo_all, kmo_model = calculate_kmo(data)
        results["kmo"] = {
            "overall": float(kmo_model),
            "interpretation": ""
        }
        
        # Interpretation
        if kmo_model >= 0.9:
            results["kmo"]["interpretation"] = "Marvelous"
        elif kmo_model >= 0.8:
            results["kmo"]["interpretation"] = "Meritorious"
        elif kmo_model >= 0.7:
            results["kmo"]["interpretation"] = "Middling"
        elif kmo_model >= 0.6:
            results["kmo"]["interpretation"] = "Mediocre"
        elif kmo_model >= 0.5:
            results["kmo"]["interpretation"] = "Miserable"
        else:
            results["kmo"]["interpretation"] = "Unacceptable"
    except:
        results["kmo"] = {"error": "Could not calculate KMO"}
    
    # Bartlett's test of sphericity
    try:
        chi_square, p_value = calculate_bartlett_sphericity(data)
        results["bartlett"] = {
            "chi_square": float(chi_square),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "interpretation": "Correlations exist (suitable for FA)" if p_value < 0.05 else "No correlations (not suitable for FA)"
        }
    except:
        results["bartlett"] = {"error": "Could not calculate Bartlett's test"}
    
    # Determine number of factors if not specified
    if n_factors is None:
        # Kaiser criterion (eigenvalue > 1)
        fa_test = FactorAnalyzer(n_factors=len(variables), rotation=None, method=method)
        fa_test.fit(data)
        eigenvalues = fa_test.get_eigenvalues()[0]
        n_factors = sum(eigenvalues > 1)
        results["n_factors_kaiser"] = int(n_factors)
        results["eigenvalues"] = [float(e) for e in eigenvalues]
    else:
        results["n_factors_specified"] = n_factors
    
    # Fit factor analysis
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method)
    fa.fit(data)
    
    # Factor loadings
    loadings = fa.loadings_
    results["factor_loadings"] = {}
    
    for i in range(n_factors):
        factor_name = f"Factor_{i+1}"
        results["factor_loadings"][factor_name] = {}
        for j, var in enumerate(variables):
            results["factor_loadings"][factor_name][var] = float(loadings[j, i])
    
    # Variance explained
    variance = fa.get_factor_variance()
    results["variance_explained"] = {
        "proportional_variance": [float(v) for v in variance[1]],
        "cumulative_variance": [float(v) for v in variance[2]],
        "total_variance_explained": float(variance[2][-1])
    }
    
    # Communalities
    communalities = fa.get_communalities()
    results["communalities"] = {var: float(communalities[i]) for i, var in enumerate(variables)}
    
    return results

# ----------------------------- 
# Item Analysis
# ----------------------------- 
def run_item_analysis(df, config):
    """
    Detailed item analysis for scale development
    Config should have:
    - items: list of items to analyze
    - scale_name: name of scale (optional)
    """
    items = config["items"]
    scale_name = config.get("scale_name", "Scale")
    
    # Prepare data
    data = df[items].dropna()
    
    results = {
        "test": "Item Analysis",
        "scale_name": scale_name,
        "n_items": len(items),
        "n_observations": len(data),
        "item_statistics": {},
        "inter_item_correlations": {}
    }
    
    # Descriptive statistics for each item
    for item in items:
        item_stats = {
            "mean": float(data[item].mean()),
            "std": float(data[item].std()),
            "min": float(data[item].min()),
            "max": float(data[item].max()),
            "skewness": float(data[item].skew()),
            "kurtosis": float(data[item].kurtosis())
        }
        
        # Floor/ceiling effects
        if data[item].dtype in ['int64', 'float64']:
            item_range = data[item].max() - data[item].min()
            if item_range > 0:
                floor_pct = (data[item] == data[item].min()).sum() / len(data) * 100
                ceiling_pct = (data[item] == data[item].max()).sum() / len(data) * 100
                
                item_stats["floor_effect_pct"] = float(floor_pct)
                item_stats["ceiling_effect_pct"] = float(ceiling_pct)
                item_stats["floor_ceiling_warning"] = floor_pct > 15 or ceiling_pct > 15
        
        results["item_statistics"][item] = item_stats
    
    # Inter-item correlation matrix
    corr_matrix = data.corr()
    
    # Average inter-item correlation
    n = len(items)
    sum_corr = 0
    count = 0
    
    for i in range(n):
        for j in range(i+1, n):
            sum_corr += corr_matrix.iloc[i, j]
            count += 1
    
    avg_inter_item_corr = sum_corr / count if count > 0 else 0
    
    results["average_inter_item_correlation"] = float(avg_inter_item_corr)
    results["inter_item_correlation_range"] = {
        "min": float(corr_matrix.min().min()),
        "max": float(corr_matrix.max().max())
    }
    
    # Guidelines: 0.15-0.50 is good range
    if avg_inter_item_corr < 0.15:
        results["inter_item_interpretation"] = "Too low - items may not measure same construct"
    elif avg_inter_item_corr > 0.50:
        results["inter_item_interpretation"] = "Too high - items may be redundant"
    else:
        results["inter_item_interpretation"] = "Good - items measure related but distinct aspects"
    
    # Store correlation matrix
    results["correlation_matrix"] = corr_matrix.to_dict()
    
    return results

# ----------------------------- 
# Scale Scoring
# ----------------------------- 
def run_scale_scoring(df, config):
    """
    Create scale scores from items
    Config should have:
    - scales: dict of scale_name: {items: [...], method: 'mean'/'sum', reverse: [...]}
    - output_to_dataset: whether to add scores back to dataset (default: False)
    """
    scales = config["scales"]
    output_to_dataset = config.get("output_to_dataset", False)
    
    results = {
        "test": "Scale Scoring",
        "scales": {}
    }
    
    df_scored = df.copy() if output_to_dataset else None
    
    for scale_name, scale_config in scales.items():
        items = scale_config["items"]
        method = scale_config.get("method", "mean")
        reverse_items = scale_config.get("reverse", [])
        
        # Prepare data
        data = df[items].copy()
        
        # Reverse code if needed
        for rev_item in reverse_items:
            if rev_item in data.columns:
                # Assume Likert scale - reverse by max + min - value
                item_max = data[rev_item].max()
                item_min = data[rev_item].min()
                data[rev_item] = (item_max + item_min) - data[rev_item]
        
        # Calculate scale score
        if method == "mean":
            scale_score = data.mean(axis=1)
        elif method == "sum":
            scale_score = data.sum(axis=1)
        else:
            scale_score = data.mean(axis=1)
        
        scale_results = {
            "items": items,
            "n_items": len(items),
            "method": method,
            "reverse_coded": reverse_items,
            "score_statistics": {
                "mean": float(scale_score.mean()),
                "std": float(scale_score.std()),
                "min": float(scale_score.min()),
                "max": float(scale_score.max()),
                "skewness": float(scale_score.skew()),
                "n_valid": int(scale_score.notna().sum())
            }
        }
        
        results["scales"][scale_name] = scale_results
        
        # Add to dataset if requested
        if output_to_dataset:
            df_scored[f"{scale_name}_score"] = scale_score
    
    # Save scored dataset if requested
    if output_to_dataset and df_scored is not None:
        output_file = config.get("scored_dataset_file", "dataset_scored.csv")
        results["scored_dataset_saved"] = output_file
    
    return results

# ----------------------------- 
# Convergent/Discriminant Validity
# ----------------------------- 
def run_validity_analysis(df, config):
    """
    Test convergent and discriminant validity
    Config should have:
    - focal_scale: items in the focal scale
    - convergent_scales: dict of related scale names and items (should correlate highly)
    - discriminant_scales: dict of unrelated scale names and items (should correlate weakly)
    """
    focal_items = config["focal_scale"]
    convergent = config.get("convergent_scales", {})
    discriminant = config.get("discriminant_scales", {})
    
    # Calculate focal scale score
    focal_score = df[focal_items].mean(axis=1)
    
    results = {
        "test": "Convergent and Discriminant Validity",
        "focal_scale": focal_items,
        "convergent_validity": {},
        "discriminant_validity": {}
    }
    
    # Convergent validity (should be high, r > 0.5)
    for scale_name, items in convergent.items():
        scale_score = df[items].mean(axis=1)
        corr = focal_score.corr(scale_score)
        
        results["convergent_validity"][scale_name] = {
            "correlation": float(corr),
            "adequate": bool(abs(corr) > 0.5),
            "interpretation": "Good convergent validity" if abs(corr) > 0.5 else "Weak convergent validity"
        }
    
    # Discriminant validity (should be low, r < 0.3)
    for scale_name, items in discriminant.items():
        scale_score = df[items].mean(axis=1)
        corr = focal_score.corr(scale_score)
        
        results["discriminant_validity"][scale_name] = {
            "correlation": float(corr),
            "adequate": bool(abs(corr) < 0.3),
            "interpretation": "Good discriminant validity" if abs(corr) < 0.3 else "Weak discriminant validity"
        }
    
    return results

# ----------------------------- 
# Main survey runner
# ----------------------------- 
def run_survey(project_path, select_plans=None):
    processed_dir = os.path.join(project_path, "data", "processed")
    results_dir = os.path.join(project_path, "results", "survey")
    config_dir = os.path.join(project_path, "survey", "config")
    
    os.makedirs(results_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(config_dir, "*.json"))
    if not json_files:
        print(f"No survey analysis configuration JSONs found in {config_dir}")
        return
    
    # Map test types to functions
    test_functions = {
        "reliability": run_reliability_analysis,
        "factor_analysis": run_exploratory_factor_analysis,
        "item_analysis": run_item_analysis,
        "scale_scoring": run_scale_scoring,
        "validity": run_validity_analysis
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
                if test_type == "reliability" and "scales" in result:
                    for scale, data in result["scales"].items():
                        if "cronbach_alpha" in data:
                            print(f"   {scale}: α = {data['cronbach_alpha']:.3f} ({data['interpretation']})")
                
                elif test_type == "factor_analysis":
                    if "kmo" in result:
                        print(f"   KMO: {result['kmo'].get('overall', 'N/A'):.3f} ({result['kmo'].get('interpretation', '')})")
                    if "variance_explained" in result:
                        print(f"   Total variance explained: {result['variance_explained']['total_variance_explained']:.1%}")
                
                elif test_type == "item_analysis":
                    print(f"   Average inter-item correlation: {result['average_inter_item_correlation']:.3f}")
                    print(f"   {result.get('inter_item_interpretation', '')}")
                
                # Save result
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"Saved survey analysis result to {output_path}\n")
                
            except Exception as e:
                print(f"Error running {plan_name}: {str(e)}\n")
                import traceback
                traceback.print_exc()
                all_results[plan_name] = {"error": str(e)}
        
        # Save all results summary
        summary_path = os.path.join(results_dir, f"{dataset_name}_survey_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved survey analysis summary to {summary_path}")

# ----------------------------- 
# CLI interface
# ----------------------------- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run survey analysis for project according to JSON configs."
    )
    parser.add_argument("--project_path", type=str, 
                        help="Path to project folder")
    parser.add_argument("--select-plans", type=str, default=None, 
                        help="Comma-separated list of plan names to run")
    
    args = parser.parse_args()
    
    select_plans = args.select_plans.split(",") if args.select_plans else None
    run_survey(args.project_path, select_plans)