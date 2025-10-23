#!/usr/bin/env python3
import argparse
import os
import glob
import json
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import mcnemar
import warnings

warnings.filterwarnings('ignore')

# ----------------------------- 
# T-test (independent samples)
# ----------------------------- 
def run_ttest_ind(df, config):
    """
    Independent samples t-test
    Config should have:
    - variable: continuous variable to test
    - group: grouping variable (must have exactly 2 groups)
    - alternative: 'two-sided', 'less', or 'greater' (default: 'two-sided')
    - equal_var: boolean for equal variance assumption (default: True)
    """
    variable = config["variable"]
    group = config["group"]
    alternative = config.get("alternative", "two-sided")
    equal_var = config.get("equal_var", True)
    
    groups = df[group].unique()
    if len(groups) != 2:
        return {"error": f"Group variable must have exactly 2 groups, found {len(groups)}"}
    
    group1_data = df[df[group] == groups[0]][variable].dropna()
    group2_data = df[df[group] == groups[1]][variable].dropna()
    
    statistic, pvalue = stats.ttest_ind(group1_data, group2_data, 
                                         alternative=alternative, 
                                         equal_var=equal_var)
    
    result = {
        "test": "Independent Samples T-Test",
        "variable": variable,
        "group": group,
        "group1": str(groups[0]),
        "group2": str(groups[1]),
        "n1": len(group1_data),
        "n2": len(group2_data),
        "mean1": float(group1_data.mean()),
        "mean2": float(group2_data.mean()),
        "std1": float(group1_data.std()),
        "std2": float(group2_data.std()),
        "statistic": float(statistic),
        "p_value": float(pvalue),
        "alternative": alternative,
        "equal_var": equal_var
    }
    
    return result

# ----------------------------- 
# Paired t-test
# ----------------------------- 
def run_ttest_paired(df, config):
    """
    Paired samples t-test
    Config should have:
    - variable1: first variable
    - variable2: second variable
    - alternative: 'two-sided', 'less', or 'greater' (default: 'two-sided')
    """
    variable1 = config["variable1"]
    variable2 = config["variable2"]
    alternative = config.get("alternative", "two-sided")
    
    data = df[[variable1, variable2]].dropna()
    
    statistic, pvalue = stats.ttest_rel(data[variable1], data[variable2], 
                                         alternative=alternative)
    
    result = {
        "test": "Paired Samples T-Test",
        "variable1": variable1,
        "variable2": variable2,
        "n": len(data),
        "mean1": float(data[variable1].mean()),
        "mean2": float(data[variable2].mean()),
        "mean_diff": float((data[variable1] - data[variable2]).mean()),
        "std_diff": float((data[variable1] - data[variable2]).std()),
        "statistic": float(statistic),
        "p_value": float(pvalue),
        "alternative": alternative
    }
    
    return result

# ----------------------------- 
# ANOVA (one-way)
# ----------------------------- 
def run_anova(df, config):
    """
    One-way ANOVA
    Config should have:
    - variable: continuous variable to test
    - group: grouping variable (can have 2+ groups)
    """
    variable = config["variable"]
    group = config["group"]
    
    groups = df[group].unique()
    group_data = [df[df[group] == g][variable].dropna() for g in groups]
    
    statistic, pvalue = stats.f_oneway(*group_data)
    
    result = {
        "test": "One-Way ANOVA",
        "variable": variable,
        "group": group,
        "n_groups": len(groups),
        "groups": [str(g) for g in groups],
        "n_per_group": [len(gd) for gd in group_data],
        "mean_per_group": [float(gd.mean()) for gd in group_data],
        "std_per_group": [float(gd.std()) for gd in group_data],
        "f_statistic": float(statistic),
        "p_value": float(pvalue)
    }
    
    return result

# ----------------------------- 
# Chi-square test
# ----------------------------- 
def run_chi2(df, config):
    """
    Chi-square test of independence
    Config should have:
    - variable1: first categorical variable
    - variable2: second categorical variable
    """
    variable1 = config["variable1"]
    variable2 = config["variable2"]
    
    contingency_table = pd.crosstab(df[variable1], df[variable2])
    
    chi2, pvalue, dof, expected = stats.chi2_contingency(contingency_table)
    
    result = {
        "test": "Chi-Square Test of Independence",
        "variable1": variable1,
        "variable2": variable2,
        "chi2_statistic": float(chi2),
        "p_value": float(pvalue),
        "degrees_of_freedom": int(dof),
        "contingency_table": contingency_table.to_dict()
    }
    
    return result

# ----------------------------- 
# Mann-Whitney U test (non-parametric)
# ----------------------------- 
def run_mannwhitney(df, config):
    """
    Mann-Whitney U test (non-parametric alternative to t-test)
    Config should have:
    - variable: variable to test
    - group: grouping variable (must have exactly 2 groups)
    - alternative: 'two-sided', 'less', or 'greater' (default: 'two-sided')
    """
    variable = config["variable"]
    group = config["group"]
    alternative = config.get("alternative", "two-sided")
    
    groups = df[group].unique()
    if len(groups) != 2:
        return {"error": f"Group variable must have exactly 2 groups, found {len(groups)}"}
    
    group1_data = df[df[group] == groups[0]][variable].dropna()
    group2_data = df[df[group] == groups[1]][variable].dropna()
    
    statistic, pvalue = stats.mannwhitneyu(group1_data, group2_data, 
                                            alternative=alternative)
    
    result = {
        "test": "Mann-Whitney U Test",
        "variable": variable,
        "group": group,
        "group1": str(groups[0]),
        "group2": str(groups[1]),
        "n1": len(group1_data),
        "n2": len(group2_data),
        "median1": float(group1_data.median()),
        "median2": float(group2_data.median()),
        "u_statistic": float(statistic),
        "p_value": float(pvalue),
        "alternative": alternative
    }
    
    return result

# ----------------------------- 
# Kruskal-Wallis test (non-parametric ANOVA)
# ----------------------------- 
def run_kruskal(df, config):
    """
    Kruskal-Wallis H test (non-parametric alternative to ANOVA)
    Config should have:
    - variable: variable to test
    - group: grouping variable (can have 2+ groups)
    """
    variable = config["variable"]
    group = config["group"]
    
    groups = df[group].unique()
    group_data = [df[df[group] == g][variable].dropna() for g in groups]
    
    statistic, pvalue = stats.kruskal(*group_data)
    
    result = {
        "test": "Kruskal-Wallis H Test",
        "variable": variable,
        "group": group,
        "n_groups": len(groups),
        "groups": [str(g) for g in groups],
        "n_per_group": [len(gd) for gd in group_data],
        "median_per_group": [float(gd.median()) for gd in group_data],
        "h_statistic": float(statistic),
        "p_value": float(pvalue)
    }
    
    return result

# ----------------------------- 
# Correlation test
# ----------------------------- 
def run_correlation(df, config):
    """
    Correlation test (Pearson or Spearman)
    Config should have:
    - variable1: first variable
    - variable2: second variable
    - method: 'pearson' or 'spearman' (default: 'pearson')
    """
    variable1 = config["variable1"]
    variable2 = config["variable2"]
    method = config.get("method", "pearson")
    
    data = df[[variable1, variable2]].dropna()
    
    if method == "pearson":
        corr, pvalue = stats.pearsonr(data[variable1], data[variable2])
        test_name = "Pearson Correlation Test"
    elif method == "spearman":
        corr, pvalue = stats.spearmanr(data[variable1], data[variable2])
        test_name = "Spearman Correlation Test"
    else:
        return {"error": f"Unknown correlation method: {method}"}
    
    result = {
        "test": test_name,
        "variable1": variable1,
        "variable2": variable2,
        "n": len(data),
        "correlation": float(corr),
        "p_value": float(pvalue),
        "method": method
    }
    
    return result

# ----------------------------- 
# Main hypothesis runner
# ----------------------------- 
def run_hypothesis(project_path, select_plans=None):
    processed_dir = os.path.join(project_path, "data", "processed")
    results_dir = os.path.join(project_path, "results", "hypothesis")
    config_dir = os.path.join(project_path, "hypothesis", "config")
    
    os.makedirs(results_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(config_dir, "*.json"))
    if not json_files:
        print(f"No hypothesis test configuration JSONs found in {config_dir}")
        return
    
    # Map test types to functions
    test_functions = {
        "ttest_ind": run_ttest_ind,
        "ttest_paired": run_ttest_paired,
        "anova": run_anova,
        "chi2": run_chi2,
        "mannwhitney": run_mannwhitney,
        "kruskal": run_kruskal,
        "correlation": run_correlation
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
                
                # Print result
                print(f"\n--- Results for {plan_name} ---")
                for key, value in result.items():
                    if key != "contingency_table":
                        print(f"{key}: {value}")
                
                # Save individual result
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"Saved hypothesis test result to {output_path}\n")
                
            except Exception as e:
                print(f"Error running {plan_name}: {str(e)}\n")
                all_results[plan_name] = {"error": str(e)}
        
        # Save all results in one file
        summary_path = os.path.join(results_dir, f"{dataset_name}_hypothesis_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved hypothesis summary to {summary_path}")

# ----------------------------- 
# CLI interface
# ----------------------------- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hypothesis tests for project according to JSON configs."
    )
    parser.add_argument("--project_path", type=str, 
                        help="Path to project folder")
    parser.add_argument("--select-plans", type=str, default=None, 
                        help="Comma-separated list of plan names to run")
    
    args = parser.parse_args()
    
    select_plans = args.select_plans.split(",") if args.select_plans else None
    run_hypothesis(args.project_path, select_plans)