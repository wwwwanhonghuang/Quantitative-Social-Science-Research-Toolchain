#!/usr/bin/env python3
import pandas as pd
import argparse
import numpy as np

def generate_descriptive_stats(data_file, output_file):
    """Generate descriptive statistics table in LaTeX format"""
    df = pd.read_csv(data_file)
    
    # Select numeric columns for descriptive stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats = df[numeric_cols].describe().T
    stats['count'] = stats['count'].astype(int)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(stats)
    
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"✓ Descriptive statistics saved to {output_file}")

def generate_latex_table(stats_df):
    """Convert DataFrame to LaTeX table"""
    latex = """\\begin{table}[h]
\\centering
\\caption{Descriptive Statistics}
\\begin{tabular}{lrrrrrr}
\\toprule
Variable & N & Mean & Std. Dev. & Min & Median & Max \\\\
\\midrule
"""
    
    for var, row in stats_df.iterrows():
        latex += f"{var} & {row['count']} & {row['mean']:.3f} & {row['std']:.3f} & {row['min']:.3f} & {row['50%']:.3f} & {row['max']:.3f} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    generate_descriptive_stats(args.input, args.output)