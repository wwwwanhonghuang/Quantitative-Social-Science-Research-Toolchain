#!/usr/bin/env python3
import pandas as pd
import statsmodels.api as sm
import argparse
import json

#!/usr/bin/env python3
# statistics/regression.py

import pandas as pd
import numpy as np
import argparse
import json
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import all scikit-learn regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.linear_model import SGDRegressor, HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor, VotingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.cross_decomposition import PLSRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class ComprehensiveRegressionAnalysis:
    def __init__(self):
        self.regressors = self.initialize_regressors()
        self.results = []
    
    def initialize_regressors(self):
        """Initialize all scikit-learn regressors with reasonable defaults"""
        return {
            # Linear Models
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'BayesianRidge': BayesianRidge(),
            'SGDRegressor': SGDRegressor(max_iter=1000, tol=1e-3),
            'HuberRegressor': HuberRegressor(),
            'RANSACRegressor': RANSACRegressor(),
            'TheilSenRegressor': TheilSenRegressor(),
            
            # Tree-based Models
            'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=5, random_state=42),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'AdaBoostRegressor': AdaBoostRegressor(n_estimators=50, random_state=42),
            
            # SVM Models
            'SVR': SVR(kernel='rbf', C=1.0),
            'LinearSVR': LinearSVR(random_state=42),
            
            # Neighbors
            'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=5),
            
            # Neural Networks
            'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
            
            # Ensemble Methods
            'BaggingRegressor': BaggingRegressor(n_estimators=10, random_state=42),
            
            # Gaussian Processes
            'GaussianProcessRegressor': GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), random_state=42),
            
            # Partial Least Squares
            'PLSRegression': PLSRegression(n_components=2),
            
            # Kernel Methods
            'KernelRidge': KernelRidge(alpha=1.0),
            
            # Advanced Tree-based (if installed)
            'XGBRegressor': XGBRegressor(n_estimators=100, random_state=42),
            'LGBMRegressor': LGBMRegressor(n_estimators=100, random_state=42),
        }
    
    def evaluate_regressor(self, name, regressor, X_train, X_test, y_train, y_test, cv_folds=5):
        """Evaluate a single regressor using multiple metrics"""
        try:
            # Fit the model
            regressor.fit(X_train, y_train)
            
            # Predictions
            y_pred = regressor.predict(X_test)
            y_pred_train = regressor.predict(X_train)
            
            # Calculate metrics
            test_mse = mean_squared_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            train_r2 = r2_score(y_train, y_pred_train)
            
            # Cross-validation scores
            cv_scores = cross_val_score(regressor, X_train, y_train, cv=cv_folds, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Feature importance (if available)
            feature_importance = self.get_feature_importance(regressor, X_train.shape[1])
            
            return {
                'regressor': name,
                'test_mse': test_mse,
                'test_r2': test_r2,
                'train_r2': train_r2,
                'cv_r2_mean': cv_mean,
                'cv_r2_std': cv_std,
                'overfitting_gap': train_r2 - test_r2,
                'feature_importance': feature_importance,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            return {
                'regressor': name,
                'success': False,
                'error': str(e)
            }
    
    def get_feature_importance(self, regressor, n_features):
        """Extract feature importance if available"""
        try:
            if hasattr(regressor, 'coef_'):
                return regressor.coef_.tolist()
            elif hasattr(regressor, 'feature_importances_'):
                return regressor.feature_importances_.tolist()
            else:
                return [0] * n_features
        except:
            return [0] * n_features
    
    def run_comprehensive_analysis(self, X, y, test_size=0.2, random_state=42):
        """Run analysis with all regressors"""
        print("Running comprehensive regression analysis...")
        print(f"Dataset shape: {X.shape}")
        print(f"Number of regressors: {len(self.regressors)}")
        print("-" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = []
        successful_models = 0
        
        for name, regressor in self.regressors.items():
            print(f"Testing: {name:<25}", end="")
            
            result = self.evaluate_regressor(
                name, regressor, X_train_scaled, X_test_scaled, y_train, y_test
            )
            
            if result['success']:
                successful_models += 1
                print(f"✓ R² = {result['test_r2']:.4f}")
                results.append(result)
            else:
                print(f"✗ Failed")
        
        print("-" * 60)
        print(f"Successful models: {successful_models}/{len(self.regressors)}")
        
        self.results = sorted(results, key=lambda x: x['test_r2'], reverse=True)
        return self.results
    
    def generate_latex_summary(self, feature_names):
        """Generate LaTeX table with results"""
        if not self.results:
            return "No results available"
        
        latex = """\\begin{table}[h]
\\centering
\\caption{Comprehensive Regression Model Comparison}
\\begin{tabular}{lcccccc}
\\toprule
Model & Test R² & Train R² & CV R² & Overfitting Gap & MSE \\\\
\\midrule
"""
        
        for result in self.results[:15]:  # Top 15 models
            if result['success']:
                latex += f"{result['regressor']:<20} & {result['test_r2']:.4f} & {result['train_r2']:.4f} & {result['cv_r2_mean']:.4f} & {result['overfitting_gap']:.4f} & {result['test_mse']:.4f} \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular}
\\label{tab:regression_comparison}
\\end{table}"""
        
        return latex
    
    def generate_feature_importance_table(self, feature_names, top_n=10):
        """Generate LaTeX table for feature importance of top model"""
        if not self.results or not self.results[0]['success']:
            return ""
        
        top_model = self.results[0]
        importance = top_model['feature_importance']
        
        if len(importance) != len(feature_names) or sum(importance) == 0:
            return ""
        
        # Create feature importance pairs and sort
        feature_imp = list(zip(feature_names, importance))
        feature_imp.sort(key=lambda x: abs(x[1]), reverse=True)
        
        latex = """\\begin{table}[h]
\\centering
\\caption{Feature Importance - {top_model['regressor']}}
\\begin{tabular}{lr}
\\toprule
Feature & Importance \\\\
\\midrule
"""
        
        for feature, imp in feature_imp[:top_n]:
            latex += f"{feature} & {imp:.4f} \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return latex

def main():
    parser = argparse.ArgumentParser(description='Comprehensive regression analysis')
    parser.add_argument('--input', required=True, help='Input data file')
    parser.add_argument('--output', required=True, help='Output LaTeX file')
    parser.add_argument('--target', required=True, help='Target variable name')
    parser.add_argument('--config', help='Model configuration file')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != args.target]
    X = df[feature_columns]
    y = df[args.target]
    
    # Remove non-numeric columns for this analysis
    X = X.select_dtypes(include=[np.number])
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Target variable: {args.target}")
    
    # Run comprehensive analysis
    analyzer = ComprehensiveRegressionAnalysis()
    results = analyzer.run_comprehensive_analysis(X, y)
    
    # Generate LaTeX output
    latex_output = analyzer.generate_latex_summary(feature_columns)
    
    # Add feature importance table for top model
    if results and results[0]['success']:
        feature_importance_table = analyzer.generate_feature_importance_table(feature_columns)
        latex_output += "\n\n" + feature_importance_table
    
    # Save results
    with open(args.output, 'w') as f:
        f.write(latex_output)
    
    # Save detailed results as JSON
    results_json = args.output.replace('.tex', '_detailed.json')
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ LaTeX table saved to: {args.output}")
    print(f"✓ Detailed results saved to: {results_json}")
    
    # Print top 5 models
    print("\n🏆 Top 5 Performing Models:")
    for i, result in enumerate(results[:5]):
        if result['success']:
            print(f"{i+1}. {result['regressor']:<25} R² = {result['test_r2']:.4f}")

if __name__ == "__main__":
    main()
    
    
def run_regression_models(data_file, specs_file, output_file):
    """Run multiple regression models and output LaTeX table"""
    df = pd.read_csv(data_file)
    
    with open(specs_file, 'r') as f:
        model_specs = json.load(f)
    
    results = []
    
    for spec_name, spec in model_specs.items():
        # Prepare variables
        X = df[spec['independent_vars']]
        X = sm.add_constant(X)  # Add intercept
        y = df[spec['dependent_var']]
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # Store results
        results.append({
            'specification': spec_name,
            'model': model,
            'nobs': model.nobs
        })
    
    # Generate LaTeX table
    latex_table = generate_regression_table(results, model_specs)
    
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"✓ Regression results saved to {output_file}")

def generate_regression_table(results):
    """Generate LaTeX table from regression results"""
    
    latex = """\\begin{table}[h]
\\centering
\\caption{Regression Results Comparison}
\\begin{threeparttable}
\\begin{tabular}{lcccccc}
\\toprule
 & \\multicolumn{6}{c}{Performance Metrics} \\\\
\\cmidrule(lr){2-7}
Model & Test R² & Train R² & CV R² & CV Std & MSE & Overfit Gap \\\\
\\midrule
"""
    
    for result in results:
        if result['success']:
            latex += f"{result['regressor']:<25} & "
            latex += f"{result['test_r2']:.4f} & "
            latex += f"{result['train_r2']:.4f} & "
            latex += f"{result['cv_r2_mean']:.4f} & "
            latex += f"{result['cv_r2_std']:.4f} & "
            latex += f"{result['test_mse']:.4f} & "
            latex += f"{result['overfitting_gap']:.4f} \\\\\n"
        else:
            latex += f"{result['regressor']:<25} & "
            latex += "\\multicolumn{6}{c}{\\textcolor{red}{Failed: " + result['error'][:30] + "...}} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item \\textit{Note:} CV R² represents mean cross-validation R² score (5-fold). 
Overfit Gap = Train R² - Test R². Models sorted by Test R² (descending).
\\end{tablenotes}
\\end{threeparttable}
\\label{tab:regression_comparison}
\\end{table}"""
    
    return latex