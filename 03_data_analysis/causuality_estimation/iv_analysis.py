"""
Instrumental Variables (IV) Estimation for Causal Inference
Implements 2SLS and related diagnostic tests
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple, Dict
import warnings


class IVEstimator:
    """
    Instrumental Variables estimator using Two-Stage Least Squares (2SLS).
    
    This class implements IV estimation to identify causal effects in the presence
    of endogeneity, confounding, or measurement error.
    
    Attributes:
        beta_: Estimated coefficients (after fitting)
        se_: Standard errors of coefficients
        first_stage_stats_: Dictionary of first-stage diagnostics
        fitted_: Boolean indicating if model has been fitted
    """
    
    def __init__(self, fit_intercept: bool = True):
        """
        Initialize the IV estimator.
        
        Parameters:
        -----------
        fit_intercept : bool, default=True
            Whether to include an intercept term
        """
        self.fit_intercept = fit_intercept
        self.fitted_ = False
        self.beta_ = None
        self.se_ = None
        self.first_stage_stats_ = None
        
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray, 
            Z: np.ndarray,
            W: Optional[np.ndarray] = None) -> 'IVEstimator':
        """
        Fit the IV model using Two-Stage Least Squares.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_endogenous)
            Endogenous variables (potentially correlated with error term)
        y : np.ndarray, shape (n_samples,)
            Outcome variable
        Z : np.ndarray, shape (n_samples, n_instruments)
            Instrumental variables (exogenous, correlated with X)
        W : np.ndarray, shape (n_samples, n_exogenous), optional
            Exogenous control variables (included instruments)
            
        Returns:
        --------
        self : IVEstimator
            Fitted estimator
        """
        # Input validation
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        Z = np.asarray(Z)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
            
        n_samples = X.shape[0]
        
        # Check dimensions
        if y.shape[0] != n_samples or Z.shape[0] != n_samples:
            raise ValueError("X, y, and Z must have the same number of samples")
        
        # Check instrument relevance condition
        if Z.shape[1] < X.shape[1]:
            raise ValueError("Number of instruments must be >= number of endogenous variables")
            
        # Add intercept if requested
        if self.fit_intercept:
            X = np.column_stack([np.ones(n_samples), X])
            Z = np.column_stack([np.ones(n_samples), Z])
            if W is not None:
                W = np.column_stack([np.ones(n_samples), W])
        
        # Combine exogenous controls with instruments
        if W is not None:
            W = np.asarray(W)
            if W.ndim == 1:
                W = W.reshape(-1, 1)
            Z_full = np.column_stack([Z, W])
            X_exog = W
        else:
            Z_full = Z
            X_exog = None
            
        # STAGE 1: Regress X on Z (first stage)
        # X_hat = Z * pi
        Z_Z_inv = np.linalg.inv(Z_full.T @ Z_full)
        pi = Z_Z_inv @ Z_full.T @ X
        X_hat = Z_full @ pi
        
        # First stage diagnostics
        self.first_stage_stats_ = self._first_stage_diagnostics(X, Z_full, pi, n_samples)
        
        # STAGE 2: Regress y on X_hat
        # y = X_hat * beta + error
        X_hat_X_hat_inv = np.linalg.inv(X_hat.T @ X_hat)
        self.beta_ = X_hat_X_hat_inv @ X_hat.T @ y
        
        # Calculate residuals and standard errors
        y_pred = X @ self.beta_
        residuals = y - y_pred
        n_params = X.shape[1]
        df = n_samples - n_params
        
        # Variance-covariance matrix
        sigma_sq = (residuals @ residuals) / df
        vcov = sigma_sq * X_hat_X_hat_inv
        self.se_ = np.sqrt(np.diag(vcov))
        
        # Store additional information
        self.vcov_ = vcov
        self.residuals_ = residuals
        self.n_samples_ = n_samples
        self.n_params_ = n_params
        self.df_ = df
        
        self.fitted_ = True
        return self
    
    def _first_stage_diagnostics(self, X: np.ndarray, Z: np.ndarray, 
                                 pi: np.ndarray, n_samples: int) -> Dict:
        """
        Calculate first-stage diagnostics for instrument strength.
        
        Returns:
        --------
        stats : dict
            Dictionary containing F-statistics, R-squared, etc.
        """
        stats = {}
        
        # Calculate first-stage predictions and residuals
        X_hat = Z @ pi
        
        for i in range(X.shape[1]):
            X_i = X[:, i]
            X_hat_i = X_hat[:, i]
            residuals_i = X_i - X_hat_i
            
            # R-squared
            ss_total = np.sum((X_i - np.mean(X_i))**2)
            ss_residual = np.sum(residuals_i**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            # F-statistic for instrument relevance
            n_instruments = Z.shape[1]
            f_stat = (r_squared / (n_instruments - 1)) / ((1 - r_squared) / (n_samples - n_instruments))
            f_pval = 1 - stats.f.cdf(f_stat, n_instruments - 1, n_samples - n_instruments)
            
            stats[f'endogenous_{i}'] = {
                'r_squared': r_squared,
                'f_statistic': f_stat,
                'f_pvalue': f_pval,
                'weak_instrument': f_stat < 10  # Rule of thumb: F < 10 suggests weak instruments
            }
        
        return stats
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outcomes using the fitted model.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Endogenous variables
            
        Returns:
        --------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted values
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
            
        return X @ self.beta_
    
    def summary(self) -> str:
        """
        Generate a summary of the estimation results.
        
        Returns:
        --------
        summary : str
            Formatted summary of results
        """
        if not self.fitted_:
            return "Model not fitted yet"
        
        # Calculate t-statistics and p-values
        t_stats = self.beta_ / self.se_
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), self.df_))
        
        summary = "=" * 70 + "\n"
        summary += "Instrumental Variables (2SLS) Estimation Results\n"
        summary += "=" * 70 + "\n\n"
        
        summary += f"Number of observations: {self.n_samples_}\n"
        summary += f"Number of parameters: {self.n_params_}\n"
        summary += f"Degrees of freedom: {self.df_}\n\n"
        
        summary += "Coefficient Estimates:\n"
        summary += "-" * 70 + "\n"
        summary += f"{'Variable':<15} {'Coef':<12} {'Std Err':<12} {'t':<10} {'P>|t|':<10}\n"
        summary += "-" * 70 + "\n"
        
        var_names = ['Intercept'] if self.fit_intercept else []
        var_names += [f'X{i}' for i in range(self.n_params_ - (1 if self.fit_intercept else 0))]
        
        for i, name in enumerate(var_names):
            summary += f"{name:<15} {self.beta_[i]:>11.4f} {self.se_[i]:>11.4f} "
            summary += f"{t_stats[i]:>9.3f} {p_values[i]:>9.4f}\n"
        
        summary += "\n" + "=" * 70 + "\n"
        summary += "First Stage Diagnostics:\n"
        summary += "=" * 70 + "\n"
        
        for var, diag in self.first_stage_stats_.items():
            summary += f"\n{var}:\n"
            summary += f"  R-squared: {diag['r_squared']:.4f}\n"
            summary += f"  F-statistic: {diag['f_statistic']:.4f} (p-value: {diag['f_pvalue']:.4f})\n"
            if diag['weak_instrument']:
                summary += "  WARNING: Weak instrument detected (F < 10)\n"
        
        return summary
    
    def get_causal_effect(self, variable_index: int = 0) -> Tuple[float, float, Tuple[float, float]]:
        """
        Get the estimated causal effect with confidence interval.
        
        Parameters:
        -----------
        variable_index : int, default=0
            Index of the variable (0 for first endogenous variable, etc.)
            If fit_intercept=True, use index+1
            
        Returns:
        --------
        effect : float
            Point estimate of causal effect
        se : float
            Standard error
        ci : tuple
            95% confidence interval (lower, upper)
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        idx = variable_index + (1 if self.fit_intercept else 0)
        effect = self.beta_[idx]
        se = self.se_[idx]
        
        # 95% confidence interval
        t_crit = stats.t.ppf(0.975, self.df_)
        ci = (effect - t_crit * se, effect + t_crit * se)
        
        return effect, se, ci

