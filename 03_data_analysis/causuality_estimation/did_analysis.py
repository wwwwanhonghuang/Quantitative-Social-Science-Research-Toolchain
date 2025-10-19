"""
Difference-in-Differences (DID) Estimation for Causal Inference
Implements classical 2x2 DID and regression-based DID with controls
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Union, Dict, Tuple
import warnings


class DIDEstimator:
    """
    Difference-in-Differences estimator for causal inference with panel data.
    
    DID compares the change in outcomes over time between a treatment group
    and a control group to identify causal effects.
    
    Attributes:
        att_: Average Treatment Effect on the Treated
        se_: Standard error of ATT
        parallel_trends_test_: Results of parallel trends test
        fitted_: Boolean indicating if model has been fitted
    """
    
    def __init__(self, fit_intercept: bool = True):
        """
        Initialize the DID estimator.
        
        Parameters:
        -----------
        fit_intercept : bool, default=True
            Whether to include an intercept term
        """
        self.fit_intercept = fit_intercept
        self.fitted_ = False
        self.att_ = None
        self.se_ = None
        self.parallel_trends_test_ = None
        
    def fit(self, 
            y: np.ndarray,
            treat: np.ndarray,
            post: np.ndarray,
            X: Optional[np.ndarray] = None,
            cluster: Optional[np.ndarray] = None) -> 'DIDEstimator':
        """
        Fit the DID model using regression approach.
        
        The regression specification is:
        y = β₀ + β₁·treat + β₂·post + β₃·(treat×post) + X'γ + ε
        
        where β₃ is the DID estimator (ATT).
        
        Parameters:
        -----------
        y : np.ndarray, shape (n_samples,)
            Outcome variable
        treat : np.ndarray, shape (n_samples,)
            Treatment group indicator (1 = treated, 0 = control)
        post : np.ndarray, shape (n_samples,)
            Post-treatment period indicator (1 = post, 0 = pre)
        X : np.ndarray, shape (n_samples, n_controls), optional
            Additional control variables
        cluster : np.ndarray, shape (n_samples,), optional
            Cluster identifiers for clustered standard errors
            
        Returns:
        --------
        self : DIDEstimator
            Fitted estimator
        """
        # Input validation
        y = np.asarray(y).ravel()
        treat = np.asarray(treat).ravel()
        post = np.asarray(post).ravel()
        
        n_samples = len(y)
        
        if len(treat) != n_samples or len(post) != n_samples:
            raise ValueError("y, treat, and post must have the same length")
        
        # Create interaction term (this is the DID estimator)
        treat_post = treat * post
        
        # Build design matrix
        if self.fit_intercept:
            design_matrix = np.column_stack([np.ones(n_samples), treat, post, treat_post])
            var_names = ['Intercept', 'Treat', 'Post', 'Treat×Post']
        else:
            design_matrix = np.column_stack([treat, post, treat_post])
            var_names = ['Treat', 'Post', 'Treat×Post']
        
        # Add controls if provided
        if X is not None:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            design_matrix = np.column_stack([design_matrix, X])
            var_names += [f'Control_{i}' for i in range(X.shape[1])]
        
        # OLS estimation
        X_mat = design_matrix
        X_X_inv = np.linalg.inv(X_mat.T @ X_mat)
        self.beta_ = X_X_inv @ X_mat.T @ y
        
        # ATT is the coefficient on the interaction term
        att_idx = 3 if self.fit_intercept else 2
        self.att_ = self.beta_[att_idx]
        
        # Calculate residuals
        y_pred = X_mat @ self.beta_
        residuals = y - y_pred
        
        # Degrees of freedom
        n_params = X_mat.shape[1]
        df = n_samples - n_params
        
        # Variance estimation
        if cluster is not None:
            # Clustered standard errors
            vcov = self._clustered_vcov(X_mat, residuals, cluster, X_X_inv)
        else:
            # Heteroskedasticity-robust standard errors (HC1)
            vcov = self._robust_vcov(X_mat, residuals, X_X_inv, df)
        
        self.se_ = np.sqrt(np.diag(vcov))
        
        # Store additional information
        self.vcov_ = vcov
        self.residuals_ = residuals
        self.n_samples_ = n_samples
        self.n_params_ = n_params
        self.df_ = df
        self.var_names_ = var_names
        self.treat_ = treat
        self.post_ = post
        self.y_ = y
        
        self.fitted_ = True
        return self
    
    def fit_2x2(self, 
                y_treat_pre: float,
                y_treat_post: float,
                y_control_pre: float,
                y_control_post: float,
                n_treat_pre: int,
                n_treat_post: int,
                n_control_pre: int,
                n_control_post: int) -> 'DIDEstimator':
        """
        Fit DID using simple 2×2 means (classical DID formula).
        
        ATT = (Ȳ_treat,post - Ȳ_treat,pre) - (Ȳ_control,post - Ȳ_control,pre)
        
        Parameters:
        -----------
        y_treat_pre, y_treat_post : float
            Mean outcomes for treatment group in pre/post periods
        y_control_pre, y_control_post : float
            Mean outcomes for control group in pre/post periods
        n_treat_pre, n_treat_post : int
            Sample sizes for treatment group in pre/post periods
        n_control_pre, n_control_post : int
            Sample sizes for control group in pre/post periods
            
        Returns:
        --------
        self : DIDEstimator
            Fitted estimator
        """
        # Calculate DID estimator
        diff_treat = y_treat_post - y_treat_pre
        diff_control = y_control_post - y_control_pre
        self.att_ = diff_treat - diff_control
        
        # Store components for summary
        self.diff_treat_ = diff_treat
        self.diff_control_ = diff_control
        self.means_ = {
            'treat_pre': y_treat_pre,
            'treat_post': y_treat_post,
            'control_pre': y_control_pre,
            'control_post': y_control_post
        }
        
        # Calculate variance (assuming independence and homoskedasticity)
        # Var(ATT) = Var(Ȳ_T1) + Var(Ȳ_T0) + Var(Ȳ_C1) + Var(Ȳ_C0)
        # This is approximate - better to use regression approach with real data
        self.n_samples_ = n_treat_pre + n_treat_post + n_control_pre + n_control_post
        
        # For demonstration, we'll compute standard error
        # In practice, you'd estimate variances from actual data
        self.se_ = None  # Would need variance estimates
        self.simple_2x2_ = True
        self.fitted_ = True
        
        return self
    
    def _robust_vcov(self, X: np.ndarray, residuals: np.ndarray, 
                     X_X_inv: np.ndarray, df: int) -> np.ndarray:
        """
        Calculate heteroskedasticity-robust variance-covariance matrix (HC1).
        """
        # HC1: n/(n-k) adjustment for finite sample
        meat = X.T @ np.diag(residuals**2) @ X
        vcov = (self.n_samples_ / df) * X_X_inv @ meat @ X_X_inv
        return vcov
    
    def _clustered_vcov(self, X: np.ndarray, residuals: np.ndarray,
                       cluster: np.ndarray, X_X_inv: np.ndarray) -> np.ndarray:
        """
        Calculate cluster-robust variance-covariance matrix.
        """
        cluster = np.asarray(cluster)
        unique_clusters = np.unique(cluster)
        n_clusters = len(unique_clusters)
        
        # Compute cluster-robust "meat"
        meat = np.zeros((X.shape[1], X.shape[1]))
        for c in unique_clusters:
            mask = cluster == c
            X_c = X[mask]
            e_c = residuals[mask]
            meat += (X_c.T @ e_c).reshape(-1, 1) @ (X_c.T @ e_c).reshape(1, -1)
        
        # Finite sample adjustment
        adj = (n_clusters / (n_clusters - 1)) * ((self.n_samples_ - 1) / (self.n_samples_ - X.shape[1]))
        vcov = adj * X_X_inv @ meat @ X_X_inv
        
        return vcov
    
    def parallel_trends_test(self, 
                            y: np.ndarray,
                            treat: np.ndarray,
                            time: np.ndarray,
                            treated_period: Union[int, float]) -> Dict:
        """
        Test the parallel trends assumption using pre-treatment data.
        
        Tests whether treatment and control groups had parallel trends
        before the treatment by regressing:
        y = β₀ + β₁·treat + β₂·time + β₃·(treat×time) + ε
        
        on pre-treatment data only. H₀: β₃ = 0 (parallel trends)
        
        Parameters:
        -----------
        y : np.ndarray
            Outcome variable
        treat : np.ndarray
            Treatment group indicator
        time : np.ndarray
            Time variable (continuous or discrete)
        treated_period : int or float
            First period when treatment occurred
            
        Returns:
        --------
        results : dict
            Test statistics and p-value
        """
        # Filter to pre-treatment periods
        pre_mask = time < treated_period
        y_pre = y[pre_mask]
        treat_pre = treat[pre_mask]
        time_pre = time[pre_mask]
        
        # Regression: y = β₀ + β₁·treat + β₂·time + β₃·(treat×time) + ε
        treat_time = treat_pre * time_pre
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(y_pre)), treat_pre, time_pre, treat_time])
            interaction_idx = 3
        else:
            X = np.column_stack([treat_pre, time_pre, treat_time])
            interaction_idx = 2
        
        # OLS
        beta = np.linalg.inv(X.T @ X) @ X.T @ y_pre
        residuals = y_pre - X @ beta
        
        # Standard errors
        n = len(y_pre)
        k = X.shape[1]
        df = n - k
        sigma_sq = (residuals @ residuals) / df
        vcov = sigma_sq * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(vcov))
        
        # Test H₀: β₃ = 0
        t_stat = beta[interaction_idx] / se[interaction_idx]
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
        
        self.parallel_trends_test_ = {
            'interaction_coef': beta[interaction_idx],
            'se': se[interaction_idx],
            't_statistic': t_stat,
            'p_value': p_value,
            'df': df,
            'reject_parallel_trends': p_value < 0.05
        }
        
        return self.parallel_trends_test_
    
    def event_study(self,
                   y: np.ndarray,
                   treat: np.ndarray,
                   time: np.ndarray,
                   treated_period: Union[int, float],
                   time_periods: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Estimate event study / dynamic DID coefficients.
        
        Estimates treatment effects for each time period relative to treatment:
        y_it = α_i + λ_t + Σ_k β_k·1{t-t* = k}·Treat_i + ε_it
        
        where t* is the treatment period and k indexes relative time.
        
        Parameters:
        -----------
        y : np.ndarray
            Outcome variable
        treat : np.ndarray
            Treatment group indicator
        time : np.ndarray
            Time period
        treated_period : int or float
            Treatment period
        time_periods : np.ndarray, optional
            Specific relative time periods to estimate (e.g., [-3, -2, -1, 0, 1, 2])
            
        Returns:
        --------
        results : pd.DataFrame
            Coefficients and standard errors for each relative time period
        """
        # Calculate relative time
        rel_time = time - treated_period
        
        if time_periods is None:
            time_periods = np.unique(rel_time)
        
        # Omit one pre-period for identification (typically -1)
        time_periods = time_periods[time_periods != -1]
        
        # Create indicators for each relative time period
        indicators = []
        for t in time_periods:
            indicators.append((rel_time == t).astype(float) * treat)
        
        # Stack all indicators
        X_event = np.column_stack(indicators)
        
        # Add time and group fixed effects
        time_dummies = pd.get_dummies(time, drop_first=True).values
        group_dummies = treat.reshape(-1, 1)
        
        X_full = np.column_stack([np.ones(len(y)), group_dummies, time_dummies, X_event])
        
        # OLS
        beta = np.linalg.inv(X_full.T @ X_full) @ X_full.T @ y
        residuals = y - X_full @ beta
        
        # Standard errors
        n = len(y)
        k = X_full.shape[1]
        df = n - k
        sigma_sq = (residuals @ residuals) / df
        vcov = sigma_sq * np.linalg.inv(X_full.T @ X_full)
        se = np.sqrt(np.diag(vcov))
        
        # Extract event study coefficients (last len(time_periods) coefficients)
        event_coefs = beta[-len(time_periods):]
        event_se = se[-len(time_periods):]
        
        # Create results dataframe
        results = pd.DataFrame({
            'relative_time': time_periods,
            'coefficient': event_coefs,
            'se': event_se,
            't_stat': event_coefs / event_se,
            'p_value': 2 * (1 - stats.t.cdf(np.abs(event_coefs / event_se), df)),
            'ci_lower': event_coefs - 1.96 * event_se,
            'ci_upper': event_coefs + 1.96 * event_se
        })
        
        return results
    
    def predict(self, treat: np.ndarray, post: np.ndarray, 
                X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict outcomes using the fitted model.
        
        Parameters:
        -----------
        treat : np.ndarray
            Treatment group indicator
        post : np.ndarray
            Post-treatment period indicator
        X : np.ndarray, optional
            Control variables
            
        Returns:
        --------
        y_pred : np.ndarray
            Predicted outcomes
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        treat = np.asarray(treat).ravel()
        post = np.asarray(post).ravel()
        treat_post = treat * post
        
        if self.fit_intercept:
            design_matrix = np.column_stack([np.ones(len(treat)), treat, post, treat_post])
        else:
            design_matrix = np.column_stack([treat, post, treat_post])
        
        if X is not None:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            design_matrix = np.column_stack([design_matrix, X])
        
        return design_matrix @ self.beta_
    
    def summary(self) -> str:
        """
        Generate a summary of the DID estimation results.
        
        Returns:
        --------
        summary : str
            Formatted summary of results
        """
        if not self.fitted_:
            return "Model not fitted yet"
        
        if hasattr(self, 'simple_2x2_') and self.simple_2x2_:
            return self._summary_2x2()
        
        # Calculate t-statistics and p-values
        t_stats = self.beta_ / self.se_
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), self.df_))
        
        summary = "=" * 70 + "\n"
        summary += "Difference-in-Differences Estimation Results\n"
        summary += "=" * 70 + "\n\n"
        
        summary += f"Number of observations: {self.n_samples_}\n"
        summary += f"Number of parameters: {self.n_params_}\n"
        summary += f"Degrees of freedom: {self.df_}\n\n"
        
        summary += "Coefficient Estimates:\n"
        summary += "-" * 70 + "\n"
        summary += f"{'Variable':<15} {'Coef':<12} {'Std Err':<12} {'t':<10} {'P>|t|':<10}\n"
        summary += "-" * 70 + "\n"
        
        for i, name in enumerate(self.var_names_):
            summary += f"{name:<15} {self.beta_[i]:>11.4f} {self.se_[i]:>11.4f} "
            summary += f"{t_stats[i]:>9.3f} {p_values[i]:>9.4f}"
            if 'Treat×Post' in name:
                summary += "  ← ATT (DID Estimator)"
            summary += "\n"
        
        summary += "\n" + "=" * 70 + "\n"
        summary += f"Average Treatment Effect on Treated (ATT): {self.att_:.4f}\n"
        
        att_idx = self.var_names_.index('Treat×Post')
        summary += f"Standard Error: {self.se_[att_idx]:.4f}\n"
        
        t_att = t_stats[att_idx]
        p_att = p_values[att_idx]
        ci_lower = self.att_ - 1.96 * self.se_[att_idx]
        ci_upper = self.att_ + 1.96 * self.se_[att_idx]
        
        summary += f"t-statistic: {t_att:.4f}\n"
        summary += f"p-value: {p_att:.4f}\n"
        summary += f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
        
        if p_att < 0.05:
            summary += "\n*** Treatment effect is statistically significant at 5% level ***\n"
        
        return summary
    
    def _summary_2x2(self) -> str:
        """Summary for simple 2×2 DID."""
        summary = "=" * 70 + "\n"
        summary += "Difference-in-Differences Estimation (2×2 Design)\n"
        summary += "=" * 70 + "\n\n"
        
        summary += "Group Means:\n"
        summary += "-" * 70 + "\n"
        summary += f"{'Group':<15} {'Pre-Period':<15} {'Post-Period':<15} {'Difference':<15}\n"
        summary += "-" * 70 + "\n"
        
        summary += f"{'Treatment':<15} {self.means_['treat_pre']:>14.4f} "
        summary += f"{self.means_['treat_post']:>14.4f} {self.diff_treat_:>14.4f}\n"
        
        summary += f"{'Control':<15} {self.means_['control_pre']:>14.4f} "
        summary += f"{self.means_['control_post']:>14.4f} {self.diff_control_:>14.4f}\n"
        
        summary += "-" * 70 + "\n"
        summary += f"{'DID Estimate':<15} {'':>14} {'':>14} {self.att_:>14.4f}\n"
        
        summary += "\n" + "=" * 70 + "\n"
        summary += f"Average Treatment Effect on Treated (ATT): {self.att_:.4f}\n"
        summary += "=" * 70 + "\n"
        
        return summary
    
    def get_att(self) -> Tuple[float, Optional[float], Optional[Tuple[float, float]]]:
        """
        Get the estimated ATT with confidence interval.
        
        Returns:
        --------
        att : float
            Average Treatment Effect on the Treated
        se : float or None
            Standard error
        ci : tuple or None
            95% confidence interval (lower, upper)
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self, 'simple_2x2_') and self.simple_2x2_:
            return self.att_, None, None
        
        att_idx = self.var_names_.index('Treat×Post')
        se = self.se_[att_idx]
        ci = (self.att_ - 1.96 * se, self.att_ + 1.96 * se)
        
        return self.att_, se, ci

