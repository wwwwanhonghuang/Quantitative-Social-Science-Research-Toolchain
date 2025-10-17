#!/usr/bin/env python3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import argparse

def propensity_score_matching(data_file, output_file):
    """Perform propensity score matching"""
    df = pd.read_csv(data_file)
    
    # Example: Treatment effect analysis
    treatment_var = 'treatment'
    covariates = ['age', 'income', 'education']
    
    # Calculate propensity scores
    X = df[covariates]
    y = df[treatment_var]
    
    # Standardize covariates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit propensity score model
    ps_model = LogisticRegression()
    ps_model.fit(X_scaled, y)
    
    propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
    
    # Save results
    results = pd.DataFrame({
        'propensity_score': propensity_scores,
        'treatment': y
    })
    results.to_csv(output_file, index=False)
    
    print(f"✓ Propensity scores saved to {output_file}")