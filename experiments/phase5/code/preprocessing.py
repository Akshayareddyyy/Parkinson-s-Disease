"""
Phase 5: Fold-Contained Preprocessing & Quantum Angle Mapping
Provides:
  - Yeo-Johnson power transformation for 8 continuous features (fitted strictly on X_dev)
  - Exact angle normalization to [-pi, pi]
  - Discrete orthogonal angle mapping for binary sex: 0 -> 0.0, 1 -> pi
  - Zero target leakage guarantee across train, validation, and test
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

from common import FEATS_9, CONT_FEATS_8

class FoldContainedQuantumPreprocessor:
    """
    Fits transformation parameters strictly on development data (X_dev).
    Applies frozen parameters to validation and test partitions.
    """
    def __init__(self):
        self.pt = PowerTransformer(method='yeo-johnson', standardize=False)
        self.min_vals = None
        self.max_vals = None
        self.fitted = False
        
        # Also provide classical standard scaler for fair classical control comparisons
        self.classical_scaler = StandardScaler()
        
    def fit(self, df_dev):
        X_cont = df_dev[CONT_FEATS_8].values.astype(np.float64)
        
        # 1. Fit Yeo-Johnson on continuous features
        self.pt.fit(X_cont)
        Z_cont = self.pt.transform(X_cont)
        
        # 2. Compute min and max bounds for angle normalization to [-pi, pi]
        self.min_vals = np.min(Z_cont, axis=0)
        self.max_vals = np.max(Z_cont, axis=0)
        
        # 3. Fit classical standard scaler on all 9 features
        self.classical_scaler.fit(df_dev[FEATS_9].values.astype(np.float64))
        
        self.fitted = True
        return self
        
    def transform_quantum_angles(self, df):
        """
        Transforms features into quantum rotation angles theta in [-pi, pi]^9.
        Feature order: ['age', 'sex', 'test_time', 'Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted on development data before transform.")
            
        X_cont = df[CONT_FEATS_8].values.astype(np.float64)
        Z_cont = self.pt.transform(X_cont)
        
        # Min-max scale continuous features to [-pi, pi] using development bounds
        range_vals = self.max_vals - self.min_vals + 1e-8
        theta_cont = 2.0 * np.pi * ((Z_cont - self.min_vals) / range_vals) - np.pi
        theta_cont = np.clip(theta_cont, -np.pi, np.pi)
        
        # Discrete orthogonal mapping for binary sex: 0 -> 0.0, 1 -> pi
        sex_raw = df['sex'].values.astype(np.float64)
        theta_sex = np.where(sex_raw == 1.0, np.pi, 0.0).reshape(-1, 1)
        
        # Assemble back into canonical FEATS_9 order:
        # FEATS_9: ['age', 'sex', 'test_time', 'Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']
        # CONT_FEATS_8 order: age(0), test_time(1), Jitter(2), Shimmer(3), NHR(4), HNR(5), RPDE(6), DFA(7)
        theta_9 = np.hstack([
            theta_cont[:, 0:1],   # age
            theta_sex,            # sex
            theta_cont[:, 1:]     # test_time, Jitter, Shimmer, NHR, HNR, RPDE, DFA
        ])
        return theta_9

    def transform_classical(self, df):
        """
        Returns standardized features for matched classical controls.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted on development data before transform.")
        return self.classical_scaler.transform(df[FEATS_9].values.astype(np.float64))

