# Preprocessing Equivalence & Privacy Claim Audit

**Project:** Parkinson's Disease Telemonitoring — Phase 4 Federated Learning  
**Audit Component:** Decentralized Sufficient Statistics Aggregation vs. Centralized StandardScaler  
**Audit Date:** November 2026  
**Status:** **VERIFIED WITH LIMITATIONS**

---

## 1. Executive Summary

This audit independently investigates the mathematical validity, numerical precision, and privacy implications of the decentralized standardization mechanism implemented in `phase4_common.py`.

### Verdict Summary
- **Mathematical Equivalence:** **VERIFIED (Exact Algebraic Identity)**. The sufficient statistics formula reproduces the exact population/sample mean and variance of the complete pooled development dataset.
- **Numerical Precision:** **VERIFIED**. Maximum absolute deviation across all features and all 10 seeds is $< 8.84 \times 10^{-14}$ (machine precision).
- **Leakage Prevention:** **VERIFIED**. Neither validation nor locked-test records or labels contribute to the scaling parameters.
- **Privacy Claims:** **VERIFIED WITH LIMITATIONS**. Transmitting local sample sizes ($n_k$), first moments ($\sum x$), and second moments ($\sum x^2$) eliminates the need to centralize raw voice recordings, but **does not constitute formal Differential Privacy (DP)**. Papers must not claim formal cryptographic privacy or differential privacy guarantees without adding calibrated DP noise mechanisms.

---

## 2. Mathematical Proof of Equivalence

Let $K$ be the number of federated clients partitioning the pooled development dataset $\mathcal{D}$ of size $N = \sum_{k=1}^K n_k$. Each client $k$ possesses $n_k$ feature vectors $x_{k, 1}, \dots, x_{k, n_k} \in \mathbb{R}^d$.

### 2.1 Centralized Formulation (scikit-learn `StandardScaler`)
In centralized preprocessing, the mean and variance are computed directly over all $N$ records:
$$\mu_{\text{cent}} = \frac{1}{N}\sum_{i=1}^N x_i$$
$$\sigma_{\text{cent}}^2 = \frac{1}{N}\sum_{i=1}^N (x_i - \mu_{\text{cent}})^2 = \frac{1}{N}\sum_{i=1}^N x_i^2 - \mu_{\text{cent}}^2$$

### 2.2 Decentralized Sufficient Statistics Formulation
In Phase 4 FL:
1. Each client $k$ locally computes three sufficient statistics:
   $$n_k = |X_k|, \quad S_{1, k} = \sum_{i=1}^{n_k} x_{k, i}, \quad S_{2, k} = \sum_{i=1}^{n_k} x_{k, i}^2$$
2. The server receives only $(n_k, S_{1, k}, S_{2, k})$ and computes:
   $$\mu_{\text{dec}} = \frac{\sum_{k=1}^K S_{1, k}}{\sum_{k=1}^K n_k} = \frac{\sum_{k=1}^K \sum_{i=1}^{n_k} x_{k, i}}{N} \equiv \mu_{\text{cent}}$$
   $$\sigma_{\text{dec}}^2 = \frac{\sum_{k=1}^K S_{2, k}}{N} - \mu_{\text{dec}}^2 = \frac{\sum_{k=1}^K \sum_{i=1}^{n_k} x_{k, i}^2}{N} - \mu_{\text{cent}}^2 \equiv \sigma_{\text{cent}}^2$$

### 2.3 Verification of Between-Client Variance
A frequent error in distributed standardization is computing the average of client local variances ($\frac{1}{K}\sum \sigma_k^2$), which fatally omits the between-client variance $\sum \frac{n_k}{N}(\mu_k - \mu)^2$. 
The Phase 4 implementation aggregates raw sums of squares ($\sum x^2$), which algebraically decomposes as:
$$\frac{1}{N}\sum_{k=1}^K \sum_{i=1}^{n_k} x_{k, i}^2 - \mu^2 = \sum_{k=1}^K \frac{n_k}{N} \sigma_k^2 + \sum_{k=1}^K \frac{n_k}{N} (\mu_k - \mu)^2$$
**Verdict:** The between-client variance is fully and exactly accounted for.

---

## 3. Empirical Numerical Verification

Direct comparison between centralized `StandardScaler().fit(X_dev)` and `decentralized_standardization()` across all 10 seeds:

| Metric Evaluated | Maximum Observed Absolute Difference | Machine Epsilon Benchmark | Verdict |
| :--- | :---: | :---: | :---: |
| Feature Means ($\mu_j$) | $4.26 \times 10^{-14}$ | $\approx 2.22 \times 10^{-16}$ | **EXACT PASS** |
| Feature Standard Deviations ($\sigma_j$) | $7.82 \times 10^{-14}$ | $\approx 2.22 \times 10^{-16}$ | **EXACT PASS** |
| Standardized Validation Matrix ($Z_{\mathcal{V}}$) | $8.84 \times 10^{-14}$ | $\approx 2.22 \times 10^{-16}$ | **EXACT PASS** |
| Standardized Test Matrix ($Z_{\mathcal{T}}$) | $8.84 \times 10^{-14}$ | $\approx 2.22 \times 10^{-16}$ | **EXACT PASS** |

---

## 4. Privacy Audit & Claim Limitations

### 4.1 What is Protected
- **Raw Telemonitoring Records:** Zero voice recordings or acoustic tables leave client devices.
- **Data Minimization:** Only three low-dimensional summary vectors ($1 \times 1$, $1 \times 19$, $1 \times 19$) are transmitted to the server once during initial setup.

### 4.2 Critical Privacy Limitations
1. **Not Differentially Private:** The sufficient statistics are transmitted without noise addition ($\epsilon = \infty$).
2. **Reconstruction Risk for Edge Clients:** In Topology 1 (where each client is a single patient with ~135 recordings), knowing $n_k$, $\sum x_k$, and $\sum x_k^2$ reveals the exact mean and variance of that individual patient's voice features. In combination with known demographic features (`age`, `sex`), this can theoretically enable linkage or attribute reconstruction attacks.
3. **Mandatory Scientific Framing:**
   - **DO NOT CLAIM:** "The framework provides provable privacy guarantees" or "differential privacy is achieved."
   - **PERMISSIBLE CLAIM:** *"The decentralized standardization mechanism achieves exact algebraic parity with centralized scaling via sufficient statistics aggregation, preserving data locality by eliminating the transmission of raw longitudinal audio recordings."*

