import numpy as np

## MODIFY FROM causalicp package
import scipy.stats
from causalicp import _Data

def direct_icp_test(y, S, data):
    data = _Data(data)
    """Test the hypothesis for the invariance of the set S for the
    target/response y"""
    # Compute pooled coefficients and environment-wise residuals
    coefs, intercept = data.regress_pooled(y, S)
    residuals = data.residuals(y, coefs, intercept)
    # Build p-values for the hypothesis that error distribution
    # remains invariant in each environment
    mean_pvalues = np.zeros(data.e)
    var_pvalues = np.zeros(data.e)
    for i in range(data.e):
        residuals_i = residuals[i]
        residuals_others = np.hstack([residuals[j] for j in range(data.e) if j != i])
        mean_pvalues[i] = _t_test(residuals_i, residuals_others)
        var_pvalues[i] = _f_test(residuals_i, residuals_others)
    # Combine p-values via bonferroni correction
    smallest_pvalue = min(min(mean_pvalues), min(var_pvalues))
    p_value = min(1, smallest_pvalue * 2 * (data.e - 1))  # The -1 term is from the R implementation
    # If set is accepted, compute confidence intervals
    return p_value

def _t_test(X, Y):
    """Return the p-value of the two sample t-test for the given samples."""
    result = scipy.stats.ttest_ind(X, Y, equal_var=False)
    return result.pvalue


def _f_test(X, Y):
    """Return the p-value of the two-sided f-test for the given samples."""
    F = np.var(X, ddof=1) / np.var(Y, ddof=1)
    p = scipy.stats.f.cdf(F, len(X) - 1, len(Y) - 1)
    return 2 * min(p, 1 - p)
