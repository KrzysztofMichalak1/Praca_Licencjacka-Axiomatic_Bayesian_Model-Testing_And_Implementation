from numba import njit
import numpy as np
from scipy.linalg import cholesky, solve_triangular

@njit(fastmath=True)
def x_from_w(w, clip_value=10.0):
    u = w
    u_clipped = np.clip(u, -clip_value, clip_value)
    
    max_u = np.max(u_clipped)
    exp_u = np.exp(u_clipped - max_u)
    sum_exp_u = np.sum(exp_u)
    
    denom = np.exp(-max_u) + sum_exp_u
    x1 = np.exp(-max_u) / denom
    x_rest = exp_u / denom
    
    x = np.empty(w.size + 1)
    x[0] = x1
    x[1:] = x_rest
    return x

def w_from_x(x):
    """Transformation from x to w with standardization."""
    if np.any(x <= 0):
        return None
    x1 = x[0]
    u = np.log(x[1:] / x1)
    return u

class MultivariateNormalCholesky:
    def __init__(self, Sigma):
        print("▶ [MVN] Budowa rozkładu wielowymiarowego...")
        self.L = cholesky(Sigma, lower=True)
        self.logdet = 2 * np.sum(np.log(np.diag(self.L)))
        self.dim = Sigma.shape[0]
        self.log_norm_const = -0.5*(self.dim*np.log(2*np.pi) + self.logdet)
        print("  ✔ Cholesky ukończone.")

    def logpdf(self, u):
        y = solve_triangular(self.L, u, lower=True)
        return self.log_norm_const - 0.5*np.dot(y, y)

@njit(fastmath=True)
def solve_lower_triangular(L, b):
    m = L.shape[0]
    y = np.empty(m)
    for i in range(m):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = (b[i] - s) / L[i, i]
    return y

@njit(fastmath=True)
def log_posterior_fast(w, counts_f, L, log_norm_const):
    """Optimized log-posterior with precomputation."""
    u = w
    max_u = np.max(u)
    exp_u = np.exp(u - max_u)
    denom = np.exp(-max_u) + np.sum(exp_u)
    x0=1/denom
    x1 = np.exp(-max_u)*x0
    x_rest = exp_u * x0

    if np.any(x_rest <= 0.0) or x1 <= 0.0:
        return -1e300

    y = solve_lower_triangular(L, u)
    lp = log_norm_const - 0.5 * np.dot(y, y)

    ll = np.dot(counts_f[1:], np.log(x_rest)) + counts_f[0] * np.log(x1)

    return lp + ll
