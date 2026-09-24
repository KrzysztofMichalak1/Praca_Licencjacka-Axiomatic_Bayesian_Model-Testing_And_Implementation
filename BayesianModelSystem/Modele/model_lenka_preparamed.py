import numpy as np
from scipy.linalg import cholesky, solve_triangular
from numba import njit
import math

# =====================================================================
# FUNKCJE ZOPTYMALIZOWANE (NUMBA)
# =====================================================================

@njit(fastmath=True)
def z_to_x_softmax(Z):
    """Transformacja logistyczna (softmax) zgodnie z równaniem 40."""
    max_Z = np.max(Z)
    exp_Z = np.exp(Z - max_Z)
    sum_exp_Z = np.sum(exp_Z)
    return exp_Z / sum_exp_Z if sum_exp_Z != 0 else np.full_like(Z, 1.0 / Z.size)

@njit(fastmath=True)
def log_posterior_logistic_normal_fast(Z, counts, N, K_inv, log_prior_norm_const, mu_prior):
    """
    Szybkie obliczanie log-posterior (Równanie 42 - Multinomial Likelihood + Gaussian Prior).
    """
    max_Z = np.max(Z)
    log_sum_exp_Z = max_Z + np.log(np.sum(np.exp(Z - max_Z)))
    log_lik = np.dot(counts, Z) - N * log_sum_exp_Z
    
    if not np.isfinite(log_lik): 
        return -np.inf
        
    Z_centered = Z - mu_prior
    quad_form = np.dot(Z_centered, np.dot(K_inv, Z_centered))
    log_prior = log_prior_norm_const - 0.5 * quad_form
    
    return log_prior + log_lik if np.isfinite(log_prior) else -np.inf


# =====================================================================
# KLASA BAZOWA: ModelLenkaPreparamed
# =====================================================================

class ModelLenkaPreparamed:
    """
    Model logistyczno-normalny dla danych w postaci przestrzennego procesu punktowego.
    Wykorzystuje transformację softmax dla latentnego pola Z.
    """
    def __init__(self, space_points, metric_func, observed_indices,
                 lengthscale=50.0, variance=1.0, distance_unit='km',
                 mu_prior=0.0):
        print("\n▶ [MODEL-LENKA-PREPARAMED] Inicjalizacja modelu...")
        self.space_points = list(space_points)
        self.metric = metric_func
        self.n = len(self.space_points)
        self.observed_indices = np.asarray(observed_indices)
        self.counts = np.bincount(self.observed_indices, minlength=self.n).astype(np.float64)
        self.N = self.counts.sum()
        print(f"  - Liczba punktów: {self.n}, Obserwacje: {self.N}")
        self.lengthscale = lengthscale
        self.variance = variance
        self.distance_unit = distance_unit
        self.mu_prior = mu_prior
        self.K, self.K_inv, self.log_prior_norm_const, self.samples_Z, self.samples_x = None, None, None, None, None
        print("  ✔ Model gotowy.\n")

    def przygotuj_apriori(self):
        print("▶ [APRIORI] Budowa macierzy kowariancji...")
        dist_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                # Bezpieczne pobieranie dystansu z uwzględnieniem jednostki
                try:
                    dist = self.metric(self.space_points[i], self.space_points[j], return_unit=self.distance_unit)
                except TypeError:
                    dist = self.metric(self.space_points[i], self.space_points[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
                
        # Zgodnie z równaniem 41 (Jądro wykładnicze kwadratowe)
        self.K = self.variance * np.exp(-(dist_matrix / self.lengthscale)**2) + 1e-8 * np.eye(self.n)
        
        try:
            L = cholesky(self.K, lower=True)
            L_inv = solve_triangular(L, np.eye(self.n), lower=True)
            self.K_inv = L_inv.T @ L_inv
            self.log_prior_norm_const = -0.5 * (self.n * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L))))
            print("  ✔ Prekomputacja stałych zakończona.")
        except np.linalg.LinAlgError:
            print("  ⚠ Ostrzeżenie: Problemy numeryczne z macierzą K, używam stabilizacji zapasowej.")
            self.K_inv = np.linalg.inv(self.K)
            sign, logdet = np.linalg.slogdet(self.K)
            self.log_prior_norm_const = -0.5 * (self.n * np.log(2 * np.pi) + logdet)

    def _log_posterior(self, Z):
        diff = Z - self.mu_prior
        prior_term = -0.5 * diff.T @ self.K_inv @ diff + self.log_prior_norm_const
        p = np.exp(Z - np.max(Z))
        p /= np.sum(p)
        return prior_term + np.sum(self.counts * np.log(p + 1e-12))

    def przygotuj_predykcyjny(self, num_samples, burn_in, proposal_scale, seed=None):
        print("\n▶ [MCMC] START SAMPLERA MODELU LENKA")
        if seed: np.random.seed(seed)
        if self.K_inv is None: raise RuntimeError("Uruchom 'przygotuj_apriori'.")
        Z_current = np.full(self.n, self.mu_prior)
        logpost_current = self._log_posterior(Z_current)
        try:
            L_prop = cholesky((proposal_scale**2) * self.K, lower=True)
        except:
            L_prop = np.sqrt((proposal_scale**2)) * np.eye(self.n)
            
        samples, accepted = [], 0
        total_iterations = num_samples + burn_in
        for i in range(total_iterations):
            Z_prop = Z_current + L_prop @ np.random.normal(0, 1, self.n)
            logpost_prop = self._log_posterior(Z_prop)
            if np.isfinite(logpost_prop) and (logpost_prop - logpost_current >= 0 or np.log(np.random.uniform()) < logpost_prop - logpost_current):
                Z_current, logpost_current = Z_prop, logpost_prop
                if i >= burn_in: accepted += 1
            if i >= burn_in: samples.append(Z_current.copy())
            if i % 1000 == 0 or i == total_iterations - 1:
                status = "BURN-IN" if i < burn_in else "SAMPLING"
                print(f"    🔸 Iter {i:5d}/{total_iterations} [{status:8s}] | LP={logpost_current:8.1f} | Acc={accepted/max(1, i-burn_in+1):.3f}")
        self.samples_Z = np.array(samples)
        self.samples_x = np.array([z_to_x_softmax(z) for z in self.samples_Z]) if len(self.samples_Z) > 0 else np.array([])
        print(f"✔ MCMC ZAKOŃCZONE (Acc: {accepted/max(1, num_samples):.3f})\n")

    def posterior_mean(self):
        if self.samples_x is None or len(self.samples_x) == 0: raise ValueError("Brak próbek.")
        return np.mean(self.samples_x, axis=0)


