import numpy as np
from scipy import stats
import warnings
from .model_bazowy_przestrzenny import BayesianSpatialModel

class ModelGausowskiSprzezony(BayesianSpatialModel):
    """
    Gaussowski model przestrzenny z priorami sprzezonymi, implementujący Proces Gaussowski (GP).
    Dla danych ciaglych.
    """
    
    def __init__(self, space_points, metric_func, observed_indices, 
                 counts=None, N=None, distance_unit='km',
                 mu_prior=0.0, sigma_prior=1.0):
        super().__init__(space_points, metric_func, observed_indices, 
                        counts, N, distance_unit)
        
        if counts is None:
            raise ValueError("'counts' must be provided for Gaussian model")
        
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.y = self.counts
        
        print(f"   - Model: Model Gausowski (sprzężony)")
        print(f"   - Prior: m ~ N({mu_prior}, {sigma_prior})")
    
    def fit(self, phi=4, optimize_phi=True):
        print(f"\n▶ [GAUSSIAN-CONJUGATE] Dopasowywanie modelu...")
        y_obs = self.y
        n_obs = len(y_obs)
        
        if phi is None and optimize_phi:
            phi = self._optimize_phi(y_obs)
        elif phi is None:
            phi = np.max(self.distance_matrix) / 3
            print(f"   - Phi (heurystyka): {phi:.0f}")
        
        D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
        W_obs = np.exp(-D_obs / phi)
        W_obs = W_obs + np.eye(n_obs) * 1e-8
        
        try:
            L = np.linalg.cholesky(W_obs)
        except np.linalg.LinAlgError:
            W_obs = W_obs + np.eye(n_obs) * 1e-6
            L = np.linalg.cholesky(W_obs)
        
        y_transformed = np.linalg.solve(L, y_obs)
        alpha0 = 0.001
        beta0 = 0.001
        
        y_bar = np.mean(y_transformed)
        SS = np.sum((y_transformed - y_bar)**2)
        
        alpha_n = alpha0 + n_obs / 2
        beta_n = beta0 + 0.5 * SS
        
        mu0 = self.mu_prior
        sigma0 = self.sigma_prior
        kappa0 = 1.0 / (sigma0**2)
        
        sigma2_est = beta_n / (alpha_n - 1) if alpha_n > 1 else beta_n / alpha0
        kappa_n = kappa0 + n_obs / sigma2_est
        mu_n = (kappa0 * mu0 + (n_obs / sigma2_est) * y_bar) / kappa_n
        
        self.posterior_params = {
            'mu_n': mu_n,
            'sigma_n': 1.0 / np.sqrt(kappa_n),
            'alpha_n': alpha_n,
            'beta_n': beta_n,
            'phi': phi,
            'W_obs': W_obs,
            'y_obs': y_obs
        }
        
        print(f"✔ Model dopasowany (Phi: {phi:.0f})")
        return self
    
    def _optimize_phi(self, y_obs):
        print("   ?? Optymalizacja phi dla modelu Gaussa...")
        max_dist = np.max(self.distance_matrix)
        if max_dist == 0: max_dist = 1000
        phi_values = np.logspace(np.log10(max(1.0, 0.01 * max_dist)), np.log10(max(10.0, 2.0 * max_dist)), 15)
        
        scores = []
        n_obs = len(y_obs)
        for phi in phi_values:
            try:
                D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
                W_obs = np.exp(-D_obs / phi) + np.eye(n_obs) * 1e-6
                sign, logdet = np.linalg.slogdet(W_obs)
                if sign <= 0:
                    scores.append(np.inf)
                    continue
                y_mean = np.mean(y_obs)
                residuals = y_obs - y_mean
                solved_res = np.linalg.solve(W_obs, residuals)
                rss = np.dot(residuals, solved_res)
                if rss <= 0:
                    scores.append(np.inf)
                    continue
                score = n_obs * np.log(rss) + logdet
                scores.append(score if np.isfinite(score) else np.inf)
            except:
                scores.append(np.inf)
        
        if np.all(np.isinf(scores)):
             best_phi = max_dist / 3
        else:
            best_phi = phi_values[np.nanargmin(scores)]
        print(f"   - Najlepsze phi: {best_phi:.0f}")
        return best_phi
    
    def predict(self, num_samples=1000, return_samples=False):
        if not hasattr(self, 'posterior_params'):
            raise ValueError("Model must be fitted first")
        n_total = len(self.space_points)
        n_obs = len(self.observed_indices)
        
        mu_samples = np.random.normal(loc=self.posterior_params['mu_n'], scale=self.posterior_params['sigma_n'], size=num_samples)
        sigma2_samples = stats.invgamma.rvs(a=self.posterior_params['alpha_n'], scale=self.posterior_params['beta_n'], size=num_samples)
        phi = self.posterior_params['phi']
        y_obs = self.posterior_params['y_obs']
        unobs_indices = np.setdiff1d(np.arange(n_total), self.observed_indices, assume_unique=True)
        n_unobs = len(unobs_indices)

        if n_unobs == 0:
            mean_pred = np.mean(np.tile(y_obs, (num_samples, 1)), axis=0)
            return (mean_pred, (y_obs, y_obs), np.tile(y_obs, (num_samples, 1))) if return_samples else (mean_pred, (y_obs, y_obs))

        W_obs = self.posterior_params['W_obs']
        D_obs_unobs = self.distance_matrix[np.ix_(self.observed_indices, unobs_indices)]
        W_obs_unobs = np.exp(-D_obs_unobs / phi)
        D_unobs = self.distance_matrix[np.ix_(unobs_indices, unobs_indices)]
        W_unobs = np.exp(-D_unobs / phi)

        try:
            L_obs = np.linalg.cholesky(W_obs)
            alpha = np.linalg.solve(L_obs.T, np.linalg.solve(L_obs, y_obs))
            v = np.linalg.solve(L_obs.T, np.linalg.solve(L_obs, np.ones(n_obs)))
            term1 = np.dot(W_obs_unobs.T, alpha)
            term2 = np.dot(W_obs_unobs.T, v)
            W_obs_inv_W_ou = np.linalg.solve(W_obs, W_obs_unobs)
            Cov_cond_norm = W_unobs - np.dot(W_obs_unobs.T, W_obs_inv_W_ou)
            Cov_cond_norm = 0.5 * (Cov_cond_norm + Cov_cond_norm.T) + np.eye(n_unobs) * 1e-6
        except:
            mean_pred = np.full(n_total, np.mean(y_obs))
            return mean_pred, (mean_pred, mean_pred), None

        predictions = np.zeros((num_samples, n_total))
        predictions[:, self.observed_indices] = np.tile(y_obs, (num_samples, 1))
        for i in range(num_samples):
            mu_cond = mu_samples[i] + (term1 - mu_samples[i] * term2)
            cov_cond = sigma2_samples[i] * Cov_cond_norm
            try:
                predictions[i, unobs_indices] = np.random.multivariate_normal(mu_cond, cov_cond, check_valid='warn', tol=1e-8)
            except:
                predictions[i, unobs_indices] = mu_cond
        
        mean_pred = np.mean(predictions, axis=0)
        lower_pred, upper_pred = np.percentile(predictions, [2.5, 97.5], axis=0)
        return (mean_pred, (lower_pred, upper_pred), predictions) if return_samples else (mean_pred, (lower_pred, upper_pred))
