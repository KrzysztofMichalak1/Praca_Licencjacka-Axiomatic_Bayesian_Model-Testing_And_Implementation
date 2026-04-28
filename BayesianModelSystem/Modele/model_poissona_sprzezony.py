import numpy as np
from scipy import stats
import warnings
from .model_bazowy_przestrzenny import BayesianSpatialModel

class ModelPoissonaSprzezony(BayesianSpatialModel):
    """
    Przestrzenny model Poissona z priorami Gamma (sprzężonymi).
    Przeznaczony dla danych w postaci zliczeń.
    """
    
    def __init__(self, space_points, metric_func, observed_indices,
                 counts, N=None, distance_unit='km',
                 alpha_prior=0.5, beta_prior=0.5, smoothing_strength=1.0):
        super().__init__(space_points, metric_func, observed_indices,
                        counts, N, distance_unit)
        
        if counts is None:
            raise ValueError("'counts' must be provided for Poisson model")
        
        self.exposure = np.ones(len(space_points)) if N is None else N
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.smoothing_strength = smoothing_strength
        self.y_counts = self.counts[self.observed_indices]
        self.y_exposure = self.exposure[self.observed_indices]
        
        print(f"   - Model: Model Poissona (sprzężony)")
        print(f"   - Prior: l ~ Gamma({alpha_prior}, {beta_prior})")
    
    def fit(self, phi=4, optimize_phi=True):
        print(f"\n▶ [POISSON-CONJUGATE] Dopasowywanie modelu...")
        n_obs = len(self.observed_indices)
        
        if phi is None and optimize_phi:
            phi = self._optimize_phi_poisson()
        elif phi is None:
            phi = np.max(self.distance_matrix) / 3
            print(f"   - Phi (heurystyka): {phi:.0f}")
        
        D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
        W_obs = np.exp(-D_obs / phi)
        
        alpha_n = np.zeros(n_obs)
        beta_n = np.zeros(n_obs)
        for i in range(n_obs):
            weights = W_obs[i, :]
            neighbor_counts = np.sum(weights * self.y_counts)
            neighbor_exposure = np.sum(weights * self.y_exposure)
            neighbor_rate = neighbor_counts / neighbor_exposure if neighbor_exposure > 1e-9 else (np.sum(self.y_counts) / np.sum(self.y_exposure) if np.sum(self.y_exposure) > 1e-9 else 1.0)
            spatial_pseudo_counts = self.smoothing_strength * neighbor_rate
            spatial_pseudo_exposure = self.smoothing_strength
            alpha_n[i] = self.alpha_prior + self.y_counts[i] + spatial_pseudo_counts
            beta_n[i] = self.beta_prior + self.y_exposure[i] + spatial_pseudo_exposure
        
        self.posterior_params = {
            'alpha_n': alpha_n, 'beta_n': beta_n, 'phi': phi,
            'W_obs': W_obs, 'y_counts': self.y_counts, 'y_exposure': self.y_exposure
        }
        print(f"✔ Model dopasowany (Phi: {phi:.0f})")
        return self
    
    def _optimize_phi_poisson(self):
        print("   ?? Optymalizacja phi dla modelu Poissona...")
        max_dist = np.max(self.distance_matrix)
        if max_dist == 0: max_dist = 1000
        phi_values = np.logspace(np.log10(max(1.0, 0.01 * max_dist)), np.log10(max(10.0, 2.0 * max_dist)), 15)
        
        scores = []
        for phi in phi_values:
            try:
                D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
                W_obs = np.exp(-D_obs / phi)
                log_likelihood = 0
                for i in range(len(self.y_counts)):
                    weights = W_obs[i, :]
                    sum_weights = np.sum(weights)
                    lambda_i = max(np.sum(weights * self.y_counts) / sum_weights if sum_weights > 1e-9 else np.mean(self.y_counts), 1e-9)
                    log_likelihood += (self.y_counts[i] * np.log(lambda_i) - lambda_i * self.y_exposure[i])
                scores.append(-log_likelihood if np.isfinite(log_likelihood) else np.inf)
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
        
        lambda_samples_obs = np.zeros((num_samples, n_obs))
        for i in range(n_obs):
            lambda_samples_obs[:, i] = stats.gamma.rvs(
                a=self.posterior_params['alpha_n'][i],
                scale=1.0 / self.posterior_params['beta_n'][i],
                size=num_samples
            )
        
        phi = self.posterior_params['phi']
        predictions = np.zeros((num_samples, n_total))
        for j in range(n_total):
            distances = self.distance_matrix[j, self.observed_indices]
            weights = np.exp(-distances / phi)
            weights = weights / np.sum(weights)
            predictions[:, j] = np.dot(lambda_samples_obs, weights)
        
        mean_pred = np.mean(predictions, axis=0)
        lower_pred, upper_pred = np.percentile(predictions, [2.5, 97.5], axis=0)
        return (mean_pred, (lower_pred, upper_pred), predictions) if return_samples else (mean_pred, (lower_pred, upper_pred))
    
    def predict_counts(self, num_samples=1000):
        mean_lambda, ci_lambda, samples_lambda = self.predict(num_samples, return_samples=True)
        count_samples = np.random.poisson(samples_lambda * self.exposure)
        mean_counts = np.mean(count_samples, axis=0)
        lower_counts, upper_counts = np.percentile(count_samples, [2.5, 97.5], axis=0)
        return mean_counts, (lower_counts, upper_counts), count_samples
