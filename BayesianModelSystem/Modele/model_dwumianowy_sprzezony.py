import numpy as np
from scipy import stats
import warnings
from .model_bazowy_przestrzenny import BayesianSpatialModel

class ModelDwumianowySprzezony(BayesianSpatialModel):
    """
    Przestrzenny model dwumianowy z priorami Beta (sprzężonymi).
    Przeznaczony dla danych o liczbie sukcesów i prób.
    """
    
    def __init__(self, space_points, metric_func, observed_indices,
                 counts, N, distance_unit='km',
                 alpha_prior=0.5, beta_prior=0.5, smoothing_strength=0.1):
        super().__init__(space_points, metric_func, observed_indices,
                        counts, N, distance_unit)
        
        if counts is None or N is None:
            raise ValueError("Both 'counts' and 'N' must be provided for Binomial model")
        
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.smoothing_strength = smoothing_strength
        self.y_success = self.counts[self.observed_indices]
        self.y_trials = self.N[self.observed_indices]
        self.y_failure = self.y_trials - self.y_success
        
        print(f"   - Model: Model Dwumianowy (sprzężony)")
        print(f"   - Prior: p ~ Beta({alpha_prior}, {beta_prior})")
    
    def fit(self, phi=4, optimize_phi=True):
        print(f"\n▶ [BINOMIAL-CONJUGATE] Dopasowywanie modelu...")
        n_obs = len(self.observed_indices)
        
        if phi is None and optimize_phi:
            phi = self._optimize_phi_binomial()
        elif phi is None:
            phi = np.max(self.distance_matrix) / 3
            print(f"   - Phi (heurystyka): {phi:.0f}")
        
        D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
        W_obs = np.exp(-D_obs / phi)
        
        alpha_n = np.zeros(n_obs)
        beta_n = np.zeros(n_obs)
        for i in range(n_obs):
            weights = W_obs[i, :]
            weighted_success_sum = np.sum(weights * self.y_success)
            weighted_trials_sum = np.sum(weights * self.y_trials)
            p_hat_i = np.clip(weighted_success_sum / weighted_trials_sum if weighted_trials_sum > 0 else (np.mean(self.y_success) / np.mean(self.y_trials) if np.mean(self.y_trials) > 0 else 0.5), 1e-9, 1 - 1e-9)
            spatial_alpha = self.smoothing_strength * p_hat_i
            spatial_beta = self.smoothing_strength * (1 - p_hat_i)
            alpha_n[i] = self.alpha_prior + self.y_success[i] + spatial_alpha
            beta_n[i] = self.beta_prior + self.y_failure[i] + spatial_beta
        
        self.posterior_params = {
            'alpha_n': alpha_n, 'beta_n': beta_n, 'phi': phi,
            'W_obs': W_obs, 'y_success': self.y_success, 'y_trials': self.y_trials
        }
        print(f"✔ Model dopasowany (Phi: {phi:.0f})")
        return self
    
    def _optimize_phi_binomial(self):
        print("   ?? Optymalizacja phi dla modelu dwumianowego...")
        max_dist = np.max(self.distance_matrix)
        if max_dist == 0: max_dist = 1000
        phi_values = np.logspace(np.log10(max(1.0, 0.01 * max_dist)), np.log10(max(10.0, 2.0 * max_dist)), 15)
        
        scores = []
        for phi in phi_values:
            try:
                D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
                W_obs = np.exp(-D_obs / phi)
                log_likelihood = 0
                for i in range(len(self.y_success)):
                    w = W_obs[i, :]
                    ws = np.sum(w * self.y_success)
                    wt = np.sum(w * self.y_trials)
                    p_i = np.clip(ws / wt if wt > 1e-9 else 0.5, 1e-9, 1 - 1e-9)
                    log_likelihood += (self.y_success[i] * np.log(p_i) + self.y_failure[i] * np.log(1 - p_i))
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
        
        p_samples_obs = np.zeros((num_samples, n_obs))
        for i in range(n_obs):
            p_samples_obs[:, i] = stats.beta.rvs(
                a=self.posterior_params['alpha_n'][i],
                b=self.posterior_params['beta_n'][i],
                size=num_samples
            )
        
        phi = self.posterior_params['phi']
        predictions = np.zeros((num_samples, n_total))
        for j in range(n_total):
            distances = self.distance_matrix[j, self.observed_indices]
            weights = np.exp(-distances / phi)
            weights = weights / np.sum(weights)
            predictions[:, j] = np.dot(p_samples_obs, weights)
        
        # Normalizacja: każdy wiersz (próbka) musi sumować się do 1
        row_sums = predictions.sum(axis=1, keepdims=True)
        predictions = np.divide(predictions, row_sums, out=np.zeros_like(predictions), where=row_sums!=0)
        
        mean_pred = np.mean(predictions, axis=0)
        lower_pred, upper_pred = np.percentile(predictions, [2.5, 97.5], axis=0)
        return (mean_pred, (lower_pred, upper_pred), predictions) if return_samples else (mean_pred, (lower_pred, upper_pred))
    
    def predict_counts(self, num_samples=1000):
        mean_p, ci_p, samples_p = self.predict(num_samples, return_samples=True)
        success_samples = np.zeros((num_samples, len(self.space_points)))
        for i in range(num_samples):
            for j in range(len(self.space_points)):
                if j < len(self.N):
                    success_samples[i, j] = np.random.binomial(n=self.N[j], p=samples_p[i, j])
        mean_success = np.mean(success_samples, axis=0)
        lower_success, upper_success = np.percentile(success_samples, [2.5, 97.5], axis=0)
        return mean_success, (lower_success, upper_success), success_samples
