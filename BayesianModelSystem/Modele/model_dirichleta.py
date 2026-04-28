import numpy as np
from scipy.stats import dirichlet

class ModelDirichleta:
    """
    Model bazujący na rozkładzie Dirichleta (sprzężony prior dla Multinomial).
    Prosty model bazowy bez korelacji przestrzennych.
    """
    def __init__(self, observed_indices, n_points):
        print("▶ [MODEL-DIRICHLETA] Inicjalizacja modelu...")
        self.observed_indices = np.asarray(observed_indices)
        self.n = n_points
        self.counts = np.bincount(self.observed_indices, minlength=self.n)
        self.alpha = self.counts + 1  # Laplace smoothing
        print(f"  - Liczba punktów: {self.n}, Obserwacje: {self.counts.sum()}")
        
    def posterior_mean(self):
        return self.alpha / self.alpha.sum()
    
    def sample_posterior(self, n_samples=1000):
        return dirichlet.rvs(self.alpha, size=n_samples)
    
    def posterior_quantiles(self, q=(0.025, 0.975)):
        samples = self.sample_posterior(1000)
        return np.quantile(samples, q, axis=0)
