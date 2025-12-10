"""This module contains the DirichletModel class."""
import numpy as np
from scipy.stats import dirichlet

class DirichletModel:
    """Simple Dirichlet model as a baseline."""
    
    def __init__(self, observed_indices, n_points):
        print("▶ [DIRICHLET] Inicjalizacja modelu Dirichleta...")
        self.observed_indices = np.asarray(observed_indices)
        self.n = n_points
        self.counts = np.bincount(self.observed_indices, minlength=self.n)
        self.alpha = self.counts + 1  # Laplace smoothing
        print(f"  - Liczba punktów: {self.n}")
        print(f"  - Suma counts: {self.counts.sum()}")
        print(f"  - Parametry alpha: {self.alpha}")
        
    def posterior_mean(self):
        """Dirichlet posterior mean."""
        return self.alpha / self.alpha.sum()
    
    def sample_posterior(self, n_samples=1000):
        """Samples from the posterior Dirichlet distribution."""
        return dirichlet.rvs(self.alpha, size=n_samples)
    
    def posterior_quantiles(self, q=(0.025, 0.975)):
        """Posterior quantiles (approximate)."""
        samples = self.sample_posterior(1000)
        return np.quantile(samples, q, axis=0)
