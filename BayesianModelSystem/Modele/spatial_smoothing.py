"""This module contains the BayesianSpatialSmoothing class."""
import numpy as np
from ..Wczytywanie_danych.metric import haversine

class BayesianSpatialSmoothing:
    """Simple Bayesian spatial smoothing model."""
    
    def __init__(self, space_points, observed_indices, smoothing_factor=0.1):
        self.space_points = np.array(space_points)
        self.observed_indices = np.array(observed_indices)
        self.n_points = len(space_points)
        self.smoothing_factor = smoothing_factor
        
        # Zlicz obserwacje
        self.counts = np.bincount(observed_indices, minlength=self.n_points)
        self.total_obs = self.counts.sum()
        
        print(f"▶ [SPATIAL SMOOTHING] Inicjalizacja:")
        print(f"  - Punkty: {self.n_points}, Obserwacje: {self.total_obs}")
        print(f"  - Smoothing factor: {self.smoothing_factor}")
    
    def compute_spatial_weights(self):
        """Computes spatial weights based on distance."""
        print("▶ [SPATIAL SMOOTHING] Obliczanie wag przestrzennych...")
        weights = np.zeros((self.n_points, self.n_points))
        
        for i in range(self.n_points):
            for j in range(self.n_points):
                if i == j:
                    weights[i, j] = 1.0
                else:
                    dist = haversine(self.space_points[i], self.space_points[j])
                    # Waga maleje wykładniczo z odległością
                    weights[i, j] = np.exp(-dist / (self.smoothing_factor * 1000))
        
        # Normalizuj wagi wierszowo
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = weights / row_sums
        
        print(f"  - Min waga: {np.min(weights):.4f}")
        print(f"  - Max waga: {np.max(weights):.4f}")
        print(f"  - Średnia waga: {np.mean(weights):.4f}")
        
        return weights
    
    def posterior_mean(self):
        """Posterior mean with spatial smoothing."""
        # Empiryczne prawdopodobieństwa
        empirical_probs = self.counts / self.total_obs
        
        print(f"  - Empiryczne probs - min: {np.min(empirical_probs):.2e}, max: {np.max(empirical_probs):.2e}")
        
        # Macierz wag przestrzennych
        spatial_weights = self.compute_spatial_weights()
        
        # Wygładzone prawdopodobieństwa
        smoothed_probs = spatial_weights @ empirical_probs
        
        # Normalizacja
        smoothed_probs = smoothed_probs / smoothed_probs.sum()
        
        print(f"  - Wygładzone probs - min: {np.min(smoothed_probs):.2e}, max: {np.max(smoothed_probs):.2e}")
        print(f"  - Suma wygładzonych: {smoothed_probs.sum():.6f}")
        
        return smoothed_probs
