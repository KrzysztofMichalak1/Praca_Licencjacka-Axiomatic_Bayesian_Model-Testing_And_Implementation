import numpy as np
from ..Wczytywanie_danych.metric import haversine

class ModelWygladzaniaPrzestrzennego:
    """
    Prosty, nieparametryczny model wygładzania przestrzennego (kernel smoothing).
    """
    def __init__(self, space_points, observed_indices, smoothing_factor=0.1):
        self.space_points = np.array(space_points)
        self.observed_indices = np.array(observed_indices)
        self.n_points = len(space_points)
        self.smoothing_factor = smoothing_factor
        self.counts = np.bincount(observed_indices, minlength=self.n_points)
        self.total_obs = self.counts.sum()
        
        print(f"▶ [MODEL-WYGLADZANIA-PRZESTRZENNEGO] Inicjalizacja:")
        print(f"  - Punkty: {self.n_points}, Obserwacje: {self.total_obs}")
        print(f"  - Smoothing factor: {self.smoothing_factor}")
    
    def compute_spatial_weights(self):
        weights = np.zeros((self.n_points, self.n_points))
        for i in range(self.n_points):
            for j in range(self.n_points):
                if i == j: weights[i, j] = 1.0
                else:
                    dist = haversine(self.space_points[i], self.space_points[j])
                    weights[i, j] = np.exp(-dist / (self.smoothing_factor * 1000))
        weights /= weights.sum(axis=1, keepdims=True)
        return weights
    
    def posterior_mean(self):
        empirical_probs = self.counts / self.total_obs
        spatial_weights = self.compute_spatial_weights()
        smoothed_probs = spatial_weights @ empirical_probs
        return smoothed_probs / smoothed_probs.sum()
