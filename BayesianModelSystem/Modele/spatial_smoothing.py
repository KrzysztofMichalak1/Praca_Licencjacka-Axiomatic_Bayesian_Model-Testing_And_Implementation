"""This module contains the BayesianSpatialSmoothing class."""
import numpy as np
from ..Wczytywanie_danych.metric import haversine

class BayesianSpatialSmoothing:
    """
    Prosty, nieparametryczny model wygładzania przestrzennego, działający na
    zasadzie wygładzania jądrowego (kernel smoothing). Nie jest to model w pełni
    Bayesowski w sensie estymacji `posterior`, ale raczej heurystyka oparta na
    dystansie.

    Algorytm działania:
    1. Inicjalizacja:
       - Model przyjmuje geometrię punktów, obserwowane indeksy (`obs_idx`) oraz
         współczynnik wygładzania (`smoothing_factor`), który pełni rolę
         podobną do `lengthscale`.
       - Zlicza obserwacje i oblicza empiryczne prawdopodobieństwa (`counts / total_obs`).

    2. Obliczenie Predykcji (`posterior_mean`):
       - Obliczana jest macierz wag (kernel) na podstawie odległości między
         wszystkimi punktami i współczynnika `smoothing_factor`. Waga między
         dwoma punktami jest tym większa, im są one bliżej siebie (np. `exp(-d/phi)`).
       - Predykcja dla każdego punktu na siatce jest obliczana jako iloczyn
         macierzowy macierzy wag i wektora empirycznych prawdopodobieństw. W efekcie,
         prawdopodobieństwo w każdym punkcie jest "rozmywane" na jego otoczenie.
       - Wynikowy wektor jest normalizowany tak, aby suma prawdopodobieństw
         wynosiła 1, tworząc ostateczny rozkład prawdopodobieństwa.
    """
    
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
