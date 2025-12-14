#
# Plik: test_models.py
#
# Opis:
# Ten plik zawiera "smoke tests" dla poszczególnych modeli statystycznych.
#
# Cel testów:
# Celem nie jest weryfikacja numerycznej poprawności predykcji (co jest trudne
# i niestabilne dla modeli probabilistycznych), ale upewnienie się, że każdy
# model można poprawnie zainicjalizować, wytrenować (`fit`/`predict`) i że
# zwraca on wyniki w oczekiwanym formacie (właściwy kształt, typ danych,
# prawidłowy zakres wartości, np. suma p-stw równa 1).
# Testy te działają jak zabezpieczenie przed regresją, np. zmianami w kodzie,
# które powodują, że model przestaje działać.
#
# Metodologia:
# - Używana jest mała, stała próbka danych (`small_dataset` fixture), aby testy
#   były szybkie i powtarzalne.
# - Sprawdzane są podstawowe właściwości wyniku, takie jak kształt, suma i znak wartości.
#
# Testowane modele:
# - `DirichletModel`
# - `GaussianSpatialModelConjugate`
#

import pytest
import numpy as np
from BayesianModelSystem.Wczytywanie_danych.metric import haversine
from BayesianModelSystem.Modele.dirichlet_model import DirichletModel
from BayesianModelSystem.Modele.bayesian_spatial import GaussianSpatialModelConjugate

# Fixture to provide a consistent, small dataset for tests
@pytest.fixture(scope="module")
def small_dataset():
    """
    Provides a small, consistent dataset for model smoke tests.
    A 4x4 grid of 16 points.
    """
    side = 4
    n_points = side * side
    space_points = []
    for i in range(side):
        for j in range(side):
            space_points.append((i * 10.0, j * 10.0)) # 10km separation
    
    # 20 observations spread across a few points
    observed_indices = np.array([2, 3, 3, 5, 5, 5, 6, 7, 9, 9, 10, 10, 11, 13, 14, 14, 14, 14, 15, 15])
    
    return {
        "space_points": space_points,
        "n_points": n_points,
        "observed_indices": observed_indices
    }

def test_dirichlet_model_smoke(small_dataset):
    """
    Smoke test for DirichletModel. Checks if it runs and output has correct format.
    """
    n_points = small_dataset["n_points"]
    obs_idx = small_dataset["observed_indices"]
    
    # Inicjalizacja i predykcja
    model = DirichletModel(obs_idx, n_points)
    pred = model.posterior_mean()
    
    # Asercje
    assert pred.shape == (n_points,), "Output shape should match number of points"
    assert np.all(pred >= 0), "Probabilities must be non-negative"
    assert np.isclose(np.sum(pred), 1.0), "Probabilities should sum to 1"

def test_gaussian_spatial_model_smoke(small_dataset):
    """
    Smoke test for GaussianSpatialModelConjugate.
    """
    space_points = small_dataset["space_points"]
    n_points = small_dataset["n_points"]
    obs_idx = small_dataset["observed_indices"]

    # Przygotowanie danych (jak w test_manager.py)
    observed_counts = np.bincount(obs_idx, minlength=n_points)
    total_observations = len(obs_idx)
    empirical_probs = observed_counts / total_observations
    
    unique_indices = np.where(observed_counts > 0)[0]
    observed_values = empirical_probs[unique_indices]

    # Parametry modelu
    constructor_params = {
        'space_points': space_points, 
        'metric_func': haversine, 
        'observed_indices': unique_indices,
        'counts': observed_values,
        'distance_unit': 'km',
        'mu_prior': np.mean(observed_values),
        'sigma_prior': np.std(observed_values) if np.std(observed_values) > 0 else 1.0
    }
    
    # Inicjalizacja, trening i predykcja
    # Używamy małego `phi` i wyłączamy optymalizację, żeby test był szybki
    model = GaussianSpatialModelConjugate(**constructor_params)
    model.fit(phi=100, optimize_phi=False)
    pred, (lower, upper) = model.predict(num_samples=10) # Mało próbek dla szybkości
    
    # Asercje
    assert pred.shape == (n_points,), "Mean prediction shape should match number of points"
    assert lower.shape == (n_points,), "Lower CI shape should match number of points"
    assert upper.shape == (n_points,), "Upper CI shape should match number of points"
    assert np.all(np.isfinite(pred)), "All prediction values should be finite"
    
    # Sprawdźmy, czy predykcje dla obserwowanych punktów są równe wejściowym
    # (zgodnie z logiką modelu)
    pred_observed_values = pred[unique_indices]
    assert np.allclose(pred_observed_values, observed_values), "Predictions for observed points should match input"

