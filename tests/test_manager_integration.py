#
# Plik: test_manager_integration.py
#
# Opis:
# Ten plik zawiera test integracyjny dla klasy `TestManager`.
#
# Cel testu:
# Sprawdzenie, czy główna pętla `TestManager.test()` jest w stanie wykonać
# kompletny cykl testowy dla każdego zaimplementowanego modelu, używając
# rzeczywistego zbioru danych. Weryfikuje to, czy każdy model poprawnie
# integruje się z `TestManager`.
#
# Metodologia:
# - Test jest parametryzowany za pomocą `pytest.mark.parametrize`, aby iterować
#   po wszystkich zdefiniowanych modelach.
# - Używa rzeczywistego pliku CSV: "Global_2020_MarineSpeciesRichness_AquaMaps (4).csv".
# - Dla każdego modelu uruchamiany jest `TestManager` z uproszczonymi parametrami
#   (np. mniejsza liczba próbek MCMC), aby test był wykonalny czasowo.
# - Asercje sprawdzają, czy proces zakończył się z flagą `success: True`
#   i czy zwrócony obiekt z wynikami ma oczekiwaną strukturę.
#

import pytest
import numpy as np
import os
import pandas as pd
from BayesianModelSystem.TestMenager.test_manager import TestManager

# Definicja modeli i ich parametrów do testowania
# Używamy mniejszej liczby próbek MCMC, aby testy były szybsze.
MODELS_TO_TEST = [
    pytest.param('dirichlet', {}, id="dirichlet"),
    pytest.param('spatial', {'smoothing_factor': 0.1}, id="spatial_smoothing"),
    pytest.param('spatial_gaussian', {'phi': 100, 'optimize_phi': False}, id="spatial_gaussian"),
    pytest.param('spatial_binomial', {'phi': 100, 'optimize_phi': False}, id="spatial_binomial"),
    pytest.param('spatial_poisson', {'phi': 100, 'optimize_phi': False}, id="spatial_poisson"),
    pytest.param('bayesian', {
        'lengthscale': 500, 'variance': 1.0,
        'mcmc_samples': 50, 'mcmc_burn': 20, 'mcmc_scale': 0.05
    }, id="bayesian_mcmc"),
    pytest.param('logistic_normal_mcmc', {
        'lengthscale': 500, 'variance': 1.0,
        'mcmc_samples': 50, 'mcmc_burn': 20, 'mcmc_scale': 0.05
    }, id="logistic_normal_mcmc"),
    pytest.param('lenk_adaptive', {
        'start_ls': 1000, 'step_size': 1000, 'k_steps': 1,
        'num_samples_search': 10, 'burn_in_search': 5,
        'mcmc_samples': 20, 'mcmc_burn': 10,
    }, id="lenk_adaptive"),
    pytest.param('gaussian', {
        'lengthscale_prior': (1000, 500), 'variance_prior': (2, 1),
        'n_samples': 50, 'burn_in': 20
    }, id="bayesian_gp"),
]

@pytest.mark.parametrize("model_name, model_params", MODELS_TO_TEST)
def test_manager_integration_for_each_model(model_name, model_params, tmp_path):
    """
    Integration smoke test for TestManager that runs for each specified model.
    It uses the real data file to ensure models work with the actual data format.
    """
    # 1. Define path to the real data
    real_csv_path = "Global_2020_MarineSpeciesRichness_AquaMaps (4).csv"
    
    # Sprawdź, czy plik z danymi istnieje, w przeciwnym razie pomiń test
    if not os.path.exists(real_csv_path):
        pytest.skip(f"Plik z danymi '{real_csv_path}' nie został znaleziony.")

    # 2. Define minimal parameters for the test run
    base_params = {
        'cutoff_km': 100,
        'co_ktory': 1500,  # Użyj co 50-tego punktu, aby przyspieszyć
        'n_observations': 50
    }
    
    models_to_test = [(model_name, model_params)]
    
    # 3. Instantiate and run TestManager
    db_path = tmp_path / f"test_results_{model_name}.csv"
    test_manager = TestManager(csv_path=real_csv_path, db_path=str(db_path))
    
    print(f"\n--- Uruchamianie TestManager dla modelu: {model_name} ---")
    result = test_manager.test(
        test_params=base_params,
        models_to_test=models_to_test,
        save=False,
        visualise=False
    )
    
    # 4. Assert that the run was successful and produced results
    assert result is not None, "TestManager.test() powinien zwrócić obiekt wyniku"
    assert result.get('success') is True, f"Flaga 'success' w wyniku powinna być True dla modelu {model_name}"
    
    assert 'metrics' in result, "Wynik powinien zawierać 'metrics'"
    
    unique_model_key = f"{model_name}_0"
    assert unique_model_key in result['metrics'], f"Metryki dla modelu '{unique_model_key}' powinny być obecne"
    
    metrics = result['metrics'][unique_model_key]
    assert 'MSE' in metrics, "Metryki powinny zawierać 'MSE'"
    assert 'MAE' in metrics, "Metryki powinny zawierać 'MAE'"
    assert np.isfinite(metrics['MSE']), f"MSE dla modelu {model_name} powinno być skończoną liczbą"