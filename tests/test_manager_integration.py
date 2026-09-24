import pytest
import numpy as np
import os
from BayesianModelSystem.TestMenager.test_manager import TestManager

MODELS_TO_TEST = [
    pytest.param('Model_Dirichleta', {}, id="dirichlet"),
    pytest.param('Model_wygładzania_przestrzennego', {'smoothing_factor': 0.1}, id="spatial_smoothing"),
    pytest.param('Model_Gausowski_sprzężony', {'phi': 100, 'optimize_phi': False}, id="spatial_gaussian"),
    pytest.param('Model_Dwumianowy_sprzężony', {'phi': 100, 'optimize_phi': False}, id="spatial_binomial"),
    pytest.param('Model_Poissona_sprzężony', {'phi': 100, 'optimize_phi': False}, id="spatial_poisson"),
    pytest.param('Model_Aksjomatyczny-preparamed', {
        'lengthscale': 500, 'variance': 1.0,
        'mcmc_samples': 50, 'mcmc_burn': 20, 'mcmc_scale': 0.05
    }, id="bayesian_mcmc"),
    pytest.param('Model_Lenka-preparamed', {
        'lengthscale': 500, 'variance': 1.0,
        'mcmc_samples': 50, 'mcmc_burn': 20, 'mcmc_scale': 0.05
    }, id="logistic_normal_mcmc"),
    pytest.param('Model_Lenka', {
        'start_ls': 1000, 'step_size': 1000, 'k_steps': 1,
        'num_samples_search': 10, 'burn_in_search': 5,
        'mcmc_samples': 20, 'mcmc_burn': 10,
    }, id="lenk_adaptive"),
]

@pytest.mark.parametrize("model_name, model_params", MODELS_TO_TEST)
def test_manager_integration_for_each_model(model_name, model_params, tmp_path):
    real_csv_path = "Global_2020_MarineSpeciesRichness_AquaMaps (4).csv"
    if not os.path.exists(real_csv_path):
        pytest.skip(f"Plik '{real_csv_path}' nie znaleziony.")

    base_params = {'cutoff_km': 100, 'co_ktory': 1500, 'n_observations': 50}
    test_manager = TestManager(csv_path=real_csv_path, db_path=str(tmp_path / "test.csv"))
    
    result = test_manager.test(test_params=base_params, models_to_test=[(model_name, model_params)], save=False, visualise=False)
    
    assert result.get('success') is True
    unique_model_key = f"{model_name}_0"
    assert unique_model_key in result['metrics']
    assert 'MSE' in result['metrics'][unique_model_key]
