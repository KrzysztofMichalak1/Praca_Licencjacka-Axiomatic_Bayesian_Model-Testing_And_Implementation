"""This module contains the main program logic."""
from .TestMenager.test_manager import TestManager

def program(base_params, csv_path, n_tests=1, models_to_test=None, save_results=False, run_options={"rt":False,"it":False, "lt":False, "nt":False}, impact_models_to_test=None, k=15, lengthscale_list=None, db_path="wyniki_testow.csv", visualise=False, nemenyi_params=None, **kwargs):
    """Main program function."""
    print("================================================")
    print("🎯 SYSTEM TESTOWANIA MODELI BAYESOWSKICH")
    print("================================================")
    
    if models_to_test is None:
        models_to_test = [
            ('Model_Lenka', {}),
            ('Model_Aksjomatyczny-preparamed', {
                'lengthscale': base_params.get('lengthscale', 1000),
                'variance': base_params.get('variance', 1.0),
                'distance_unit': base_params.get('distance_unit', 'km')
            }),
            ('Model_Dirichleta', {}),
            ('Model_wygładzania_przestrzennego', {'smoothing_factor': 0.1})
        ]
    
    active = [name for name, params in models_to_test]
    print(f"TESTOWANE MODELE: {', '.join(active)}")
    print("================================================\n")
    
    test_manager = TestManager(csv_path, db_path=db_path)
    
    if run_options.get("rt", False):
        print("🎯 ETAP 1: STANDARDOWE TESTY")
        test_manager.run_tests(
            base_params, n_tests=n_tests, 
            models_to_test=models_to_test, save=save_results
        )
    
    if run_options.get("it", False):
        print("\n🎯 ETAP 2: TEST WPŁYWU LICZBY OBSERWACJI")
        if impact_models_to_test is None:
            impact_models_to_test = models_to_test # Use main models if not specified
        test_manager.test_observation_length_impact(
            base_params, models_to_test=impact_models_to_test, k=k, save=save_results
        )
    
    if run_options.get("lt", False):
        print("\n🎯 ETAP 3: TEST WPŁYWU WARTOŚCI LENGTHSCALE")
        lengthscale_models_to_test = [
            ('Model_Aksjomatyczny-preparamed', {
                'variance': 1.0, 'distance_unit': "km",
                'mcmc_samples': 5000, 'mcmc_burn': 3000, 'mcmc_scale': 0.05,
            }),
            ('Model_Lenka-preparamed', {
                'variance': 1.0, 'distance_unit': "km",
                'mcmc_samples': 5000, 'mcmc_burn': 3000, 'mcmc_scale': 0.05,
            }),
        ]
        test_manager.test_lengthscale_impact(
            base_params, models_to_test=lengthscale_models_to_test, 
            lengthscale_list=lengthscale_list, k=k, save=save_results
        )

    if run_options.get("nt", False):
        print("\n🎯 ETAP 4: ANALIZA NEMENYI")
        n_obs = nemenyi_params.get('n_observations', 1000) if nemenyi_params else 1000
        n_k = nemenyi_params.get('k', 10) if nemenyi_params else 10
        test_manager.test_nemenei(
            base_params, models_to_test=models_to_test, 
            n_observations=n_obs, k=n_k, save=save_results
        )

    print("\n================================================")
    print("✅ WSZYSTKIE TESTY ZAKOŃCZONE")
    print("================================================")
