"""This module contains the main program logic."""
from .TestMenager.test_manager import TestManager

def program(base_params, n_tests, csv_path, models_to_test=None, save_results=False, run_options={"rt":False,"it":False, "lt":False}, impact_models_to_test=None, k=7, lengthscale_list=None):
    """Main program function."""
    print("================================================")
    print("🎯 SYSTEM TESTOWANIA MODELI BAYESOWSKICH")
    print("================================================")
    
    if models_to_test is None:
        models_to_test = [
            ('lenk_adaptive', {}),
            ('bayesian', {
                'lengthscale': base_params['lengthscale'],
                'variance': base_params['variance'],
                'distance_unit': base_params.get('distance_unit', 'km')
            }),
            ('dirichlet', {}),
            ('spatial', {'smoothing_factor': 0.1})
        ]
    
    active = [name for name, params in models_to_test]
    print(f"TESTOWANE MODELE: {', '.join(active)}")
    print("================================================\n")
    
    test_manager = TestManager(csv_path)
    
    print("🎯 ETAP 1: STANDARDOWE TESTY")
    if run_options.get("rt", False):
        results = test_manager.run_tests(
            base_params, n_tests=n_tests, 
            models_to_test=models_to_test, save=save_results
        )
    
    print("\n🎯 ETAP 2: TEST WPŁYWU LICZBY OBSERWACJI")
    if run_options.get("it", False):
        if impact_models_to_test is None:
            impact_models_to_test = [
                ('lenk_adaptive', {}),
                ('bayesian', {
                    'lengthscale': base_params['lengthscale'],
                    'variance': base_params['variance'],
                    'distance_unit': base_params.get('distance_unit', 'km')
                }),
                ('dirichlet', {}),
                ('spatial', {'smoothing_factor': 0.1})
            ]
        
        active_impact = [name for name, params in impact_models_to_test]
        print(f"MODELE W TEŚCIE WPŁYWU: {', '.join(active_impact)}")
        
        test_manager.test_observation_length_impact(
            base_params, models_to_test=impact_models_to_test, k=k, save=save_results
        )
    
    print("\n🎯 ETAP 3: TEST WPŁYWU WARTOŚCI LENGTHSCALE")
    if run_options.get("lt", False):
        lengthscale_models_to_test = [
            ('bayesian', {
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
            }),
            ('logistic_normal_mcmc', {
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
            }),
        ]
        
        active_ls_impact = [name for name, params in lengthscale_models_to_test]
        print(f"MODELE W TEŚCIE WPŁYWU LENGTHSCALE: {', '.join(active_ls_impact)}")
        
        test_manager.test_lengthscale_impact(
            base_params, models_to_test=lengthscale_models_to_test, 
            lengthscale_list=lengthscale_list, k=k, save=save_results
        )

    print("\n================================================")
    print("✅ WSZYSTKIE TESTY ZAKOŃCZONE")
    print("================================================")
