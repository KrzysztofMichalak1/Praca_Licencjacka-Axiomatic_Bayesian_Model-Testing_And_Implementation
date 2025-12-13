"""This module contains the main program logic."""
from .TestMenager.test_manager import TestManager

def program(base_params, n_tests, csv_path, models_to_test=None, save_results=False, run_options={"rt":False,"it":False}, impact_models_to_test=None, k=10):
    """Main program function."""
    print("================================================")
    print("🎯 SYSTEM TESTOWANIA MODELI BAYESOWSKICH")
    print("================================================")
    
    if models_to_test is None:
        models_to_test = [
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
    if run_options["rt"]:
        results = test_manager.run_tests(
            base_params, n_tests=n_tests, 
            models_to_test=models_to_test, save=save_results
        )
    
    print("\n🎯 ETAP 2: TEST WPŁYWU LICZBY OBSERWACJI")
    if run_options["it"]:
        if impact_models_to_test is None:
            impact_models_to_test = [
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
    
    print("\n================================================")
    print("✅ WSZYSTKIE TESTY ZAKOŃCZONE")
    print("================================================")
