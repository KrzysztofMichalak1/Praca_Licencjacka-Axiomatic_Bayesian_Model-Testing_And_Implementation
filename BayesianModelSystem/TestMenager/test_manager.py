"This module contains the TestManager class for running tests."
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..Wczytywanie_danych.preprocessing import przygotuj_dane

from ..Wczytywanie_danych.loaders import losuj_obserwacje

from ..Wyniki.results_db import ResultsDatabase

from ..Modele import (
    BayesianFieldModel, 
    BayesianFieldModelAdaptiveSearchBinary,
    BayesianFieldModelCVGridSearch,
    LogisticNormalMCMC,
    LenkAdaptiveSearchModel,
    DirichletModel, 
    BayesianGaussianProcess, 
    BayesianSpatialSmoothing,
    SpatialBinomialConjugate,
    GaussianSpatialModelConjugate,
    SpatialPoissonConjugate
)

from ..Pomocnicze.metrics import oblicz_metryki

from ..Pomocnicze.visualization import stworz_mape_porownawcza, pokaz_punkt_referencyjny

from ..Wczytywanie_danych.metric import haversine

class TestManager:
    """Manager for running multiple tests."""
    
    __test__ = False
    
    def __init__(self, csv_path, db_path="wyniki_testow.csv"):
        self.csv_path = csv_path
        self.db = ResultsDatabase(db_path)
        self.results = []
        self.cached_data = None
    
    def prepare_data_once(self, cutoff_km, co_ktory, max_observations):
        """Prepares data once and caches it."""
        if self.cached_data is None:
            print("▶ [DATA] Pierwsze przygotowanie danych...")
            gdf, _, true_probs = przygotuj_dane(
                self.csv_path, cutoff_km, co_ktory, max_observations
            )
            self.cached_data = {
                'gdf': gdf,
                'true_probs': true_probs,
                'points': list(zip(gdf["Longitude"], gdf["Latitude"]))
            }
            print(f"  ✔ Dane zapisane w cache: {len(gdf)} punktów")
        else:
            print("▶ [DATA] Używam danych z cache")
        
        return self.cached_data
    
    def test(self, test_params, test_number=1, total_tests=1, 
             models_to_test=None, save=False,visualise=False):
        """
        Main testing function.
        """
        print(f"\n{'='*60}")
        print(f"▶ TEST {test_number}/{total_tests}")
        print(f"{ '='*60}")
        
        if models_to_test is None:
            models_to_test = [
                ('bayesian', {
                    'lengthscale': 500,
                    'variance': 1.0,
                    'distance_unit': "km",
                    'mcmc_samples': 5000,
                    'mcmc_burn': 3000,
                    'mcmc_scale': 0.05,
                    'mcmc_seed': 42
                }),
                ('dirichlet', {}),
                ('spatial', {'smoothing_factor': 0.1})
            ]
        
        active_models = [f"{name}_{i}" for i, (name, params) in enumerate(models_to_test)]
        print(f"🎯 TESTOWANE MODELE: {', '.join(active_models)}")
        
        start_time = datetime.datetime.now()
        
        try:
            if test_number == 1 or self.cached_data is None:
                cached = self.prepare_data_once(
                    test_params['cutoff_km'], 
                    test_params['co_ktory'], 
                    test_params['n_observations']
                )
            else:
                cached = self.cached_data
            
            gdf = cached['gdf']
            true_probs = cached['true_probs']
            points = cached['points']
            
            test_params['n_points'] = len(gdf)
            
            print(f"▶ [OBS] Losowanie {test_params['n_observations']} obserwacji...")
            obs_idx = losuj_obserwacje(gdf, test_params['n_observations'])
            
            predictions = {}
            metrics = {}
            models = {}

            for i, (model_name, model_params) in enumerate(models_to_test):
                unique_model_key = f"{model_name}_{i}"
                model_display_name = f"{model_name.capitalize()} ({i})"
                
                try:
                    if model_name == 'bayesian':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        # === KROK 1: Inicjalizacja Modelu ===
                        # Model pola Gaussowskiego z funkcją linku probit. Tworzy macierz kowariancji `Sigma`
                        # na podstawie `lengthscale` i odległości między punktami.
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': obs_idx,
                            'lengthscale': model_params.get('lengthscale', 500),
                            'variance': model_params.get('variance', 1.0),
                            'distance_unit': model_params.get('distance_unit', 'km')
                        }
                        model = BayesianFieldModel(**constructor_params)
                        model.przygotuj_apriori()
                        
                        # === KROK 2: Przygotowanie Predykcji (Sampling MCMC) ===
                        # Używa algorytmu Metropolis-Hastings do próbkowania z `posterior` dla latentnego pola `w`.
                        model.przygotuj_predykcyjny(
                            num_samples=model_params.get('mcmc_samples', 5000),
                            burn_in=model_params.get('mcmc_burn', 3000),
                            proposal_scale=model_params.get('mcmc_scale', 0.05),
                            seed=model_params.get('mcmc_seed', 42) + test_number
                        )
                        
                        # === KROK 3: Obliczenie Predykcji (Średnia Posterior) ===
                        # Transformuje próbki `w` na `p` (p-stwo) i oblicza ich średnią.
                        pred = model.posterior_mean()

                    elif model_name == 'bayesian_adaptive_search_binary':
                        print(f"\n--- MODEL: {model_display_name.upper()} (Analytic Moment Estimator) ---")
    
                        # === KROK 1: Inicjalizacja Modelu ===
                        # Zostawiamy tylko to, co faktycznie definiuje strukturę modelu.
                        # Nowy estymator sam wyliczy 'c' na podstawie danych i geometrii.
                        constructor_params = {
                            'p':model_params.get('p',0.8),
                            'space_points': points, 
                            'metric_func': haversine, 
                            'observed_indices': obs_idx,
                            'variance': model_params.get('variance', 1.0),
                            'distance_unit': model_params.get('distance_unit', 'km')
                        }
    
                        # Jeśli mimo wszystko chcesz przekazać stare parametry (np. do logów), 
                        # **kwargs w __init__ je obsłuży, ale tutaj jest czyściej bez nich.
                        model = BayesianFieldModelAdaptiveSearchBinary(**constructor_params)
                        
                        # === KROK 2: Adaptacyjne Wyszukiwanie `lengthscale` (Binarne) ===
                        # Model używa wyszukiwania binarnego do znalezienia optymalnego `lengthscale`.
                        model.przygotuj_apriori() # This will run the binary adaptive search
                        
                        # === KROK 3: Finalny Sampling MCMC ===
                        # Z użyciem znalezionego `lengthscale`, uruchamiana jest pełna symulacja MCMC.
                        model.przygotuj_predykcyjny(
                            num_samples=model_params.get('mcmc_samples', 5000),
                            burn_in=model_params.get('mcmc_burn', 3000),
                            proposal_scale=model_params.get('mcmc_scale', 0.05),
                            seed=model_params.get('mcmc_seed', 42) + test_number
                        )
                        
                        # === KROK 4: Obliczenie Predykcji ===
                        pred = model.posterior_mean()
                    elif model_name == 'lenk_adaptive':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        # === KROK 1: Inicjalizacja Modelu ===
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': obs_idx,
                            'variance': model_params.get('variance', 1.0),
                            'distance_unit': model_params.get('distance_unit', 'km'),
                            'mu_prior': model_params.get('mu_prior', 0.0),
                            'start_ls': model_params.get('start_ls', 1000),
                            'step_size': model_params.get('step_size', 2000),
                            'k_steps': model_params.get('k_steps', 3),
                            'num_samples_search': model_params.get("num_samples_search", 100),
                            'burn_in_search': model_params.get("burn_in_search", 50),
                            'proposal_scale_search': model_params.get("proposal_scale_search", 0.05)
                        }
                        model = LenkAdaptiveSearchModel(**constructor_params)
                        
                        # === KROK 2: Adaptacyjne Wyszukiwanie `lengthscale` ===
                        # Model iteracyjnie szuka optymalnego `lengthscale` poprzez estymację
                        # wiarygodności brzegowej dla różnych wartości.
                        model.przygotuj_apriori() # This will run the adaptive search
                        
                        # === KROK 3: Finalny Sampling MCMC ===
                        # Z użyciem znalezionego `lengthscale`, uruchamiana jest pełna symulacja MCMC.
                        model.przygotuj_predykcyjny(
                            num_samples=model_params.get('mcmc_samples', 5000),
                            burn_in=model_params.get('mcmc_burn', 3000),
                            proposal_scale=model_params.get('mcmc_scale', 0.05),
                            seed=model_params.get('mcmc_seed', 42) + test_number
                        )
                        
                        # === KROK 4: Obliczenie Predykcji ===
                        pred = model.posterior_mean()
                    elif model_name == 'bayesian_cv_gridsearch':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        # === KROK 1: Inicjalizacja Modelu ===
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': obs_idx,
                            'variance': model_params.get('variance', 1.0),
                            'distance_unit': model_params.get('distance_unit', 'km'),
                            'lengthscale_grid': model_params.get('lengthscale_grid'),
                            'cv_k': model_params.get('cv_k', 5),
                            'cv_mcmc_samples': model_params.get('cv_mcmc_samples', 1000),
                            'cv_mcmc_burn': model_params.get('cv_mcmc_burn', 500)
                        }
                        model = BayesianFieldModelCVGridSearch(**constructor_params)
                        
                        # === KROK 2: Grid Search z Walidacją Krzyżową ===
                        # Model testuje każdą wartość `lengthscale` z siatki, używając k-krotnej 
                        # walidacji krzyżowej do oceny błędu i wyboru najlepszej wartości.
                        model.przygotuj_apriori()
                        
                        # === KROK 3: Finalny Sampling MCMC ===
                        # Z użyciem znalezionego `lengthscale`, uruchamiana jest pełna symulacja MCMC.
                        model.przygotuj_predykcyjny(
                            num_samples=model_params.get('mcmc_samples', 5000),
                            burn_in=model_params.get('mcmc_burn', 3000),
                            proposal_scale=model_params.get('mcmc_scale', 0.05),
                            seed=model_params.get('mcmc_seed', 42) + test_number
                        )
                        
                        # === KROK 4: Obliczenie Predykcji ===
                        pred = model.posterior_mean()
                    elif model_name == 'logistic_normal_mcmc':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        # === KROK 1: Inicjalizacja Modelu ===
                        # Model pola Gaussowskiego z funkcją linku logit (odmiana modelu `bayesian`).
                        # Tworzy macierz kowariancji `Sigma` na podstawie `lengthscale`.
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': obs_idx,
                            'lengthscale': model_params.get('lengthscale', 1000.0),
                            'variance': model_params.get('variance', 1.0),
                            'distance_unit': model_params.get('distance_unit', 'km'),
                            'mu_prior': model_params.get('mu_prior', 0.0)
                        }
                        model = LogisticNormalMCMC(**constructor_params)
                        model.przygotuj_apriori()
                        
                        # === KROK 2: Przygotowanie Predykcji (Sampling MCMC) ===
                        # Używa algorytmu Metropolis-Hastings do próbkowania z `posterior` dla latentnego pola `w`.
                        model.przygotuj_predykcyjny(
                            num_samples=model_params.get('mcmc_samples', 5000),
                            burn_in=model_params.get('mcmc_burn', 3000),
                            proposal_scale=model_params.get('mcmc_scale', 0.05),
                            seed=model_params.get('mcmc_seed', 42) + test_number
                        )
                        
                        # === KROK 3: Obliczenie Predykcji (Średnia Posterior) ===
                        # Transformuje próbki `w` na `p` (p-stwo) za pomocą funkcji logistycznej i oblicza ich średnią.
                        pred = model.posterior_mean()
                    elif model_name == 'dirichlet':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        # === KROK 1 i 2: Przygotowanie Danych i Obliczenia ===
                        # Model Dirichlet zlicza obserwacje (`obs_idx`) i oblicza parametry `posterior`.
                        # Prior `alpha=1` (jednostajny), posterior `alpha_n = alpha + zliczenia`.
                        model = DirichletModel(obs_idx, len(gdf))
                        
                        # === KROK 3: Predykcja (Średnia Posterior) ===
                        # Oblicza średnią rozkładu posterior, która jest oczekiwanym prawdopodobieństwem.
                        pred = model.posterior_mean()

                    elif model_name == 'gaussian':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        # === KROK 1: Inicjalizacja Modelu ===
                        # W pełni Bayesowski Proces Gaussowski.
                        gp_constructor_params = {
                            'space_points': points, 'observed_indices': obs_idx,
                            'lengthscale_prior': model_params.get('lengthscale_prior', (1000, 500)),
                            'variance_prior': model_params.get('variance_prior', (2, 1))
                        }
                        model = BayesianGaussianProcess(**gp_constructor_params)
                        
                        # === KROK 2: Próbkowanie Posterior (MCMC) ===
                        # Używa MCMC do próbkowania z połączonego rozkładu posterior dla pola latentnego `w`
                        # ORAZ hiperparametrów (`lengthscale`, `variance`).
                        model.sample_posterior(
                            n_samples=model_params.get('n_samples', 1000),
                            burn_in=model_params.get('burn_in', 500),
                            step_size=model_params.get('step_size', 0.1)
                        )
                        
                        # === KROK 3: Predykcja Posterior ===
                        # Uśrednia predykcje po wszystkich próbkach z MCMC, uwzględniając niepewność hiperparametrów.
                        pred = model.posterior_predictive()

                    elif model_name == 'spatial':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        # === KROK 1: Inicjalizacja Modelu ===
                        # Prosty model wygładzania przestrzennego (kernel smoothing).
                        model = BayesianSpatialSmoothing(
                            points, obs_idx, smoothing_factor=model_params.get('smoothing_factor', 0.1)
                        )
                        
                        # === KROK 2: Obliczenie Predykcji ===
                        # Tworzy binarny wektor obserwacji (1 tam, gdzie była obs., 0 gdzie nie).
                        # Następnie "rozmywa" ten wektor za pomocą jądra (kernela) przestrzennego,
                        # tworząc ważoną sumę, która jest normalizowana do rozkładu p-stwa.
                        pred = model.posterior_mean()
                    elif model_name == 'spatial_binomial':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        
                        # === KROK 1: Przygotowanie Danych Obserwacyjnych ===
                        # Zlicz obserwacje (`obs_idx`) jako liczbę sukcesów (`counts`) w każdej lokalizacji.
                        counts = np.bincount(obs_idx, minlength=len(points))
                        # Ustal liczbę prób (`N`) jako stałą dla wszystkich lokalizacji.
                        N = np.full(len(points), test_params['n_observations'])
                        # Model będzie działał tylko na punktach, gdzie były obserwacje.
                        actual_obs_indices = np.where(counts > 0)[0]
                        
                        # === KROK 2: Inicjalizacja i Trening Modelu ===
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': actual_obs_indices,
                            'counts': counts, 'N': N,
                            'alpha_prior': model_params.get('alpha_prior', 0.5),
                            'beta_prior': model_params.get('beta_prior', 0.5),
                            'smoothing_strength': model_params.get('smoothing_strength', 0.1)
                        }
                        model = SpatialBinomialConjugate(**constructor_params)
                        # Metoda fit estymuje parametry `posterior` rozkładu Beta dla każdego punktu,
                        # z uwzględnieniem wygładzania przestrzennego i optymalizacją `phi`.
                        model.fit(phi=model_params.get('phi'), optimize_phi=model_params.get('optimize_phi', True))
                        
                        # === KROK 3: Predykcja na Całej Siatce ===
                        # Generuje próbki `p` z rozkładu posterior dla obserwowanych punktów,
                        # a następnie interpoluje je na całą siatkę przez ważenie przestrzenne.
                        pred, _ = model.predict(num_samples=model_params.get('num_samples', 1000))

                    elif model_name == 'spatial_gaussian':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        
                        # === KROK 1: Przygotowanie Danych Obserwacyjnych ===
                        # Zlicz wystąpienia każdego obserwowanego indeksu, aby uzyskać liczbę obserwacji w każdej lokalizacji.
                        observed_counts = np.bincount(obs_idx, minlength=len(points))
                        
                        # Oblicz empiryczne prawdopodobieństwo dla każdej lokalizacji. To jest wektor 'y' dla modelu.
                        total_observations = len(obs_idx)
                        if total_observations > 0:
                            empirical_probs = observed_counts / total_observations
                        else:
                            empirical_probs = np.zeros(len(points))
                        
                        # Wyodrębnij unikalne indeksy, w których dokonano obserwacji, oraz odpowiadające im wartości.
                        unique_indices = np.where(observed_counts > 0)[0]
                        observed_values = empirical_probs[unique_indices]

                        # === KROK 2: Inicjalizacja i Trening Modelu ===
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': unique_indices,
                            'counts': observed_values, # dla modelu Gaussa, 'counts' to wartości y
                            'mu_prior': model_params.get('mu_prior', np.mean(observed_values) if len(observed_values) > 0 else 0),
                            'sigma_prior': model_params.get('sigma_prior', np.std(observed_values) if len(observed_values) > 0 else 1),
                        }
                        model = GaussianSpatialModelConjugate(**constructor_params)
                        # Metoda fit optymalizuje phi, oblicza wagi i parametry posterior
                        model.fit(phi=model_params.get('phi'), optimize_phi=model_params.get('optimize_phi', True))
                        
                        # === KROK 3: Predykcja na Całej Siatce ===
                        # Metoda predict generuje predykcje dla wszystkich punktów siatki
                        pred, _ = model.predict(num_samples=model_params.get('num_samples', 1000))
                    
                    elif model_name == 'spatial_poisson':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        
                        # === KROK 1: Przygotowanie Danych Obserwacyjnych ===
                        # Zlicz obserwacje (`obs_idx`) jako liczbę zdarzeń (`counts`).
                        counts = np.bincount(obs_idx, minlength=len(points))
                        # Ustal ekspozycję jako 1 dla wszystkich lokalizacji.
                        exposure = np.ones(len(points)) 
                        # Model będzie działał na punktach, gdzie były obserwacje.
                        actual_obs_indices = np.where(counts > 0)[0]
                        
                        # === KROK 2: Inicjalizacja i Trening Modelu ===
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': actual_obs_indices,
                            'counts': counts, 'N': exposure, # Dla Poissona, N to ekspozycja
                            'alpha_prior': model_params.get('alpha_prior', 0.5),
                            'beta_prior': model_params.get('beta_prior', 0.5),
                            'smoothing_strength': model_params.get('smoothing_strength', 1.0),
                        }
                        model = SpatialPoissonConjugate(**constructor_params)
                        # Metoda fit estymuje parametry `posterior` rozkładu Gamma dla `lambda`,
                        # z uwzględnieniem wygładzania przestrzennego.
                        model.fit(phi=model_params.get('phi'), optimize_phi=model_params.get('optimize_phi', True))
                        
                        # === KROK 3: Predykcja na Całej Siatce ===
                        # Model przewiduje intensywność `lambda` przez interpolację przestrzenną.
                        pred, _ = model.predict(num_samples=model_params.get('num_samples', 1000))
                        
                        # === KROK 4: Normalizacja Predykcji ===
                        # Przekształć przewidziane intensywności na rozkład prawdopodobieństwa.
                        if pred is not None and np.sum(pred) > 0:
                            pred = pred / np.sum(pred)
                        else:
                            pred = np.ones(len(points)) / len(points)
                        
                    else:
                        print(f"⚠️ Nieznany model: {model_name}")
                        continue
                        
                    predictions[unique_model_key] = pred
                    models[unique_model_key] = model
                    metrics[unique_model_key] = oblicz_metryki(true_probs, pred, model_display_name, verbose=True)

                except Exception as e:
                    print(f"❌ Błąd podczas uruchamiania modelu {model_display_name}: {e}")
                    import traceback
                    traceback.print_exc()

            if len(predictions) >= 2:
                comparison_stats = self._calculate_comparison_stats(
                    true_probs, predictions
                )
            else:
                comparison_stats = {
                    'best_model': list(predictions.keys())[0] if predictions else None,
                    'wilcoxon_pvalue': None,
                    'better_counts': {},
                    'equal_count': 0
                }
            
            duration = (datetime.datetime.now() - start_time).total_seconds()
            
            test_id = None
            if save and len(metrics) >= 2:
                save_test_params = test_params.copy() 
                
                bayesian_model_params_for_save = {}
                for mn, mp in models_to_test:
                    if mn == 'bayesian':
                        bayesian_model_params_for_save = mp
                        break
                
                if not bayesian_model_params_for_save:
                    bayesian_model_params_for_save = {
                        'lengthscale': 500,
                        'variance': 1.0,
                        'distance_unit': "km",
                        'mcmc_samples': 5000,
                        'mcmc_burn': 3000,
                        'mcmc_scale': 0.05,
                        'mcmc_seed': 42
                    }
                save_test_params['lengthscale'] = bayesian_model_params_for_save.get('lengthscale', 0.0)
                save_test_params['variance'] = bayesian_model_params_for_save.get('variance', 0.0)
                save_test_params['mcmc_samples'] = bayesian_model_params_for_save.get('mcmc_samples', 0)
                save_test_params['mcmc_burn'] = bayesian_model_params_for_save.get('mcmc_burn', 0)
                save_test_params['mcmc_scale'] = bayesian_model_params_for_save.get('mcmc_scale', 0.0)
                save_test_params['mcmc_seed'] = bayesian_model_params_for_save.get('mcmc_seed', 0)
                
                test_id = self._save_to_database(save_test_params, metrics, 
                                               comparison_stats, models_to_test, duration)
            
            self._print_test_summary(metrics, comparison_stats, duration)
            
            if visualise:
                for model_name, pred in predictions.items():
                    stworz_mape_porownawcza(gdf, true_probs, pred, 
                                          f"{model_name.capitalize()} Model")
                pokaz_punkt_referencyjny(gdf, title="Punkt referencyjny (indeks 1)")
            
            return {
                'test_id': test_id,
                'predictions': predictions,
                'metrics': metrics,
                'models': models,
                'comparison_stats': comparison_stats,
                'duration': duration,
                'success': True,
                'true_probs': true_probs,
                'gdf': gdf
            }
            
        except Exception as e:
            print(f"❌ Błąd podczas testu {test_number}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'test_id': f"failed_test_{test_number}",
                'error': str(e),
                'success': False
            }
    
    def _calculate_comparison_stats(self, true_probs, predictions):
        """Calculates comparison statistics between models."""
        errors = {}
        for model_name, pred in predictions.items():
            errors[model_name] = np.abs(pred - true_probs)
        
        better_counts = {model_name: 0 for model_name in predictions.keys()}
        equal_count = 0
        
        n_points = len(true_probs)
        for i in range(n_points):
            point_errors = {model: errors[model][i] for model in predictions.keys()}
            min_error = min(point_errors.values())
            
            best_models = [model for model, error in point_errors.items() 
                          if error == min_error]
            
            if len(best_models) == 1:
                better_counts[best_models[0]] += 1
            else:
                equal_count += 1
        
        wilcoxon_pvalue = None
        model_names = list(predictions.keys())
        if len(model_names) >= 2:
            from scipy.stats import wilcoxon
            try:
                stat, pvalue = wilcoxon(errors[model_names[0]], errors[model_names[1]])
                wilcoxon_pvalue = float(pvalue)
            except:
                wilcoxon_pvalue = 1.0
        
        best_model = None
        best_mse = float('inf')
        for model_name, pred in predictions.items():
            mse = np.mean((pred - true_probs) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_model = model_name
        
        return {
            'better_counts': better_counts,
            'equal_count': equal_count,
            'wilcoxon_pvalue': wilcoxon_pvalue,
            'best_model': best_model,
            'best_mse': best_mse
        }
    
    def _save_to_database(self, test_params, metrics, comparison_stats, models_to_test, duration):
        """Saves the results to the database."""
        test_ids = []
        
        remaining_models = list(models_to_test)

        for model_key, model_metrics in metrics.items():
            model_type = model_key.rsplit('_', 1)[0]

            current_model_params = {}
            model_found_index = -1
            for i, (name, params) in enumerate(remaining_models):
                if name == model_type:
                    current_model_params = params
                    model_found_index = i
                    break
            
            if model_found_index != -1:
                remaining_models.pop(model_found_index)

            save_params = test_params.copy()
            if model_type == 'bayesian':
                save_params['lengthscale'] = current_model_params.get('lengthscale', 0)
                save_params['variance'] = current_model_params.get('variance', 0)
                save_params['mcmc_samples'] = current_model_params.get('mcmc_samples', 0)
                save_params['mcmc_burn'] = current_model_params.get('mcmc_burn', 0)
                save_params['mcmc_scale'] = current_model_params.get('mcmc_scale', 0)
                save_params['mcmc_seed'] = current_model_params.get('mcmc_seed', 0)
            
            metrics_bayesian = {}
            metrics_dirichlet = {}
            metrics_gp = {}
            metrics_spatial = {}

            if model_type == 'bayesian':
                metrics_bayesian = model_metrics
            elif model_type == 'dirichlet':
                metrics_dirichlet = model_metrics
            elif model_type == 'gaussian':
                metrics_gp = model_metrics
            elif model_type == 'spatial':
                metrics_spatial = model_metrics

            test_id = self.db.save_test_results(
                save_params,
                metrics_bayesian,
                metrics_dirichlet,
                metrics_gp,
                metrics_spatial,
                comparison_stats,
                duration
            )
            test_ids.append(test_id)
            
        return test_ids
    
    def _print_test_summary(self, metrics, comparison_stats, duration):
        """Displays the test summary."""
        print(f"\n📈 PODSUMOWANIE TESTU:")
        print(f"   - Czas trwania: {duration:.1f}s")
        
        if metrics:
            print(f"\n   MSE:")
            for model_key, model_metrics in metrics.items():
                print(f"     - {model_key.capitalize():15s}: {model_metrics.get('MSE', 'N/A'):.6f}")
            
            print(f"\n   MAE:")
            for model_key, model_metrics in metrics.items():
                print(f"     - {model_key.capitalize():15s}: {model_metrics.get('MAE', 'N/A'):.6f}")
        
        if comparison_stats.get('best_model'):
            best_mse_val = comparison_stats.get('best_mse', 'N/A')
            if isinstance(best_mse_val, float):
                print(f"\n   NAJLEPSZY MODEL: {comparison_stats['best_model'].capitalize()} "
                      f"(MSE={best_mse_val:.6f})")
            else:
                print(f"\n   NAJLEPSZY MODEL: {comparison_stats['best_model'].capitalize()} "
                      f"(MSE={best_mse_val})")
        
        if 'better_counts' in comparison_stats:
            print(f"\n   LICZBA PUNKTÓW Z NAJLEPSZYM WYNIKIEM:")
            for model_key, count in comparison_stats['better_counts'].items():
                print(f"     - {model_key.capitalize():15s}: {count}")
            if comparison_stats.get('equal_count', 0) > 0:
                print(f"     - Równe wyniki: {comparison_stats['equal_count']}")
        
        if comparison_stats.get('wilcoxon_pvalue') is not None:
            print(f"   - Wilcoxon p-value: {comparison_stats.get('wilcoxon_pvalue'):.4f}")
    
    def clear_cache(self):
        """Clears the data cache."""
        self.cached_data = None
        print("✅ Cache danych wyczyszczony")
    
    def run_tests(self, base_params, n_tests=1, models_to_test=None, 
                  varying_params=None, save=False):
        """Runs a series of tests."""
        print(f"\n🎯 ROZPOCZĘCIE SERII {n_tests} TESTÓW")
        print(f"Parametry bazowe: {json.dumps({k: v for k, v in base_params.items() if k != 'csv_path'}, indent=2, default=str)}")
        
        all_results = []
        
        for i in range(n_tests):
            test_params = base_params.copy()
            
            if varying_params:
                for param_name, values in varying_params.items():
                    if i < len(values):
                        test_params[param_name] = values[i]
                    else:
                        test_params[param_name] = values[-1]
            
            result = self.test(test_params, i+1, n_tests, models_to_test, save,i==0)
            all_results.append(result)
            
            if i < n_tests - 1:
                print("\n⏳ Przygotowanie do następnego testu...")
                import time
                time.sleep(0.1)
        
        successful_tests = [r for r in all_results if r['success']]
        print(f"\n✅ Zakończono serię testów: {len(successful_tests)}/{n_tests} udanych")
        
        if save:
            self.db.print_summary()
        
        return all_results
    
    def test_observation_length_impact(self, base_params, n_observations_list=None,
                                     models_to_test=None, k=5, save=False):
        
        if n_observations_list is None:
            n_observations_list = [#250, 750, 1000, 2000, 
                                  #2500, 5000, 7500, 10000,15000,20000,
                                  35000,50000,100000]
        
        print(f"\n🎯 BADANIE WPŁYWU LICZBY OBSERWACJI (k={k})")
        print(f"{ '='*60}")
        
        all_results = []
        
        max_obs = max(n_observations_list)
        cached = self.prepare_data_once(
            base_params['cutoff_km'], 
            base_params['co_ktory'], 
            max_obs
        )
        self.cached_data = cached
        
        for i, n_obs in enumerate(n_observations_list):
            obs_results = []
            for j in range(k):
                print(f"\n{'='*50}")
                print(f"▶ TEST {i*k+j+1}/{len(n_observations_list)*k} - {n_obs} obserwacji (próba {j+1}/{k})")
                print(f"{ '='*50}")
                
                test_params = base_params.copy()
                test_params['n_observations'] = n_obs
                
                result = self.test(test_params, i*k+j+1, len(n_observations_list)*k, 
                                 models_to_test, save=False)
                
                if result['success']:
                    result['n_observations'] = n_obs
                    result['k_run'] = j
                    obs_results.append(result)
            
            if obs_results:
                all_results.extend(obs_results)
        
        if save:
            self._save_observation_length_results(all_results)
        
        self._plot_observation_length_results(all_results)
        
        return all_results

    def _save_observation_length_results(self, results):
        """Saves the results of the observation length impact study."""
        if not results:
            return
        
        data = []
        for result in results:
            if not result['success']:
                continue
            
            row = {
                'n_observations': result.get('n_observations', 0),
                'k_run': result.get('k_run', 0)
            }
            
            for model_name, metrics in result.get('metrics', {}).items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        row[f"{model_name}_{metric_name.lower()}"] = float(value)
            
            data.append(row)
        
        if data:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"observation_length_impact_{timestamp}.csv"
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            print(f"\n💾 Wyniki badania wpływu obserwacji zapisane do: {output_file}")
    
    def _plot_observation_length_results(self, results):
        """Plots the results of the observation length impact study."""
        if not results:
            return
            
        data = []
        model_keys = [] # Collect all unique model keys
        for result in results:
            if not result.get('success'):
                continue
            
            n_obs = result.get('n_observations', 0)
            k_run = result.get('k_run', 0)
            row = {'n_observations': n_obs, 'k_run': k_run}
            
            if 'metrics' in result:
                for model_name, metrics_dict in result['metrics'].items():
                    # Ensure model_name (which is actually unique_model_key like 'bayesian_0') is in model_keys
                    if model_name not in model_keys:
                        model_keys.append(model_name)
                    for metric_name, value in metrics_dict.items():
                        if isinstance(value, (int, float, np.number)):
                            row[f"{model_name}_{metric_name.lower()}"] = float(value)
            data.append(row)

        if not data:
            print("⚠️ Brak danych do wykreślenia")
            return

        df = pd.DataFrame(data)
        df = df.sort_values(['n_observations', 'k_run'])

        # --- Plot 1: Mean values ---
        mean_df = df.groupby('n_observations').mean().reset_index()
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Wpływ liczby obserwacji na jakość predykcji', 
                     fontsize=16, fontweight='bold')
        
        # Generate colors dynamically for each unique model_key
        colors_cmap = plt.cm.get_cmap('tab10', len(model_keys))
        model_colors_map = {model_key: plt.cm.get_cmap('tab10')(i) for i, model_key in enumerate(model_keys)}
        
        # Add a specific color for spatial_binomial if it's not already there
        if 'spatial_binomial_0' not in model_colors_map:
            model_colors_map['spatial_binomial_0'] = 'cyan'
        
        ax = axes[0, 0]
        mse_plotted = False
        for col_name in mean_df.columns:
            if col_name.endswith('_mse'):
                model_key = col_name.replace('_mse', '') # This is the unique_model_key like 'bayesian_0'
                if model_key in model_colors_map and not mean_df[col_name].isna().all():
                    ax.plot(mean_df['n_observations'], mean_df[col_name], 
                            color=model_colors_map[model_key], marker='o', 
                            label=model_key.capitalize(), linewidth=2)
                    mse_plotted = True
        
        if mse_plotted:
            ax.set_xlabel('Liczba obserwacji')
            ax.set_ylabel('MSE')
            ax.set_title('MSE vs liczba obserwacji')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_xscale('log')
        else:
            ax.text(0.5, 0.5, 'Brak danych MSE', ha='center', va='center', transform=ax.transAxes)

        ax = axes[0, 1]
        # This plot compares bayesian vs dirichlet, assuming only one of each.
        # This will still use the first bayesian and first dirichlet found.
        bayesian_col = next((c for c in mean_df.columns if c.startswith('bayesian_') and c.endswith('_mse')), None)
        dirichlet_col = next((c for c in mean_df.columns if c.startswith('dirichlet_') and c.endswith('_mse')), None)

        if bayesian_col and dirichlet_col and not mean_df[bayesian_col].isna().all() and not mean_df[dirichlet_col].isna().all():
            mse_diff = mean_df[bayesian_col] - mean_df[dirichlet_col]
            ax.plot(mean_df['n_observations'], mse_diff, 'g-', marker='^', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Liczba obserwacji')
            ax.set_ylabel('Różnica MSE')
            ax.set_title(f'Różnica MSE: {bayesian_col.split("_")[0].capitalize()} vs {dirichlet_col.split("_")[0].capitalize()}')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
        else:
            ax.text(0.5, 0.5, 'Brak danych do porównania MSE', ha='center', va='center', transform=ax.transAxes)

        ax = axes[1, 0]
        mae_plotted = False
        for col_name in mean_df.columns:
            if col_name.endswith('_mae'):
                model_key = col_name.replace('_mae', '')
                if model_key in model_colors_map and not mean_df[col_name].isna().all():
                    ax.plot(mean_df['n_observations'], mean_df[col_name], 
                            color=model_colors_map[model_key], marker='s', 
                            label=model_key.capitalize(), linewidth=2)
                    mae_plotted = True
        
        if mae_plotted:
            ax.set_xlabel('Liczba obserwacji')
            ax.set_ylabel('MAE')
            ax.set_title('MAE vs liczba obserwacji')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_xscale('log')
        else:
            ax.text(0.5, 0.5, 'Brak danych MAE', ha='center', va='center', transform=ax.transAxes)

        ax = axes[1, 1]
        corr_plotted = False
        for col_name in mean_df.columns:
            if col_name.endswith('_correlation'):
                model_key = col_name.replace('_correlation', '')
                if model_key in model_colors_map and not mean_df[col_name].isna().all():
                    ax.plot(mean_df['n_observations'], mean_df[col_name], 
                            color=model_colors_map[model_key], marker='x', 
                            label=model_key.capitalize(), linewidth=2)
                    corr_plotted = True

        if corr_plotted:
            ax.set_xlabel('Liczba obserwacji')
            ax.set_ylabel('Korelacja')
            ax.set_title('Korelacja vs liczba obserwacji')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
        else:
            ax.text(0.5, 0.5, 'Brak danych korelacji', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"observation_length_impact_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n  ✔ Wykresy zapisane do: {plot_file}")
        
        # --- Plot 2: Boxplots ---
        long_df_data = []
        for result in results:
            if not result.get('success'):
                continue
            
            n_obs = result.get('n_observations', 0)
            k_run = result.get('k_run', 0)
            
            if 'metrics' in result:
                for model_name, metrics_dict in result['metrics'].items():
                    for metric_name, value in metrics_dict.items():
                        if isinstance(value, (int, float, np.number)):
                            long_df_data.append({
                                'n_observations': n_obs,
                                'k_run': k_run,
                                'model': model_name, # Use unique_model_key directly
                                'metric': metric_name,
                                'value': float(value)
                            })
        
        if not long_df_data:
            return
            
        long_df = pd.DataFrame(long_df_data)
        
        import seaborn as sns
        for metric in long_df['metric'].unique():
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='n_observations', y='value', hue='model', data=long_df[long_df['metric'] == metric], palette=model_colors_map)
            plt.xlabel('Liczba obserwacji')
            plt.ylabel(metric.upper())
            plt.title(f'Rozkład {metric.upper()} vs liczba obserwacji')
            plt.legend(title='Model')
            plt.grid(True, alpha=0.3)
            # plt.xscale('log')
            plt.yscale('log')
            plt.tight_layout()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file_box = f"observation_length_impact_boxplot_{metric}_{timestamp}.png"
            plt.savefig(plot_file_box, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"\n  ✔ Wykresy boxplot dla {metric} zapisane do: {plot_file_box}")

    def test_lengthscale_impact(self, base_params, lengthscale_list=None,
                                 models_to_test=None, k=5, save=False):
        
        if lengthscale_list is None:
            lengthscale_list = [100, 500, 1000, 1500, 2000, 3000, 5000]
        
        print(f"\n🎯 BADANIE WPŁYWU WARTOŚCI LENGTHSCALE (k={k})")
        print(f"{ '='*60}")
        
        all_results = []
        
        for i, ls in enumerate(lengthscale_list):
            ls_results = []
            for j in range(k):
                print(f"\n{'='*50}")
                print(f"▶ TEST {i*k+j+1}/{len(lengthscale_list)*k} - lengthscale={ls} (próba {j+1}/{k})")
                print(f"{ '='*50}")
                
                test_params = base_params.copy()
                
                # Update lengthscale for the relevant models
                current_models_to_test = []
                for model_name, model_params in models_to_test:
                    if model_name in ['bayesian', 'logistic_normal_mcmc']:
                        new_params = model_params.copy()
                        new_params['lengthscale'] = ls
                        current_models_to_test.append((model_name, new_params))
                    else:
                        current_models_to_test.append((model_name, model_params))

                result = self.test(test_params, i*k+j+1, len(lengthscale_list)*k, 
                                 current_models_to_test, save=False)
                
                if result['success']:
                    result['lengthscale'] = ls
                    result['k_run'] = j
                    ls_results.append(result)
            
            if ls_results:
                all_results.extend(ls_results)
        
        if save:
            self._save_lengthscale_results(all_results)
        
        self._plot_lengthscale_results(all_results)
        
        return all_results

    def _save_lengthscale_results(self, results):
        """Saves the results of the lengthscale impact study."""
        if not results:
            return
        
        data = []
        for result in results:
            if not result['success']:
                continue
            
            row = {
                'lengthscale': result.get('lengthscale', 0),
                'k_run': result.get('k_run', 0)
            }
            
            for model_name, metrics in result.get('metrics', {}).items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        row[f"{model_name}_{metric_name.lower()}"] = float(value)
            
            data.append(row)
        
        if data:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"lengthscale_impact_{timestamp}.csv"
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            print(f"\n💾 Wyniki badania wpływu lengthscale zapisane do: {output_file}")
    
    def _plot_lengthscale_results(self, results):
        """Plots the results of the lengthscale impact study."""
        if not results:
            return
            
        data = []
        model_keys = []
        for result in results:
            if not result.get('success'):
                continue
            
            ls = result.get('lengthscale', 0)
            k_run = result.get('k_run', 0)
            row = {'lengthscale': ls, 'k_run': k_run}
            
            if 'metrics' in result:
                for model_name, metrics_dict in result['metrics'].items():
                    if model_name not in model_keys:
                        model_keys.append(model_name)
                    for metric_name, value in metrics_dict.items():
                        if isinstance(value, (int, float, np.number)):
                            row[f"{model_name}_{metric_name.lower()}"] = float(value)
            data.append(row)

        if not data:
            print("⚠️ Brak danych do wykreślenia")
            return

        df = pd.DataFrame(data)
        df = df.sort_values(['lengthscale', 'k_run'])

        mean_df = df.groupby('lengthscale').mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Wpływ Lengthscale na jakość predykcji', 
                     fontsize=16, fontweight='bold')
        
        # Generate a color for each unique model key
        colors = plt.cm.get_cmap('tab10', len(model_keys))
        model_colors = {model_key: colors(i) for i, model_key in enumerate(model_keys)}
        
        ax = axes[0]
        for col_name in mean_df.columns:
            if col_name.endswith('_mse'):
                model_key = col_name.replace('_mse', '')
                if model_key in model_colors and not mean_df[col_name].isna().all():
                    ax.plot(mean_df['lengthscale'], mean_df[col_name], 
                            color=model_colors[model_key], marker='o', 
                            label=model_key.capitalize(), linewidth=2)
        
        ax.set_xlabel('Lengthscale')
        ax.set_ylabel('MSE')
        ax.set_title('MSE vs Lengthscale')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xscale('log')

        ax = axes[1]
        for col_name in mean_df.columns:
            if col_name.endswith('_mae'):
                model_key = col_name.replace('_mae', '')
                if model_key in model_colors and not mean_df[col_name].isna().all():
                    ax.plot(mean_df['lengthscale'], mean_df[col_name], 
                            color=model_colors[model_key], marker='s', 
                            label=model_key.capitalize(), linewidth=2)

        ax.set_xlabel('Lengthscale')
        ax.set_ylabel('MAE')
        ax.set_title('MAE vs Lengthscale')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xscale('log')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"lengthscale_impact_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n  ✔ Wykresy zapisane do: {plot_file}")
