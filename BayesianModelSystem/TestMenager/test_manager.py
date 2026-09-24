"This module contains the TestManager class for running tests."
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

from ..Wczytywanie_danych.preprocessing import przygotuj_dane
from ..Wczytywanie_danych.loaders import losuj_obserwacje
from ..Wyniki.results_db import ResultsDatabase
from ..Modele import (
    ModelAksjomatycznyPreparamed, 
    ModelAksjomatyczny,
    ModelLenkaPreparamed,
    ModelLenka,
    ModelDirichleta, 
    BayesianGaussianProcess, 
    ModelWygladzaniaPrzestrzennego,
    ModelDwumianowySprzezony,
    ModelGausowskiSprzezony,
    ModelPoissonaSprzezony
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
             models_to_test=None, save=False, visualise=False):
        """Main testing function."""
        print(f"\n{'='*60}")
        print(f"▶ TEST {test_number}/{total_tests}")
        print(f"{ '='*60}")
        
        if models_to_test is None:
            models_to_test = [
                ('Model_Aksjomatyczny-preparamed', {
                    'lengthscale': 500,
                    'variance': 1.0,
                    'distance_unit': "km",
                    'mcmc_samples': 5000,
                    'mcmc_burn': 3000,
                    'mcmc_scale': 0.05,
                    'mcmc_seed': 42
                }),
                ('Model_Dirichleta', {}),
                ('Model_wygładzania_przestrzennego', {'smoothing_factor': 0.1})
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
            
            predictions, metrics, models = {}, {}, {}

            for i, (model_name, model_params) in enumerate(models_to_test):
                unique_model_key = f"{model_name}_{i}"
                model_display_name = f"{model_name} ({i})"
                
                try:
                    if model_name == 'Model_Aksjomatyczny-preparamed':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': obs_idx,
                            'lengthscale': model_params.get('lengthscale', 500),
                            'variance': model_params.get('variance', 1.0),
                            'distance_unit': model_params.get('distance_unit', 'km')
                        }
                        model = ModelAksjomatycznyPreparamed(**constructor_params)
                        model.przygotuj_apriori()
                        model.przygotuj_predykcyjny(
                            num_samples=model_params.get('mcmc_samples', 5000),
                            burn_in=model_params.get('mcmc_burn', 3000),
                            proposal_scale=model_params.get('mcmc_scale', 0.05),
                            seed=model_params.get('mcmc_seed', 42) + test_number
                        )
                        pred = model.posterior_mean()

                    elif model_name == 'Model_Aksjomatyczny':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        constructor_params = {
                            'p': model_params.get('p', 1.0),
                            'space_points': points, 'metric_func': haversine, 'observed_indices': obs_idx,
                            'variance': model_params.get('variance', 1.0),
                            'distance_unit': model_params.get('distance_unit', 'km')
                        }
                        model = ModelAksjomatyczny(**constructor_params)
                        model.przygotuj_apriori()
                        model.przygotuj_predykcyjny(
                            num_samples=model_params.get('mcmc_samples', 5000),
                            burn_in=model_params.get('mcmc_burn', 3000),
                            proposal_scale=model_params.get('mcmc_scale', 0.05),
                            seed=model_params.get('mcmc_seed', 42) + test_number
                        )
                        pred = model.posterior_mean()

                    elif model_name == 'Model_Lenka':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
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
                        model = ModelLenka(**constructor_params)
                        model.przygotuj_apriori()
                        model.przygotuj_predykcyjny(
                            num_samples=model_params.get('mcmc_samples', 5000),
                            burn_in=model_params.get('mcmc_burn', 3000),
                            proposal_scale=model_params.get('mcmc_scale', 0.05),
                            seed=model_params.get('mcmc_seed', 42) + test_number
                        )
                        pred = model.posterior_mean()

                    elif model_name == 'Model_Lenka-preparamed':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': obs_idx,
                            'lengthscale': model_params.get('lengthscale', 1000.0),
                            'variance': model_params.get('variance', 1.0),
                            'distance_unit': model_params.get('distance_unit', 'km'),
                            'mu_prior': model_params.get('mu_prior', 0.0)
                        }
                        model = ModelLenkaPreparamed(**constructor_params)
                        model.przygotuj_apriori()
                        model.przygotuj_predykcyjny(
                            num_samples=model_params.get('mcmc_samples', 5000),
                            burn_in=model_params.get('mcmc_burn', 3000),
                            proposal_scale=model_params.get('mcmc_scale', 0.05),
                            seed=model_params.get('mcmc_seed', 42) + test_number
                        )
                        pred = model.posterior_mean()

                    elif model_name == 'Model_Dirichleta':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        model = ModelDirichleta(obs_idx, len(gdf))
                        pred = model.posterior_mean()

                    elif model_name == 'Model_wygładzania_przestrzennego':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        model = ModelWygladzaniaPrzestrzennego(
                            points, obs_idx, smoothing_factor=model_params.get('smoothing_factor', 0.1)
                        )
                        pred = model.posterior_mean()

                    elif model_name == 'Model_Dwumianowy_sprzężony':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        counts = np.bincount(obs_idx, minlength=len(points))
                        N = np.full(len(points), test_params['n_observations'])
                        actual_obs_indices = np.where(counts > 0)[0]
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': actual_obs_indices,
                            'counts': counts, 'N': N,
                            'alpha_prior': model_params.get('alpha_prior', 0.5),
                            'beta_prior': model_params.get('beta_prior', 0.5),
                            'smoothing_strength': model_params.get('smoothing_strength', 0.1)
                        }
                        model = ModelDwumianowySprzezony(**constructor_params)
                        model.fit(phi=model_params.get('phi'), optimize_phi=model_params.get('optimize_phi', True))
                        pred, _ = model.predict(num_samples=model_params.get('num_samples', 1000))

                    elif model_name == 'Model_Gausowski_sprzężony':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        observed_counts = np.bincount(obs_idx, minlength=len(points))
                        total_observations = len(obs_idx)
                        empirical_probs = observed_counts / total_observations if total_observations > 0 else np.zeros(len(points))
                        unique_indices = np.where(observed_counts > 0)[0]
                        observed_values = empirical_probs[unique_indices]
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': unique_indices,
                            'counts': observed_values,
                            'mu_prior': model_params.get('mu_prior', np.mean(observed_values) if len(observed_values) > 0 else 0),
                            'sigma_prior': model_params.get('sigma_prior', np.std(observed_values) if len(observed_values) > 0 else 1),
                        }
                        model = ModelGausowskiSprzezony(**constructor_params)
                        model.fit(phi=model_params.get('phi'), optimize_phi=model_params.get('optimize_phi', True))
                        pred, _ = model.predict(num_samples=model_params.get('num_samples', 1000))
                    
                    elif model_name == 'Model_Poissona_sprzężony':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        counts = np.bincount(obs_idx, minlength=len(points))
                        exposure = np.ones(len(points)) 
                        actual_obs_indices = np.where(counts > 0)[0]
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': actual_obs_indices,
                            'counts': counts, 'N': exposure,
                            'alpha_prior': model_params.get('alpha_prior', 0.5),
                            'beta_prior': model_params.get('beta_prior', 0.5),
                            'smoothing_strength': model_params.get('smoothing_strength', 1.0),
                        }
                        model = ModelPoissonaSprzezony(**constructor_params)
                        model.fit(phi=model_params.get('phi'), optimize_phi=model_params.get('optimize_phi', True))
                        pred, _ = model.predict(num_samples=model_params.get('num_samples', 1000))
                        pred = pred / np.sum(pred) if pred is not None and np.sum(pred) > 0 else np.ones(len(points)) / len(points)
                        
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

            comparison_stats = self._calculate_comparison_stats(true_probs, predictions) if len(predictions) >= 2 else {'best_model': list(predictions.keys())[0] if predictions else None, 'best_mse': 0, 'better_counts': {}, 'equal_count': 0}
            duration = (datetime.datetime.now() - start_time).total_seconds()
            
            test_id = None
            if save and len(metrics) >= 2:
                test_id = self._save_to_database(test_params, metrics, comparison_stats, models_to_test, duration)
            
            self._print_test_summary(metrics, comparison_stats, duration)
            if visualise:
                for model_key, pred in predictions.items():
                    stworz_mape_porownawcza(gdf, true_probs, pred, f"{model_key.capitalize()} Model")
                pokaz_punkt_referencyjny(gdf, title="Punkt referencyjny")
            
            return {'test_id': test_id, 'predictions': predictions, 'metrics': metrics, 'models': models, 'comparison_stats': comparison_stats, 'duration': duration, 'success': True, 'true_probs': true_probs, 'gdf': gdf}
        except Exception as e:
            print(f"❌ Błąd podczas testu: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_comparison_stats(self, true_probs, predictions):
        errors = {m: np.abs(p - true_probs) for m, p in predictions.items()}
        better_counts = {m: 0 for m in predictions.keys()}
        equal_count = 0
        for i in range(len(true_probs)):
            pe = {m: errors[m][i] for m in predictions.keys()}
            me = min(pe.values())
            bm = [m for m, e in pe.items() if e == me]
            if len(bm) == 1: better_counts[bm[0]] += 1
            else: equal_count += 1
        
        best_model, best_mse = None, float('inf')
        for m, p in predictions.items():
            mse = np.mean((p - true_probs)**2)
            if mse < best_mse: best_mse, best_model = mse, m
            
        return {'better_counts': better_counts, 'equal_count': equal_count, 'best_model': best_model, 'best_mse': best_mse}

    def _save_to_database(self, test_params, metrics, comparison_stats, models_to_test, duration):
        """Aggregates and saves ALL model metrics, creating multiple rows if necessary."""
        
        # 1. Grupuj modele wg typów
        groups = {'bayesian': [], 'dirichlet': [], 'gp': [], 'spatial': []}
        better_counts = comparison_stats.get('better_counts', {})
        
        for model_key, model_metrics in metrics.items():
            key_lower = model_key.lower()
            cat = None
            if 'aksjomatyczny' in key_lower or 'lenka' in key_lower: cat = 'bayesian'
            elif 'dirichlet' in key_lower: cat = 'dirichlet'
            elif any(x in key_lower for x in ['gausowski', 'poissona', 'dwumianowy', 'gp']): cat = 'gp'
            elif any(x in key_lower for x in ['wygladzania', 'wygładzania', 'spatial']): cat = 'spatial'
            
            if cat:
                groups[cat].append({
                    'name': model_key,
                    'metrics': model_metrics,
                    'better_count': better_counts.get(model_key, 0)
                })

        # 2. Wyznacz liczbę potrzebnych wierszy (tyle, by każdy model wystąpił co najmniej raz)
        num_rows = max([len(v) for v in groups.values()] + [1])
        test_ids = []

        # 3. Wyciągnij parametry wspólne dla całego testu
        if models_to_test:
            p = models_to_test[0][1]
            for k in ['lengthscale', 'variance', 'mcmc_samples', 'mcmc_burn', 'mcmc_scale', 'mcmc_seed']:
                if k in p: test_params[k] = p[k]

        # 4. Generuj wiersze
        for i in range(num_rows):
            # Wybieramy dane dla każdego slotu (jeśli modele się skończyły, bierzemy ostatni dostępny)
            row_data = {}
            row_names = {}
            row_comp_stats = {'equal_count': comparison_stats.get('equal_count', 0)}
            
            for cat in ['bayesian', 'dirichlet', 'gp', 'spatial']:
                models_in_cat = groups[cat]
                if models_in_cat:
                    # Bierzemy i-ty model lub ostatni, jeśli i wykracza poza zakres
                    idx = min(i, len(models_in_cat) - 1)
                    m = models_in_cat[idx]
                    row_data[cat] = m['metrics']
                    row_names[cat] = m['name']
                    row_comp_stats[f'{cat}_better_count'] = m['better_count']
                else:
                    row_data[cat] = {}
                    row_names[cat] = ''
                    row_comp_stats[f'{cat}_better_count'] = 0

            # Zapisz wiersz
            tid = self.db.save_test_results(
                test_params, 
                row_data['bayesian'], row_data['dirichlet'], row_data['gp'], row_data['spatial'],
                row_comp_stats, duration, names=row_names
            )
            test_ids.append(tid)
            print(f"  ✔ Zapisano wyniki modelu: {row_names['bayesian'] or row_names['gp'] or 'Model'}")

        return test_ids
    
    def _print_test_summary(self, metrics, comparison_stats, duration):
        print(f"\n📈 PODSUMOWANIE TESTU ({duration:.1f}s):")
        for m, met in metrics.items():
            print(f"   - {m.capitalize():20s}: MSE={met.get('MSE',0):.6f}, MAE={met.get('MAE',0):.6f}")
        if comparison_stats.get('best_model'):
            print(f"   ★ NAJLEPSZY: {comparison_stats['best_model'].capitalize()} (MSE={comparison_stats['best_mse']:.6f})")

    def run_tests(self, base_params, n_tests=1, models_to_test=None, varying_params=None, save=False):
        all_results = []
        for i in range(n_tests):
            tp = base_params.copy()
            if varying_params:
                for k, v in varying_params.items(): tp[k] = v[i] if i < len(v) else v[-1]
            all_results.append(self.test(tp, i+1, n_tests, models_to_test, save, i==0))
        return all_results

    def test_observation_length_impact(self, base_params, n_observations_list=None, models_to_test=None, k=5, save=False):
        if n_observations_list is None: n_observations_list = [250, 750, 1000, 2000,4000,7000,10000,15000,20000, 35000, 50000, 100000]
        all_results = []
        self.prepare_data_once(base_params['cutoff_km'], base_params['co_ktory'], max(n_observations_list))
        for n_obs in n_observations_list:
            for j in range(k):
                tp = base_params.copy()
                tp['n_observations'] = n_obs
                res = self.test(tp, len(all_results)+1, len(n_observations_list)*k, models_to_test, save=False)
                if res['success']:
                    res['n_observations'], res['k_run'] = n_obs, j
                    all_results.append(res)
        self._plot_observation_length_results(all_results)
        return all_results

    def _clean_label(self, label):
        """Oczyszcza nazwy modeli z podkreślników i końcówek numerycznych."""
        import re
        # Usuwamy końcówki typu _0, _1, -1 itp.
        label = re.sub(r'[-_]\d+$', '', label)
        # Zamieniamy podkreślniki na spacje i czyścimy białe znaki
        label = label.replace('_', ' ').strip()
        # Poprawiamy znane nazwy dla lepszej estetyki
        replacements = {
            'Model Lenka': 'Model Lenka',
            'Model Aksjomatyczny preparamed': 'Model Aksjomatyczny (P)',
            'Model Aksjomatyczny': 'Model Aksjomatyczny',
            'Model Dirichleta': 'Model Dirichleta',
            'Model wygładzania przestrzennego': 'Wygładzanie Przestrzenne'
        }
        for old, new in replacements.items():
            if label == old:
                return new
        return label

    def _plot_observation_length_results(self, results):
        if not results: return
        data = []
        for r in results:
            if not r['success']: continue
            row = {'n_observations': r['n_observations'], 'k_run': r['k_run']}
            for m, met in r['metrics'].items():
                # Pomijamy model Dirichleta ze względu na zazwyczaj znacznie gorsze wyniki
                if 'dirichlet' in m.lower():
                    continue
                for name, val in met.items():
                    if isinstance(val, (int, float)): 
                        row[f"{m}_{name.lower()}"] = float(val)
            data.append(row)
        
        df = pd.DataFrame(data)
        mean_df = df.groupby('n_observations').mean(numeric_only=True).reset_index()
        
        # Lista metryk do narysowania (np. mse, mae, rmse)
        available_metrics = set()
        for col in mean_df.columns:
            if '_' in col and col not in ['n_observations', 'k_run']:
                metric_name = col.split('_')[-1]
                available_metrics.add(metric_name)
        
        # Tworzymy osobny wykres dla każdej metryki
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ustawienia estetyczne
        plt.style.use('bmh') # Stabilny i czytelny styl
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.autolayout': False
        })
        
        for metric in available_metrics:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            metric_cols = [c for c in mean_df.columns if c.endswith(f'_{metric}')]
            
            if not metric_cols:
                continue
                
            for col in metric_cols:
                raw_label = col.replace(f'_{metric}', '')
                label = self._clean_label(raw_label)
                ax.plot(mean_df['n_observations'], mean_df[col], marker='o', linewidth=1.5, markersize=5, label=label, alpha=0.8)
            
            ax.set_title(f"Wpływ liczby obserwacji na {metric.upper()}", pad=15, fontweight='bold')
            ax.set_xlabel("Liczba obserwacji (skala log)", labelpad=10)
            ax.set_ylabel(f"Wartość {metric.upper()} (skala log)", labelpad=10)
            ax.set_yscale('log')
            ax.set_xscale('log')
            
            # Legenda umieszczona obok wykresu, by nie zasłaniała danych
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=True, fancybox=True, shadow=True)
            
            # Siatka: główna i pomocnicza (delikatna)
            ax.grid(True, which="major", ls="-", alpha=0.6)
            ax.grid(True, which="minor", ls=":", alpha=0.3)
            
            plt.tight_layout()
            file_name = f"observation_length_impact_{metric}_{timestamp}.png"
            plt.savefig(file_name, bbox_inches='tight')
            print(f"  ✔ Zapisano wykres: {file_name}")
            plt.close()
        
        # --- BOXPLOTY ---
        unique_n_obs = sorted(df['n_observations'].unique())
        if len(unique_n_obs) >= 5:
            indices = [0, len(unique_n_obs)//4, len(unique_n_obs)//2, 3*len(unique_n_obs)//4, -1]
            target_n_obs = [unique_n_obs[i] for i in indices]
        else:
            target_n_obs = unique_n_obs
            
        print(f"▶ [PLOT] Tworzenie boxplotów dla n_obs: {target_n_obs}")
        
        for metric in available_metrics:
            fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
            
            # Przygotowujemy dane do boxplota (long format)
            metric_cols = [c for c in df.columns if c.endswith(f'_{metric}')]
            df_filtered = df[df['n_observations'].isin(target_n_obs)].copy()
            
            # Przekształcamy na format długi dla seaborn
            df_long = pd.melt(df_filtered, id_vars=['n_observations'], value_vars=metric_cols,
                              var_name='Model', value_name=metric.upper())
            
            # Czyścimy nazwy modeli
            df_long['Model'] = df_long['Model'].str.replace(f'_{metric}', '').apply(self._clean_label)
            
            sns.boxplot(data=df_long, x='n_observations', y=metric.upper(), hue='Model', palette="Set2", linewidth=1.0, ax=ax)
            
            ax.set_title(f"Rozkład {metric.upper()} dla wybranych liczb obserwacji", pad=15, fontweight='bold')
            ax.set_xlabel("Liczba obserwacji", labelpad=10)
            ax.set_ylabel(f"Wartość {metric.upper()} (skala log)", labelpad=10)
            ax.set_yscale('log')
            ax.grid(True, axis='y', alpha=0.3, ls="--")
            
            # Legenda obok wykresu
            ax.legend(title="Modele", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=True)
            
            plt.tight_layout()
            file_name = f"observation_length_impact_boxplot_{metric.upper()}_{timestamp}.png"
            plt.savefig(file_name, bbox_inches='tight')
            print(f"  ✔ Zapisano boxplot: {file_name}")
            plt.close()

    def test_lengthscale_impact(self, base_params, lengthscale_list=None, models_to_test=None, k=5, save=False):
        if lengthscale_list is None: lengthscale_list = [500, 1000, 2000, 5000, 10000]
        all_results = []
        
        for ls in lengthscale_list:
            print(f"\n📏 TESTOWANIE LENGTHSCALE: {ls}")
            # Update lengthscale in model params
            current_models = []
            for name, params in models_to_test:
                new_params = params.copy()
                new_params['lengthscale'] = ls
                current_models.append((name, new_params))
            
            for j in range(k):
                res = self.test(base_params, len(all_results)+1, len(lengthscale_list)*k, current_models, save=save)
                if res['success']:
                    res['lengthscale'], res['k_run'] = ls, j
                    all_results.append(res)
        
        self._plot_lengthscale_results(all_results)
        return all_results

    def _plot_lengthscale_results(self, results):
        if not results: return
        data = []
        for r in results:
            if not r['success']: continue
            row = {'lengthscale': r['lengthscale'], 'k_run': r['k_run']}
            for m, met in r['metrics'].items():
                for name, val in met.items():
                    if isinstance(val, (int, float)): row[f"{m}_{name.lower()}"] = float(val)
            data.append(row)
        df = pd.DataFrame(data)
        mean_df = df.groupby('lengthscale').mean(numeric_only=True).reset_index()
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.style.use('bmh')
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
        })
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        for col in mean_df.columns:
            if col.endswith('_mse'):
                raw_label = col.replace('_mse','')
                label = self._clean_label(raw_label)
                ax.plot(mean_df['lengthscale'], mean_df[col], marker='s', linewidth=1.5, markersize=6, label=label, alpha=0.8)
        
        ax.set_title("Wpływ parametru Lengthscale na błąd MSE", pad=15, fontweight='bold')
        ax.set_xlabel("Wartość Lengthscale (skala log)", labelpad=10)
        ax.set_ylabel("Średni błąd kwadratowy MSE (skala log)", labelpad=10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, which="major", ls="-", alpha=0.6)
        ax.grid(True, which="minor", ls=":", alpha=0.3)
        
        plt.tight_layout()
        file_name = f"lengthscale_impact_{timestamp}.png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"  ✔ Zapisano wykres: {file_name}")
        plt.close()

    def _plot_critical_difference(self, avg_ranks, cd, k_runs):
        """Plots a Critical Difference diagram."""
        print(f"▶ [PLOT] Tworzenie wykresu Critical Difference (CD={cd:.4f})...")
        
        sorted_ranks = avg_ranks.sort_values()
        names = [self._clean_label(n) for n in sorted_ranks.index]
        values = sorted_ranks.values
        
        n_models = len(names)
        
        plt.style.use('bmh')
        fig, ax = plt.subplots(figsize=(10, 3 + n_models * 0.4), dpi=300)
        
        # Draw horizontal axis for ranks
        ax.axhline(0, color='black', linewidth=1.2)
        rank_min = 1
        rank_max = max(n_models, 5) # Minimum 5 for visibility
        ax.set_xlim(rank_min - 0.5, rank_max + 0.5)
        ax.set_xticks(range(int(rank_min), int(rank_max) + 1))
        ax.set_xlabel('Średnia ranga (im niższa, tym lepiej)', labelpad=10)
        
        # Plot model ranks
        for i, (name, rank) in enumerate(zip(names, values)):
            y_pos = -(i + 1) * 0.5
            ax.plot([rank, rank], [0, y_pos], color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.plot(rank, 0, 'ro', markersize=6)
            ax.text(rank, y_pos - 0.1, f"{name}\n({rank:.2f})", 
                     ha='center', va='top', fontweight='bold', fontsize=9)

        # Find cliques (groups within CD)
        cliques = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if values[j] - values[i] <= cd and cd > 0:
                    cliques.append((values[i], values[j]))
        
        # Draw CD bar
        if cd > 0:
            y_cd = 0.6
            ax.plot([rank_min, rank_min + cd], [y_cd, y_cd], color='#2c3e50', linewidth=3)
            ax.text(rank_min + cd/2, y_cd + 0.1, f'CD = {cd:.3f}', ha='center', color='#2c3e50', fontweight='bold', fontsize=10)

            # Draw cliques
            for i, (start, end) in enumerate(cliques):
                y_clique = 0.4 - (i * 0.08)
                ax.plot([start, end], [y_clique, y_clique], color='black', linewidth=2.5, alpha=0.7)

        ax.set_title(f'Critical Difference Diagram (Nemenyi, α=0.05, runs={k_runs})', pad=25, fontweight='bold')
        ax.get_yaxis().set_visible(False)
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"nemenyi_cd_diagram_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png", bbox_inches='tight')
        plt.close()

    def test_nemenei(self, base_params, models_to_test=None, n_observations=None, k=10, save=False):
        """Runs k tests for a specific n_observations and performs Nemenyi-style rank analysis."""
        if n_observations is not None:
            base_params = base_params.copy()
            base_params['n_observations'] = n_observations
            
        print(f"\n🎯 ROZPOCZYNANIE ANALIZY NEMENYI ({k} powtórzeń, n_obs={base_params['n_observations']})")
        
        all_metrics = []
        for i in range(k):
            # Using the 'test' function as requested
            res = self.test(base_params, i + 1, k, models_to_test, save=save)
            if res['success']:
                # Extract MSE for each model
                run_mse = {model_key: m_val.get('MSE', 0) for model_key, m_val in res['metrics'].items()}
                all_metrics.append(run_mse)
        
        if not all_metrics:
            print("❌ Brak wyników do analizy.")
            return None

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(all_metrics)
        print("\n📈 Wyniki MSE (surowe):")
        print(df)

        # Calculate ranks for each run (lower MSE is better -> lower rank)
        ranks = df.rank(axis=1, ascending=True)
        avg_ranks = ranks.mean()
        
        print("\n📊 Średnie rangi (im niższa, tym lepiej):")
        for model, avg_rank in avg_ranks.items():
            print(f"   - {model}: {avg_rank:.2f}")

        # Friedman test
        try:
            stat, p_value = stats.friedmanchisquare(*(df[col] for col in df.columns))
            print(f"\n⚖️ Test Friedmana: Statystyka={stat:.4f}, p-value={p_value:.4e}")
            
            if p_value < 0.05:
                print("✅ Istnieją statystycznie istotne różnice między modelami (p < 0.05).")
                
                # Nemenyi post-hoc: CD = q_alpha * sqrt( k*(k+1) / (6*N) )
                # q_alpha values for alpha=0.05
                q_alpha_005 = {
                    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 
                    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
                }
                n_models = len(df.columns)
                q_val = q_alpha_005.get(n_models, 3.2)
                
                cd = q_val * np.sqrt((n_models * (n_models + 1)) / (6 * k))
                print(f"🔍 Masa Krytyczna (Critical Difference, CD) = {cd:.4f}")
                
                print("\n🔍 Różnice między średnimi rangami:")
                sorted_models = avg_ranks.sort_values().index
                for i in range(len(sorted_models)):
                    for j in range(i + 1, len(sorted_models)):
                        m1, m2 = sorted_models[i], sorted_models[j]
                        diff = abs(avg_ranks[m1] - avg_ranks[m2])
                        sig = "★ ISTOTNA" if diff > cd else "nieistotna"
                        print(f"   - {m1} vs {m2}: różnica rang = {diff:.2f} ({sig})")
                
                # Plot CD Diagram
                self._plot_critical_difference(avg_ranks, cd, k)
                
            else:
                print("ℹ️ Brak statystycznie istotnych różnic między modelami (p >= 0.05).")
                self._plot_critical_difference(avg_ranks, 0, k)
        except Exception as e:
            print(f"⚠️ Błąd podczas obliczeń statystycznych: {e}")
            import traceback
            traceback.print_exc()

        return {
            'mse_results': df,
            'ranks': ranks,
            'avg_ranks': avg_ranks
        }
