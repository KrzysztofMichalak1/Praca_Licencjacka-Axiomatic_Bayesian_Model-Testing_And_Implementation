
"""This module contains the ResultsDatabase class for storing test results."""
import os
import datetime
import numpy as np
import pandas as pd

class ResultsDatabase:
    """Database for storing results of multiple tests."""
    
    def __init__(self, db_path="wyniki_testow.csv"):
        self.db_path = db_path
        self.initialize_database()
    
    def initialize_database(self):
        """Initializes the database if it doesn't exist."""
        columns = [
            'test_id', 'timestamp', 'n_points', 'n_observations',
            'lengthscale', 'variance', 'mcmc_samples', 
            'mcmc_burn', 'mcmc_scale', 'mcmc_seed',
            'bayesian_name', 'bayesian_mse', 'bayesian_mae', 'bayesian_rmse', 'bayesian_correlation', 'bayesian_covariance',
            'dirichlet_name', 'dirichlet_mse', 'dirichlet_mae', 'dirichlet_rmse', 'dirichlet_correlation', 'dirichlet_covariance',
            'gp_name', 'gp_mse', 'gp_mae', 'gp_rmse', 'gp_correlation', 'gp_covariance',
            'spatial_name', 'spatial_mse', 'spatial_mae', 'spatial_rmse', 'spatial_correlation', 'spatial_covariance',
            'mse_diff', 'mae_diff', 'correlation_diff', 'better_model_mse', 'better_model_mae',
            'bayesian_better_count', 'dirichlet_better_count', 'gp_better_count', 'spatial_better_count', 'equal_count',
            'wilcoxon_pvalue', 'test_duration_seconds'
        ]
        if not os.path.exists(self.db_path):
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.db_path, index=False)
            print(f"▶ [DB] Utworzono nową bazę danych: {self.db_path}")
        else:
            # Sprawdź czy są wszystkie kolumny (wsteczna kompatybilność)
            df = pd.read_csv(self.db_path)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                for c in missing: df[c] = np.nan
                df.to_csv(self.db_path, index=False)
                print(f"▶ [DB] Zaktualizowano strukturę bazy danych o: {missing}")

    def save_test_results(self, test_params, metrics_bayesian, metrics_dirichlet, 
                         metrics_gp, metrics_spatial, comparison_stats, duration,
                         names=None):
        """Saves the results of a single test to the database."""
        if names is None: names = {}
        df = pd.read_csv(self.db_path)
        test_id = f"test_{len(df) + 1:04d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Znajdź najlepszy model wg MSE
        models_mse = {
            names.get('bayesian', 'Bayesian'): metrics_bayesian.get('MSE', np.nan),
            names.get('dirichlet', 'Dirichlet'): metrics_dirichlet.get('MSE', np.nan),
            names.get('gp', 'GP'): metrics_gp.get('MSE', np.nan),
            names.get('spatial', 'Spatial'): metrics_spatial.get('MSE', np.nan)
        }
        valid_models_mse = {k: v for k, v in models_mse.items() if not np.isnan(v)}
        best_model_mse = min(valid_models_mse, key=valid_models_mse.get) if valid_models_mse else 'N/A'
        
        new_row = {
            'test_id': test_id,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_points': test_params.get('n_points', 0),
            'n_observations': test_params.get('n_observations', 0),
            'lengthscale': test_params.get('lengthscale', 0),
            'variance': test_params.get('variance', 0),
            'mcmc_samples': test_params.get('mcmc_samples', 0),
            'mcmc_burn': test_params.get('mcmc_burn', 0),
            'mcmc_scale': test_params.get('mcmc_scale', 0),
            'mcmc_seed': test_params.get('mcmc_seed', 0),
            
            'bayesian_name': names.get('bayesian', ''),
            'bayesian_mse': float(metrics_bayesian.get('MSE', np.nan)),
            'bayesian_mae': float(metrics_bayesian.get('MAE', np.nan)),
            'bayesian_rmse': float(metrics_bayesian.get('RMSE', np.nan)),
            'bayesian_correlation': float(metrics_bayesian.get('Correlation', np.nan)),
            'bayesian_covariance': float(metrics_bayesian.get('Covariance', np.nan)),
            
            'dirichlet_name': names.get('dirichlet', ''),
            'dirichlet_mse': float(metrics_dirichlet.get('MSE', np.nan)),
            'dirichlet_mae': float(metrics_dirichlet.get('MAE', np.nan)),
            'dirichlet_rmse': float(metrics_dirichlet.get('RMSE', np.nan)),
            'dirichlet_correlation': float(metrics_dirichlet.get('Correlation', np.nan)),
            'dirichlet_covariance': float(metrics_dirichlet.get('Covariance', np.nan)),
            
            'gp_name': names.get('gp', ''),
            'gp_mse': float(metrics_gp.get('MSE', np.nan)),
            'gp_mae': float(metrics_gp.get('MAE', np.nan)),
            'gp_rmse': float(metrics_gp.get('RMSE', np.nan)),
            'gp_correlation': float(metrics_gp.get('Correlation', np.nan)),
            'gp_covariance': float(metrics_gp.get('Covariance', np.nan)),
            
            'spatial_name': names.get('spatial', ''),
            'spatial_mse': float(metrics_spatial.get('MSE', np.nan)),
            'spatial_mae': float(metrics_spatial.get('MAE', np.nan)),
            'spatial_rmse': float(metrics_spatial.get('RMSE', np.nan)),
            'spatial_correlation': float(metrics_spatial.get('Correlation', np.nan)),
            'spatial_covariance': float(metrics_spatial.get('Covariance', np.nan)),
            
            'mse_diff': float(metrics_bayesian.get('MSE', np.nan) - metrics_dirichlet.get('MSE', np.nan)),
            'mae_diff': float(metrics_bayesian.get('MAE', np.nan) - metrics_dirichlet.get('MAE', np.nan)),
            'correlation_diff': float(metrics_bayesian.get('Correlation', np.nan) - metrics_dirichlet.get('Correlation', np.nan)),
            'better_model_mse': best_model_mse,
            'better_model_mae': names.get('bayesian', 'Bayesian') if metrics_bayesian.get('MAE', np.inf) < metrics_dirichlet.get('MAE', np.inf) else names.get('dirichlet', 'Dirichlet'),
            
            'bayesian_better_count': int(comparison_stats.get('bayesian_better_count', 0)),
            'dirichlet_better_count': int(comparison_stats.get('dirichlet_better_count', 0)),
            'gp_better_count': int(comparison_stats.get('gp_better_count', 0)),
            'spatial_better_count': int(comparison_stats.get('spatial_better_count', 0)),
            'equal_count': int(comparison_stats.get('equal_count', 0)),
            'wilcoxon_pvalue': float(comparison_stats.get('wilcoxon_pvalue', np.nan)),
            'test_duration_seconds': float(duration)
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.db_path, index=False)
        return test_id
        
        # Dodaj nowy wiersz
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Zapisz zaktualizowaną bazę
        df.to_csv(self.db_path, index=False)
        print(f"  ✔ Wyniki zapisane do bazy danych: {test_id}")
        
        return test_id
    
    def get_summary_stats(self):
        """Returns summary statistics of all tests."""
        if not os.path.exists(self.db_path):
            return None
        
        df = pd.read_csv(self.db_path)
        if len(df) == 0:
            return None
        
        # Zlicz zwycięstwa każdego modelu
        better_model_counts = df['better_model_mse'].value_counts()
        
        summary = {
            'total_tests': len(df),
            'bayesian_wins_mse': better_model_counts.get('Bayesian', 0),
            'dirichlet_wins_mse': better_model_counts.get('Dirichlet', 0),
            'gp_wins_mse': better_model_counts.get('GP', 0),
            'spatial_wins_mse': better_model_counts.get('Spatial', 0),
            'avg_bayesian_mse': float(df['bayesian_mse'].mean()),
            'avg_dirichlet_mse': float(df['dirichlet_mse'].mean()),
            'avg_gp_mse': float(df['gp_mse'].mean()),
            'avg_spatial_mse': float(df['spatial_mse'].mean()),
            'avg_test_duration': float(df['test_duration_seconds'].mean())
        }
        
        return summary
    
    def print_summary(self):
        """Displays a summary of all tests."""
        summary = self.get_summary_stats()
        if summary is None:
            print("Brak danych w bazie.")
            return
        
        print("\n" + "="*60)
        print("📊 PODSUMOWANIE WSZYSTKICH TESTÓW")
        print("="*60)
        print(f"Łączna liczba testów: {summary['total_tests']}")
        print(f"Średni czas testu: {summary['avg_test_duration']:.1f}s")
        
        print(f"\nŚREDNIE MSE:")
        print(f"  - Bayesian Field:   {summary['avg_bayesian_mse']:.6f}")
        print(f"  - Dirichlet:        {summary['avg_dirichlet_mse']:.6f}")
        print(f"  - Gaussian Process: {summary['avg_gp_mse']:.6f}")
        print(f"  - Spatial Smoothing:{summary['avg_spatial_mse']:.6f}")
        
        print(f"\nZWYCIĘSTWA WG MSE:")
        total_wins = summary['total_tests']
        print(f"  - Bayesian Field:   {summary['bayesian_wins_mse']} ({summary['bayesian_wins_mse']/total_wins*100:.1f}%)")
        print(f"  - Dirichlet:        {summary['dirichlet_wins_mse']} ({summary['dirichlet_wins_mse']/total_wins*100:.1f}%)")
        print(f"  - Gaussian Process: {summary['gp_wins_mse']} ({summary['gp_wins_mse']/total_wins*100:.1f}%)")
        print(f"  - Spatial Smoothing:{summary['spatial_wins_mse']} ({summary['spatial_wins_mse']/total_wins*100:.1f}%)")
