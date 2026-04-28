import numpy as np
from .model_aksjomatyczny_preparamed import ModelAksjomatycznyPreparamed

class ModelAksjomatyczny(ModelAksjomatycznyPreparamed):
    """
    Model implementujący estymator Najwyższej Wiarygodności (MLE) dla parametru c 
    w oparciu o przyrosty log-ilorazu szans (log-odds). [cite: 104, 118]
    Zgodnie z Twierdzeniem 1: lengthscale = 1/c. [cite: 164, 165]
    """
    def __init__(self, space_points, metric_func, observed_indices,
                 variance, distance_unit='km',p=1, **kwargs):
        # Przyjmujemy **kwargs, aby TestManager nie wywalał błędu przy przekazywaniu 
        # start_ls, step_size itd. (starych parametrów bisekcji)
        super().__init__(space_points, metric_func, observed_indices, 1.0, variance, distance_unit,p)
        
    def przygotuj_apriori(self):
        print("\n▶ [MOMENT-ESTIMATOR] Rozpoczynanie analitycznej estymacji 'c'...")
        
        # Filtrujemy tylko te punkty, które mają jakiekolwiek obserwacje
        obs_mask = self.counts > 0
        observed_counts = self.counts[obs_mask].astype(np.float64)
        n_obs_points = len(observed_counts)
        
        if n_obs_points < 2:
            print("  ⚠️ Zbyt mało punktów z obserwacjami do estymacji. Używam domyślnego lengthscale.")
            # Jeśli n_obs_points to 1, lengthscale pozostaje 1.0 (z __init__) lub bierzemy z bazy
        else:
            print(f"   - Punkty z obserwacjami: {n_obs_points} / {self.n}")
            
            # 1. Przygotowanie danych (Logarytmowanie z pseudocountem delta=1)
            Z = np.log(observed_counts + 1.0)
            Z_bar = np.mean(Z)
            
            # 2. Obliczanie średnich odległości metrycznych (mi_bar_i)
            pts = np.array(self.space_points)[obs_mask]
            
            print("   - Obliczanie geometrii przestrzeni (średnie odległości mi)...")
            
            # Macierz odległości (n_obs x n_obs)
            dist_matrix = np.zeros((n_obs_points, n_obs_points))
            for i in range(n_obs_points):
                for j in range(i + 1, n_obs_points):
                    d = self.metric(pts[i], pts[j], return_unit=self.distance_unit)**(self.p)
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d
            
            # mi_bar_i to średni dystans od punktu i do wszystkich innych punktów z obserwacjami
            mi_bar = np.mean(dist_matrix, axis=1)
            mi_bar = np.where(mi_bar == 0, 1e-9, mi_bar)
            
            # 3. Relatywistyczny Estymator c^2 z poprawką Bessela (n-1)
            print(f"   - Stosowanie poprawki Bessela (n-1 = {n_obs_points-1})")
            
            squared_diffs = (Z - Z_bar)**2
            relative_variances = squared_diffs / mi_bar
            
            c_squared = 1/((1.0 / (n_obs_points - 1)) * np.sum(relative_variances))
            self.lengthscale = c_squared
            
            print(f"✅ ESTYMACJA ZAKOŃCZONA:")
            print(f"   - Średnia log-gęstość (Z_bar): {Z_bar:.4f}")
            print(f"   - Wyznaczona stała c (lengthscale): {self.lengthscale:.6f}")

        print("-" * 50)
        # 4. Finalne przygotowanie macierzy kowariancji z nowym 'ls'
        super().przygotuj_apriori()