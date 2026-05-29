import numpy as np
from .model_aksjomatyczny_preparamed import ModelAksjomatycznyPreparamed
from .bayesian_helpers import vectorized_haversine

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
        else:
            print(f"   - Punkty z obserwacjami: {n_obs_points} / {self.n}")
            
            # 1. Przygotowanie danych (Logarytmowanie z pseudocountem delta=1)
            Z = np.log(observed_counts + 1.0)
            
            # 2. Obliczanie geometrii i różnic (Podejście parowe - bardziej stabilne)
            pts = np.array(self.space_points)[obs_mask]
            
            print("   - Obliczanie parowych różnic i odległości (wektorowo)...")
            
            # Wykorzystujemy zoptymalizowaną metrykę jeśli to możliwe
            if self.metric.__name__ == 'haversine':
                dist_matrix = vectorized_haversine(pts, pts, return_unit=self.distance_unit)**(self.p)
            else:
                # Fallback dla innych metryk (nadal pętla, ale rzadsza sytuacja)
                dist_matrix = np.zeros((n_obs_points, n_obs_points))
                for i in range(n_obs_points):
                    for j in range(i + 1, n_obs_points):
                        d = self.metric(pts[i], pts[j], return_unit=self.distance_unit)**(self.p)
                        dist_matrix[i, j] = dist_matrix[j, i] = d
            
            # 3. Estymacja parametru c (współczynnik zmienności na jednostkę dystansu)
            upper_idx = np.triu_indices(n_obs_points, k=1)
            
            # Różnice log-gęstości
            Z_col = Z[:, np.newaxis]
            diff_matrix_sq = (Z_col - Z)**2
            
            dists = dist_matrix[upper_idx]
            diffs = diff_matrix_sq[upper_idx]
            
            valid_mask = dists > 1e-8 # Unikamy bardzo bliskich punktów
            if np.any(valid_mask):
                # Zależność: E[(Z_i - Z_j)^2] = variance * (dist_ij / L)^p
                # => L^p = variance * dist_ij^p / E[(Z_i - Z_j)^2]
                
                # Używamy mediany dla odporności na szum Poissonowski
                ratios = (self.variance * dists[valid_mask]) / (diffs[valid_mask] + 1e-12)
                med_ratio = np.median(ratios)
                if med_ratio > 0:
                    self.lengthscale = (med_ratio)**(1.0 / self.p)
                    # Ograniczamy do sensownego zakresu (np. 1km do 40000km - obwód Ziemi)
                    self.lengthscale = np.clip(self.lengthscale, 1.0, 40000.0)
                else:
                    self.lengthscale = 1000.0
            else:
                self.lengthscale = 1000.0

            c_est = 1.0 / self.lengthscale if self.lengthscale > 0 else 0
            print(f"✅ ESTYMACJA ZAKOŃCZONA: L = {self.lengthscale:.6f} (c = {c_est:.6f})")

        print("-" * 50)
        # 4. Finalne przygotowanie macierzy kowariancji z nowym 'ls'
        super().przygotuj_apriori()