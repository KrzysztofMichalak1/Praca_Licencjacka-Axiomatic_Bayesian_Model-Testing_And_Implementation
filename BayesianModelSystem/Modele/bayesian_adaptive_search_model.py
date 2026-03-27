import numpy as np
from .bayesian_field_model import BayesianFieldModel

class BayesianFieldModelAdaptiveSearchBinary(BayesianFieldModel):
    """
    Zoptymalizowana klasa estymująca stałą 'c' (lengthscale) 
    z wykorzystaniem relatywistycznej metody momentów i poprawki Bessela.
    """
    def __init__(self, space_points, metric_func, observed_indices,
                 variance, distance_unit='km',p=1, **kwargs):
        # Przyjmujemy **kwargs, aby TestManager nie wywalał błędu przy przekazywaniu 
        # start_ls, step_size itd. (starych parametrów bisekcji)
        super().__init__(space_points, metric_func, observed_indices, 1.0, variance, distance_unit,p)
        
    def przygotuj_apriori(self):
        print("\n▶ [MOMENT-ESTIMATOR] Rozpoczynanie analitycznej estymacji 'c'...")
        
        # 1. Przygotowanie danych (Logarytmowanie z pseudocountem delta=1)
        # counts_f musi być spłaszczone, by pasowało do obliczeń wektorowych
        counts_f = self.counts.astype(np.float64).flatten()
        Z = np.log(counts_f + 1.0)
        Z_bar = np.mean(Z)
        n = len(Z)
        
        print(f"   - Liczba obserwacji (n): {n}")
        
        # 2. Obliczanie średnich odległości metrycznych (mi_bar_i)
        # NAPRAWA INDEKSOWANIA: Wymuszamy typ int i spłaszczamy indeksy
        pts = np.array(self.space_points)
        idx = np.array(self.observed_indices).astype(int).flatten()
        
        # Pobieramy punkty, które faktycznie widzieliśmy
        obs_points = pts[idx]
        
        print("   - Obliczanie geometrii przestrzeni (średnie odległości mi)...")
        
        # Macierz odległości (n x n)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # self.metric to u Ciebie prawdopodobnie haversine
                d = self.metric(obs_points[i], obs_points[j])**(self.p)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        print(self.p)
        # mi_bar_i to średni dystans od punktu i do wszystkich innych (w tym do siebie, co daje 0)
        mi_bar = np.mean(dist_matrix, axis=1)
        
        # Unikamy dzielenia przez zero (jeśli punkty by się pokrywały)
        mi_bar = np.where(mi_bar == 0, 1e-9, mi_bar)
        
        # 3. Relatywistyczny Estymator c^2 z poprawką Bessela (n-1)
        print(f"   - Stosowanie poprawki Bessela (n-1 = {n-1})")
        
        squared_diffs = (Z - Z_bar)**2
        relative_variances = squared_diffs / mi_bar
        
        c_squared = 1/((1.0 / (n - 1)) * np.sum(relative_variances))
        
        # Przypisujemy wynik do lengthscale
        self.lengthscale = c_squared
        
        print(f"✅ ESTYMACJA ZAKOŃCZONA:")
        print(f"   - Średnia log-gęstość (Z_bar): {Z_bar:.4f}")
        print(f"   - Wyznaczona stała c (lengthscale): {self.lengthscale:.6f}")
        print("-" * 50)

        # 4. Finalne przygotowanie macierzy kowariancji z nowym 'ls'
        super().przygotuj_apriori()