"""This module contains the DirichletModel class."""
import numpy as np
from scipy.stats import dirichlet

class DirichletModel:
    """
    Model bazujący na rozkładzie Dirichleta, który jest sprzężonym priorem
    dla rozkładu wielomianowego (Multinomial). Jest to prosty model bazowy, który nie
    uwzględnia korelacji przestrzennych w sposób jawny.

    Algorytm działania:
    1. Inicjalizacja:
       - Model przyjmuje listę indeksów wszystkich dokonanych obserwacji (`obs_idx`)
         oraz całkowitą liczbę lokalizacji (`n_points`).
       - Zliczane są obserwacje w każdej lokalizacji, tworząc wektor zliczeń `counts`.
       - Ustawiany jest parametr `alpha` dla prioru Dirichleta. W tym przypadku,
         `alpha = counts + 1`, co jest równoznaczne z zastosowaniem priora
         o wartości `alpha=1` (wygładzanie Laplace'a), a następnie obliczeniem
         parametru `posterior`.

    2. Predykcja (Średnia `posterior`):
       - Metoda `posterior_mean` oblicza wartość oczekiwaną rozkładu `posterior`.
       - Dla rozkładu Dirichleta, jest to wektor `p` o elementach:
         `p_i = alpha_i / sum(alpha)`.
       - Wynikowy wektor `p` reprezentuje oczekiwane prawdopodobieństwo dla każdej
         lokalizacji i jest zwracany jako finalna predykcja modelu.
    """
    
    def __init__(self, observed_indices, n_points):
        print("▶ [DIRICHLET] Inicjalizacja modelu Dirichleta...")
        self.observed_indices = np.asarray(observed_indices)
        self.n = n_points
        self.counts = np.bincount(self.observed_indices, minlength=self.n)
        self.alpha = self.counts + 1  # Laplace smoothing
        print(f"  - Liczba punktów: {self.n}")
        print(f"  - Suma counts: {self.counts.sum()}")
        print(f"  - Parametry alpha: {self.alpha}")
        
    def posterior_mean(self):
        """Dirichlet posterior mean."""
        return self.alpha / self.alpha.sum()
    
    def sample_posterior(self, n_samples=1000):
        """Samples from the posterior Dirichlet distribution."""
        return dirichlet.rvs(self.alpha, size=n_samples)
    
    def posterior_quantiles(self, q=(0.025, 0.975)):
        """Posterior quantiles (approximate)."""
        samples = self.sample_posterior(1000)
        return np.quantile(samples, q, axis=0)
