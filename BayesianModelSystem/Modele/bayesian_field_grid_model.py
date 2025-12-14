import numpy as np
from .bayesian_field_model import BayesianFieldModel

class BayesianFieldModelCVGridSearch(BayesianFieldModel):
    """
    Rozszerzenie modelu `BayesianFieldModel`, które automatycznie wyszukuje
    optymalną wartość hiperparametru `lengthscale` za pomocą przeszukiwania
    siatki (grid search) z walidacją krzyżową (cross-validation).

    Algorytm działania:
    1. Inicjalizacja:
       - Podobna do `BayesianFieldModel`, ale zamiast pojedynczego `lengthscale`
         przyjmuje siatkę wartości do przetestowania (`lengthscale_grid`) oraz
         liczbę fałd do walidacji krzyżowej (`cv_k`).

    2. Grid Search z Walidacją Krzyżową (`przygotuj_apriori`):
       - Cel: Znalezienie `lengthscale` z siatki, które najlepiej generalizuje
         do niewidzianych danych.
       - Model iteruje po wszystkich wartościach `lengthscale` z podanej siatki.
       - Dla każdej wartości `lengthscale`:
         a) Zbiór obserwowanych danych jest dzielony na `cv_k` rozłącznych części (fałd).
         b) Uruchamiana jest pętla walidacji krzyżowej: `cv_k` razy model jest
            trenowany na `cv_k - 1` częściach danych i walidowany na pozostałej.
         c) Jako miara błędu używana jest uśredniona log-wiarygodność na zbiorach
            walidacyjnych.
       - Wybierana jest wartość `lengthscale`, która dała najlepszy (najwyższy)
         średni wynik log-wiarygodności w procesie CV.

    3. Finalna pętla MCMC (`przygotuj_predykcyjny`):
       - Używając znalezionego optymalnego `lengthscale`, model uruchamia
         pełną, długą symulację MCMC, dokładnie tak jak w bazowym `BayesianFieldModel`.

    4. Obliczenie predykcji (`posterior_mean`):
       - Wyniki z finalnej pętli MCMC są uśredniane do uzyskania ostatecznej
         predykcji, analogicznie do `BayesianFieldModel`.
    """
    def __init__(self, space_points, metric_func, observed_indices,
                 variance, distance_unit='km', lengthscale_grid=None, 
                 cv_k=5, cv_mcmc_samples=1000, cv_mcmc_burn=500):
        
        super().__init__(space_points, metric_func, observed_indices, 1, variance, distance_unit)
        self.lengthscale_grid = lengthscale_grid or [500, 1500, 2500, 4000, 6000]
        self.cv_k = cv_k
        self.cv_mcmc_samples = cv_mcmc_samples
        self.cv_mcmc_burn = cv_mcmc_burn
        self._score_cache = {}
        print(f"\n? [CV-GRID-SEARCH MODEL] Inicjalizacja z K={self.cv_k} i siatk?: {self.lengthscale_grid}")

    def _evaluate_lengthscale_cv(self, ls):
        """Evaluates a given 'lengthscale' using K-fold cross-validation."""
        if ls in self._score_cache:
            print(f"  - Pobieranie z cache dla lengthscale={ls}")
            return self._score_cache[ls]

        print(f"--- Ewaluacja lengthscale = {ls} (z K={self.cv_k} walidacj?) ---")
        shuffled_indices = np.random.permutation(self.observed_indices)
        folds = np.array_split(shuffled_indices, self.cv_k)
        fold_scores = []

        for k in range(self.cv_k):
            val_indices = folds[k]
            if len(val_indices) == 0: continue
            
            train_indices = np.concatenate([folds[i] for i in range(self.cv_k) if i != k])
            
            try:
                temp_model = BayesianFieldModel(
                    space_points=self.space_points, metric_func=self.metric,
                    observed_indices=train_indices, lengthscale=ls,
                    variance=self.variance, distance_unit=self.distance_unit
                )
                temp_model.przygotuj_apriori()
                temp_model.przygotuj_predykcyjny(
                    num_samples=self.cv_mcmc_samples, burn_in=self.cv_mcmc_burn, 
                    proposal_scale=0.05, seed=42
                )
                pred_probs = temp_model.posterior_mean()
                
                score = np.sum(np.log(pred_probs[val_indices] + 1e-9))
                fold_scores.append(score)
            except Exception as e:
                print(f"    Fold {k+1}/{self.cv_k} B??D: {e}")
                fold_scores.append(-np.inf)

        avg_score = np.mean(fold_scores) if fold_scores else -np.inf
        print(f"  > ?redni wynik dla lengthscale={ls}: {avg_score:.2f}\n")
        self._score_cache[ls] = avg_score
        return avg_score

    def przygotuj_apriori(self):
        print("? [CV GRID SEARCH] Rozpoczynanie poszukiwania 'lengthscale'...")
        best_ls = -1
        best_avg_score = -np.inf
        
        for ls in self.lengthscale_grid:
            avg_score = self._evaluate_lengthscale_cv(ls)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_ls = ls

        if best_ls == -1:
            raise ValueError("Grid search z walidacj? krzy?ow? nie znalaz? poprawnego 'lengthscale'.")
        
        print(f"? Najlepszy 'lengthscale' (CV): {best_ls} (wynik: {best_avg_score:.2f})")
        self.lengthscale = best_ls
        
        print("\n? [APRIORI] Finalne przygotowanie z najlepszym 'lengthscale'...")
        super().przygotuj_apriori()