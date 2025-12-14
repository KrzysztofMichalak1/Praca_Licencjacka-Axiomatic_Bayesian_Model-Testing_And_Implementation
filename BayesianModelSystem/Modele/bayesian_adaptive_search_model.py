
import numpy as np
from .bayesian_field_model import BayesianFieldModel
from .bayesian_helpers import log_posterior_fast

class BayesianFieldModelAdaptiveSearchBinary(BayesianFieldModel):
    """
    Rozszerzenie modelu `BayesianFieldModel`, które automatycznie wyszukuje
    optymalną wartość hiperparametru `lengthscale` za pomocą wyszukiwania binarnego.

    Algorytm działania jest bardzo podobny do `LenkAdaptiveSearchModel`, ale różni
    się strategią wyszukiwania:

    2. Adaptacyjne wyszukiwanie `lengthscale` (`przygotuj_apriori`):
       - Cel: Znalezienie `lengthscale`, które maksymalizuje wiarygodność brzegową.
       - Model przeprowadza wyszukiwanie binarne w zadanym zakresie `lengthscale`.
       - W każdej iteracji:
         a) Sprawdzany jest środkowy punkt aktualnego przedziału `lengthscale`.
         b) Uruchamiana jest krótka symulacja MCMC do estymacji wiarygodności
            brzegowej w tym punkcie.
         c) Porównuje się wiarygodność w punkcie środkowym z wiarygodnościami na
            krańcach przedziału, aby zdecydować, którą połowę przedziału odrzucić.
       - Proces jest powtarzany, zawężając przedział poszukiwań, aż do znalezienia
         optymalnej wartości `lengthscale`.

    Pozostałe kroki (Inicjalizacja, Finalna pętla MCMC, Obliczenie predykcji) są
    analogiczne do `LenkAdaptiveSearchModel` i bazowego `BayesianFieldModel`.
    """
    def __init__(self, space_points, metric_func, observed_indices,
                 variance, distance_unit='km',
                 start_ls=3000, step_size=2000, k_steps=3,
                 num_samples_search=100, burn_in_search=50, 
                 proposal_scale_search=0.05):
        
        super().__init__(space_points, metric_func, observed_indices, 1, variance, distance_unit)
        self.start_ls = start_ls
        self.step_size = step_size
        self.k_steps = k_steps
        self.num_samples_search = num_samples_search
        self.burn_in_search = burn_in_search
        self.proposal_scale_search = proposal_scale_search
        print(f"\n▶ [ADAPTIVE-BINARY-SEARCH] Inicjalizacja:")
        print(f"   - Start lengthscale: {start_ls}")
        print(f"   - Step size: {step_size}")
        print(f"   - Kroki bisekcji (k): {k_steps}")

    def _evaluate_lengthscale(self, ls):
        """Quickly evaluates a lengthscale."""
        try:
            temp_model = BayesianFieldModel(
                self.space_points, self.metric, self.observed_indices, 
                ls, self.variance, self.distance_unit
            )
            temp_model.przygotuj_apriori()
            temp_model.przygotuj_predykcyjny(
                num_samples=self.num_samples_search, 
                burn_in=self.burn_in_search, 
                proposal_scale=self.proposal_scale_search, 
                seed=42
            )
            
            if temp_model.samples_w is None or len(temp_model.samples_w) == 0:
                return -np.inf
                
            n_samples = min(50, len(temp_model.samples_w))
            log_post_samples = []
            for w in temp_model.samples_w[:n_samples]:
                log_post = log_posterior_fast(w, self.counts_f, 
                                            temp_model.mvn_u.L, 
                                            temp_model.mvn_u.log_norm_const)
                if np.isfinite(log_post):
                    log_post_samples.append(log_post)
            
            return np.mean(log_post_samples) if log_post_samples else -np.inf
            
        except Exception as e:
            print(f"    [ERROR] ls={ls}: {str(e)[:50]}...")
            return -np.inf

    def przygotuj_apriori(self):
        print("▶ [BINARY SEARCH] Rozpoczynanie poszukiwania 'lengthscale'...")
        self.counts_f = self.counts.astype(np.float64)
        
        print(f"\n📌 FAZA 1: Szukanie optimum co {self.step_size}")
        print("-" * 50)
        
        current_ls = self.start_ls
        current_score = self._evaluate_lengthscale(current_ls)
        
        print(f"  Start: ls={current_ls:.0f}, score={current_score:.2f}")
        
        iteration = 1
        i=0
        while i<20:
            i+=1

            next_ls = current_ls + self.step_size
            next_score = self._evaluate_lengthscale(next_ls)
            
            print(f"  Krok {iteration}: ls={next_ls:.0f}, score={next_score:.2f}")
            
            if next_score <= current_score:
                print(f"  ⬆️  Optimum znalezione: ls={current_ls:.0f} (następny krok gorszy)")
                break
            else:
                current_ls = next_ls
                current_score = next_score
                iteration += 1
        
        best_ls = current_ls
        best_score = current_score
        current_step = self.step_size
        
        print(f"\n✅ ZGRUBNE OPTIMUM: ls={best_ls:.0f}, score={best_score:.2f}")
        print(f"   Aktualny step: {current_step:.0f}")
        
        print(f"\n📌 FAZA 2: {self.k_steps} kroków bisekcji")
        print("-" * 50)
        
        for k in range(1, self.k_steps + 1):
            print(f"\n  🔄 KROK BISEKCJI {k}/{self.k_steps}:")
            
            current_step = current_step / 2.0
            print(f"    Nowy step: {current_step:.0f} (połowa poprzedniego)")
            
            test_points = [
                best_ls - current_step,
                best_ls,
                best_ls + current_step
            ]
            
            test_points = [max(100, p) for p in test_points]
            
            print(f"    Testowane punkty: {[f'{p:.0f}' for p in test_points]}")
            
            scores = {}
            for ls in test_points:
                score = self._evaluate_lengthscale(ls)
                scores[ls] = score
                print(f"      ls={ls:.0f}: score={score:.2f}")
            
            new_best_ls = max(scores, key=scores.get)
            new_best_score = scores[new_best_ls]
            
            if new_best_score > best_score:
                print(f"    ✅ Znaleziono lepszy: {new_best_ls:.0f} "
                      f"(poprawa: {new_best_score - best_score:.2f})")
                best_ls = new_best_ls
                best_score = new_best_score
            else:
                print(f"    ℹ️  Najlepszy pozostaje: {best_ls:.0f}")
        
        self.lengthscale = best_ls
        print(f"\n🎯 KONIEC OPTYMALIZACJI")
        print(f"   Finalny lengthscale: {self.lengthscale:.0f}")
        print(f"   Finalny score: {best_score:.2f}")
        print(f"   Finalny step: {current_step:.1f}")
        print("=" * 60)
        
        print("\n▶ [APRIORI] Finalne przygotowanie z najlepszym 'lengthscale'...")
        super().przygotuj_apriori()
