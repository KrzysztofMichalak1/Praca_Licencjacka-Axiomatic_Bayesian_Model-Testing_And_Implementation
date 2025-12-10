#MODELE
#region
#region
#  MODEL BAYES FIELD
#region
from numba import njit
import numpy as np
from scipy.linalg import cholesky, solve_triangular
import math

@njit(fastmath=True)
def x_from_w(w, clip_value=10.0):
    u = w 
    u_clipped = np.clip(u, -clip_value, clip_value)  # ⬅️ ZAPOBIEGA ekstremalnym wartościom
    
    max_u = np.max(u_clipped)
    exp_u = np.exp(u_clipped - max_u)
    sum_exp_u = np.sum(exp_u)
    
    denom = np.exp(-max_u) + sum_exp_u
    x1 = np.exp(-max_u) / denom
    x_rest = exp_u / denom
    
    x = np.empty(w.size + 1)
    x[0] = x1
    x[1:] = x_rest
    return x
def w_from_x(x):
    """Transformacja z x do w z uwzględnieniem standaryzacji"""
    if np.any(x <= 0):
        return None
    x1 = x[0]
    # Najpierw oblicz u
    u = np.log(x[1:] / x1)
    return u
class MultivariateNormalCholesky:
    def __init__(self, Sigma):
        print("▶ [MVN] Budowa rozkładu wielowymiarowego...")
        self.L = cholesky(Sigma, lower=True)
        self.logdet = 2 * np.sum(np.log(np.diag(self.L)))
        self.dim = Sigma.shape[0]
        self.log_norm_const = -0.5*(self.dim*np.log(2*np.pi) + self.logdet)
        print("  ✔ Cholesky ukończone.")

    def logpdf(self, u):  
        y = solve_triangular(self.L, u, lower=True)
        return self.log_norm_const - 0.5*np.dot(y, y)

@njit(fastmath=True)
def log_posterior_fast(w, counts_f, L, log_norm_const):
    """ZOPTYMALIZOWANA wersja log-posterior z prekomputacją"""
    u = w
    max_u = np.max(u)
    exp_u = np.exp(u - max_u)
    denom = np.exp(-max_u) + np.sum(exp_u)
    x0=1/denom
    x1 = np.exp(-max_u)*x0
    x_rest = exp_u * x0

    # Stabilność
    if np.any(x_rest <= 0.0) or x1 <= 0.0:
        return -1e300

    # MVN logpdf (dla u)
    y = solve_lower_triangular(L, u)
    lp = log_norm_const - 0.5 * np.dot(y, y)

    # Log-likelihood
    ll = np.dot(counts_f[1:], np.log(x_rest)) + counts_f[0] * np.log(x1)
   

    return lp + ll
@njit(fastmath=True)
def solve_lower_triangular(L, b):
    m = L.shape[0]
    y = np.empty(m)
    for i in range(m):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = (b[i] - s) / L[i, i]
    return y
#endregion
class BayesianFieldModel:

    def __init__(self, space_points, metric_func, observed_indices,
                 lengthscale, variance, distance_unit='km'):

        print("\n▶ [MODEL] Inicjalizacja modelu...")
        self.space_points = list(space_points)
        self.metric = metric_func
        self.n = len(self.space_points)
        self.m = self.n - 1

        self.observed_indices = np.asarray(observed_indices)
        self.counts = np.bincount(self.observed_indices, minlength=self.n)
        self.N = int(self.counts.sum())

        print(f"  - Liczba punktów: {self.n}")
        print(f"  - Liczba obserwacji: {self.N}")
        self.lengthscale = lengthscale
        self.variance = variance
        self.distance_unit = distance_unit

        self.Sigma_u = None
        self.mvn_u = None
        self.samples_w = None
        self.samples_x = None
        
        self.counts_f = None
        
        print("  ✔ Model gotowy.\n")

    def przygotuj_apriori(self):
        print("▶ [APRIORI] Budowa macierzy kowariancji dla ZNORMALIZOWANYCH u...")
    
        p1 = self.space_points[0]
    
        Sigma_w = np.zeros((self.m, self.m))
        for i in range(self.m):
            pi = self.space_points[i+1]
            d_i1 = self.metric(pi, p1, return_unit=self.distance_unit)/self.lengthscale
            Sigma_w[i, i] = d_i1
        
            for j in range(i+1, self.m):
                pj = self.space_points[j+1]
                d_j1 = self.metric(pj, p1, return_unit=self.distance_unit)/self.lengthscale
                d_ij = self.metric(pi, pj, return_unit=self.distance_unit)/self.lengthscale
            
                cov_ij = (d_i1 + d_j1 - d_ij) / 2.0
                Sigma_w[i, j] = cov_ij
                Sigma_w[j, i] = cov_ij 
        self.Sigma_u = Sigma_w      
        print("  ✔ Macierz kowariancji dla znormalizowanych u gotowa.")
        
        self.mvn_u = MultivariateNormalCholesky(self.Sigma_u)
        self.counts_f = self.counts.astype(np.float64)
        
        print("  ✔ Prekomputacja stałych zakończona.")

    def znajdz_dobry_punkt_startowy(self, n_trials=20):
        return np.zeros(self.n-1)

    def przygotuj_predykcyjny(self, num_samples, burn_in, proposal_scale, seed=None):
        print("\n" + "="*70)
        print("▶ [MCMC] START SAMPLERA Z ADAPTACYJNYM MCMC (POPRAWIONA WERSJA)")
        print("="*70)

        import time
        timers = {
            'log_posterior': 0.0, 'proposal_generation': 0.0,
            'transform_w_to_x': 0.0, 'adaptation': 0.0, 'other': 0.0
        }
        total_start_time = time.time()

        if seed is not None:
            np.random.seed(seed)

        w_current = self.znajdz_dobry_punkt_startowy()
        logpost_current = log_posterior_fast(w_current, self.counts_f, self.mvn_u.L, 
                                           self.mvn_u.log_norm_const)

        adaptation_period = min(1000, burn_in // 2)
        adaptation_updates = 0
    
        if self.Sigma_u is not None:
            cov_prop = (proposal_scale**2) * self.Sigma_u
            try:
                L_prop = np.linalg.cholesky(cov_prop)
                use_cholesky = True
            except np.linalg.LinAlgError:
                print("  ⚠️ Macierz kowariancji nie jest dodatnio określona, używam diagonalnej")
                cov_prop = (proposal_scale**2) * np.eye(self.m)
                L_prop = np.sqrt(cov_prop)
                use_cholesky = True
        else:
            cov_prop = (proposal_scale**2) * np.eye(self.m)
            L_prop = np.sqrt(cov_prop)
            use_cholesky = True

        samples = []
        total_iterations = num_samples + burn_in
        accepted = 0
        accepted_burnin = 0
        acceptance_history = []
    
        print(f"📊 PARAMETRY MCMC:")
        print(f"   - Całkowita liczba iteracji: {total_iterations}")
        print(f"   - Burn-in: {burn_in}")
        print(f"   - Próbki posteriora: {num_samples}")
        print(f"   - Adaptacja przez: {adaptation_period} iteracji")
        print(f"   - Początkowy proposal scale: {proposal_scale}")
        print(f"   - Start LP: {logpost_current:.1f}")
        print("-" * 50)

        print("🔄 ROZPOCZĘCIE ITERACJI MCMC...")
    
        for i in range(total_iterations):
            iter_start_time = time.time()
            prop_start = time.time()
            
            z = np.random.normal(0, 1, self.m)
            w_prop = w_current + L_prop @ z
            timers['proposal_generation'] += time.time() - prop_start

            lp_start = time.time()
            try:
                logpost_prop = log_posterior_fast(w_prop, self.counts_f, self.mvn_u.L, 
                                                self.mvn_u.log_norm_const)
            except (ValueError, RuntimeError):
                logpost_prop = -np.inf
            timers['log_posterior'] += time.time() - lp_start

            accept = False
            if np.isfinite(logpost_prop):
                log_alpha = logpost_prop - logpost_current
                if log_alpha >= 0 or np.log(np.random.uniform()) < log_alpha:
                    accept = True
            
            if accept:
                w_current = w_prop
                logpost_current = logpost_prop
                if i >= burn_in: accepted += 1
                else: accepted_burnin += 1
        
            if i >= burn_in:
                samples.append(w_current.copy())
            
            acceptance_history.append(1 if accept else 0)
            if len(acceptance_history) > 100:
                acceptance_history.pop(0)

            adaptation_start = time.time()
            if i < adaptation_period and i >= 100 and i % 50 == 0:
                current_acc_rate = np.mean(acceptance_history)
                old_scale = proposal_scale
                
                if current_acc_rate < 0.15: proposal_scale *= 0.8
                elif current_acc_rate > 0.35: proposal_scale *= 1.2
                
                if proposal_scale != old_scale:
                    adaptation_updates += 1
                    cov_prop = (proposal_scale**2) * self.Sigma_u
                    try:
                        L_prop = np.linalg.cholesky(cov_prop)
                    except np.linalg.LinAlgError:
                        cov_prop = (proposal_scale**2) * np.eye(self.m)
                        L_prop = np.sqrt(cov_prop)
                    
                    if i % 200 == 0:
                        print(f"   🔄 ADAPTACJA: scale {old_scale:.4f} → {proposal_scale:.4f} (acc: {current_acc_rate:.3f})")
            timers['adaptation'] += time.time() - adaptation_start
            
            if i % 500 == 0 or i == total_iterations - 1:
                status = "BURN-IN" if i < burn_in else "SAMPLING"
                progress = (i + 1) / total_iterations * 100
                current_acc_display = (accepted_burnin / max(1, i + 1)) if i < burn_in else (accepted / max(1, i - burn_in + 1))
                print(f"   🔸 Iter {i:5d}/{total_iterations} [{status:8s}] | "
                      f"LP={logpost_current:8.1f} | AccRate={current_acc_display:.3f}")
        
        self.samples_w = np.array(samples)
        if len(self.samples_w) > 0:
            self.samples_x = np.array([x_from_w(w) for w in self.samples_w])
        else:
            self.samples_x = np.array([])
        
        final_acc_rate = accepted / max(1, num_samples)
        print("\n" + "="*70)
        print("✅ MCMC ZAKOŃCZONE - PODSUMOWANIE")
        print(f"   ✔ Akceptacje (sampling): {accepted}/{num_samples} ({final_acc_rate:.4f})")
        print(f"   ✔ Końcowy proposal scale: {proposal_scale:.4f}")
        print("="*70 + "\n")

    def posterior_mean(self):
        if self.samples_x is None: raise ValueError("Brak próbek posteriora.")
        return np.mean(self.samples_x, axis=0)
    
    def posterior_quantiles(self, q=(0.025, 0.975)):
        if self.samples_x is None: raise ValueError("Brak próbek posteriora.")
        return np.quantile(self.samples_x, q, axis=0)

class BayesianFieldModelAdaptiveSearchBinary(BayesianFieldModel):
    """
    Algorytm poszukiwania lengthscale:
    1. Zaczyna od start_ls, idzie do przodu co step_size aż znajdzie optimum (następny krok gorszy)
    2. Wykonuje k_steps kroków, w każdym kroku:
       - Dzieli aktualny step_size na 2
       - Testuje punkty: current_best - step_size, current_best, current_best + step_size
       - Wybiera najlepszy
    """
    def __init__(self, space_points, metric_func, observed_indices,
                 variance, distance_unit='km',
                 start_ls=3000, step_size=2000, k_steps=3,
                 num_samples_search=100, burn_in_search=50, 
                 proposal_scale_search=0.05):
        
        super().__init__(space_points, metric_func, observed_indices, 1, variance, distance_unit)
        self.start_ls = start_ls
        self.step_size = step_size  # Początkowy krok przeszukiwania
        self.k_steps = k_steps      # Liczba kroków bisekcji po znalezieniu optimum
        self.num_samples_search = num_samples_search
        self.burn_in_search = burn_in_search
        self.proposal_scale_search = proposal_scale_search
        print(f"\n▶ [ADAPTIVE-BINARY-SEARCH] Inicjalizacja:")
        print(f"   - Start lengthscale: {start_ls}")
        print(f"   - Step size: {step_size}")
        print(f"   - Kroki bisekcji (k): {k_steps}")

    def _evaluate_lengthscale(self, ls):
        """Szybka ocena lengthscale"""
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
            
            # Oblicz średni log-posterior z próbek
            if temp_model.samples_w is None or len(temp_model.samples_w) == 0:
                return -np.inf
                
            # Używamy tylko części próbek dla szybkości
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
        
        # --- FAZA 1: Szukanie zgrubnego optimum co step_size ---
        print(f"\n📌 FAZA 1: Szukanie optimum co {self.step_size}")
        print("-" * 50)
        
        current_ls = self.start_ls
        current_score = self._evaluate_lengthscale(current_ls)
        
        print(f"  Start: ls={current_ls:.0f}, score={current_score:.2f}")
        
        # Idziemy do przodu aż znajdziemy optimum
        iteration = 1
        i=0
        while i<20:
            i+=1

            next_ls = current_ls + self.step_size
            next_score = self._evaluate_lengthscale(next_ls)
            
            print(f"  Krok {iteration}: ls={next_ls:.0f}, score={next_score:.2f}")
            
            if next_score <= current_score:
                # Znaleźliśmy optimum - następny krok gorszy
                print(f"  ⬆️  Optimum znalezione: ls={current_ls:.0f} (następny krok gorszy)")
                break
            else:
                # Idziemy dalej
                current_ls = next_ls
                current_score = next_score
                iteration += 1
        
        # Zapisujemy znalezione optimum
        best_ls = current_ls
        best_score = current_score
        current_step = self.step_size
        
        print(f"\n✅ ZGRUBNE OPTIMUM: ls={best_ls:.0f}, score={best_score:.2f}")
        print(f"   Aktualny step: {current_step:.0f}")
        
        # --- FAZA 2: k kroków bisekcji ---
        print(f"\n📌 FAZA 2: {self.k_steps} kroków bisekcji")
        print("-" * 50)
        
        for k in range(1, self.k_steps + 1):
            print(f"\n  🔄 KROK BISEKCJI {k}/{self.k_steps}:")
            
            # Dzielimy krok na 2
            current_step = current_step / 2.0
            print(f"    Nowy step: {current_step:.0f} (połowa poprzedniego)")
            
            # Testujemy 3 punkty: best - step, best, best + step
            test_points = [
                best_ls - current_step,
                best_ls,
                best_ls + current_step
            ]
            
            # Upewnij się, że nie mamy ujemnych wartości
            test_points = [max(100, p) for p in test_points]  # min 100 km
            
            print(f"    Testowane punkty: {[f'{p:.0f}' for p in test_points]}")
            
            # Oceniamy wszystkie punkty
            scores = {}
            for ls in test_points:
                score = self._evaluate_lengthscale(ls)
                scores[ls] = score
                print(f"      ls={ls:.0f}: score={score:.2f}")
            
            # Znajdź najlepszy
            new_best_ls = max(scores, key=scores.get)
            new_best_score = scores[new_best_ls]
            
            # Sprawdź czy znaleźliśmy lepszy
            if new_best_score > best_score:
                print(f"    ✅ Znaleziono lepszy: {new_best_ls:.0f} "
                      f"(poprawa: {new_best_score - best_score:.2f})")
                best_ls = new_best_ls
                best_score = new_best_score
            else:
                print(f"    ℹ️  Najlepszy pozostaje: {best_ls:.0f}")
        
        # Ustaw finalny najlepszy lengthscale
        self.lengthscale = best_ls
        print(f"\n🎯 KONIEC OPTYMALIZACJI")
        print(f"   Finalny lengthscale: {self.lengthscale:.0f}")
        print(f"   Finalny score: {best_score:.2f}")
        print(f"   Finalny step: {current_step:.1f}")
        print("=" * 60)
        
        print("\n▶ [APRIORI] Finalne przygotowanie z najlepszym 'lengthscale'...")
        super().przygotuj_apriori()


class BayesianFieldModelCVGridSearch(BayesianFieldModel):
    def __init__(self, space_points, metric_func, observed_indices,
                 variance, distance_unit='km', lengthscale_grid=None, 
                 cv_k=5, cv_mcmc_samples=1000, cv_mcmc_burn=500):
        
        super().__init__(space_points, metric_func, observed_indices, 1, variance, distance_unit)
        self.lengthscale_grid = lengthscale_grid or [500, 1500, 2500, 4000, 6000]
        self.cv_k = cv_k
        self.cv_mcmc_samples = cv_mcmc_samples
        self.cv_mcmc_burn = cv_mcmc_burn
        self._score_cache = {}
        print(f"\n▶ [CV-GRID-SEARCH MODEL] Inicjalizacja z K={self.cv_k} i siatką: {self.lengthscale_grid}")

    def _evaluate_lengthscale_cv(self, ls):
        """Ocenia dany 'lengthscale' używając K-krotnej walidacji krzyżowej."""
        if ls in self._score_cache:
            print(f"  - Pobieranie z cache dla lengthscale={ls}")
            return self._score_cache[ls]

        print(f"--- Ewaluacja lengthscale = {ls} (z K={self.cv_k} walidacją) ---")
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
                
                # Walidacja na zbiorze walidacyjnym (Log-Likelihood)
                score = np.sum(np.log(pred_probs[val_indices] + 1e-9))
                fold_scores.append(score)
            except Exception as e:
                print(f"    Fold {k+1}/{self.cv_k} BŁĄD: {e}")
                fold_scores.append(-np.inf)

        avg_score = np.mean(fold_scores) if fold_scores else -np.inf
        print(f"  > Średni wynik dla lengthscale={ls}: {avg_score:.2f}\n")
        self._score_cache[ls] = avg_score
        return avg_score

    def przygotuj_apriori(self):
        print("▶ [CV GRID SEARCH] Rozpoczynanie poszukiwania 'lengthscale'...")
        best_ls = -1
        best_avg_score = -np.inf
        
        for ls in self.lengthscale_grid:
            avg_score = self._evaluate_lengthscale_cv(ls)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_ls = ls

        if best_ls == -1:
            raise ValueError("Grid search z walidacją krzyżową nie znalazł poprawnego 'lengthscale'.")
        
        print(f"✅ Najlepszy 'lengthscale' (CV): {best_ls} (wynik: {best_avg_score:.2f})")
        self.lengthscale = best_ls
        
        print("\n▶ [APRIORI] Finalne przygotowanie z najlepszym 'lengthscale'...")
        super().przygotuj_apriori()

#endregion
#  MODEL DIRICHLETA
class DirichletModel:
    """Prosty model Dirichleta jako baseline"""
    
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
        """Średnia posterior Dirichleta"""
        return self.alpha / self.alpha.sum()
    
    def sample_posterior(self, n_samples=1000):
        """Próbkuje z rozkładu posterior Dirichleta"""
        return dirichlet.rvs(self.alpha, size=n_samples)
    
    def posterior_quantiles(self, q=(0.025, 0.975)):
        """Kwantyle posterior (przybliżone)"""
        samples = self.sample_posterior(1000)
        return np.quantile(samples, q, axis=0)

#  BAYESIAN GAUSSIAN PROCESS MODEL
class BayesianGaussianProcess:
    """Bayesowski Gaussian Process do porównania - POPRAWIONA I ZOPTYMALIZOWANA WERSJA"""
    
    def __init__(self, space_points, observed_indices, 
                 lengthscale_prior=(1000, 500), variance_prior=(2, 1)):
        self.space_points = np.array(space_points)
        self.observed_indices = np.array(observed_indices)
        self.n_points = len(space_points)
        
        # ✅ NAPRAWIONY BŁĄD: inicjalizacja counts
        self.counts = np.bincount(observed_indices, minlength=self.n_points)
        self.total_obs = self.counts.sum()
        
        # Hiperparametry priorów
        self.ls_mu, self.ls_sigma = lengthscale_prior
        self.var_alpha, self.var_beta = variance_prior
        
        # ✅ OPTYMALIZACJA: prekomputacja macierzy odległości
        print("▶ [OPTIMIZED BAYESIAN GP] Prekomputacja macierzy odległości...")
        self.distance_matrix = self._precompute_distance_matrix()
        self.y = self.counts / self.total_obs  # Prekomputowane y
        
        print(f"  - Punkty: {self.n_points}, Obserwacje: {self.total_obs}")
        print(f"  - Counts: min={self.counts.min()}, max={self.counts.max()}")
    
    def _precompute_distance_matrix(self):
        """Prekomputuj macierz odległości RAZ na początku"""
        dist_matrix = np.zeros((self.n_points, self.n_points))
        for i in range(self.n_points):
            for j in range(i, self.n_points):
                dist = haversine(self.space_points[i], self.space_points[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix
    
    def rbf_kernel_fast(self, lengthscale, variance, jitter=1e-6):
        """ULTRA-SZYBKI kernel z prekomputacją"""
        # Użyj prekomputowanej macierzy odległości zamiast liczyć za każdym razem
        K = variance * np.exp(-0.5 * (self.distance_matrix / lengthscale)**2)
        K += np.eye(self.n_points) * jitter
        return K
    
    def log_prior(self, lengthscale, variance):
        """Logarytm priora dla parametrów"""
        # LogNormal dla lengthscale
        if lengthscale <= 0 or variance <= 0:
            return -np.inf
            
        lp_ls = -0.5 * ((np.log(lengthscale) - np.log(self.ls_mu)) / self.ls_sigma)**2 - np.log(lengthscale)
        
        # Inverse Gamma dla variance
        lp_var = -(self.var_alpha + 1) * np.log(variance) - self.var_beta / variance
        
        return lp_ls + lp_var
    
    def log_likelihood_fast(self, lengthscale, variance):
        """SZYBKA wiarygodność z prekomputacją"""
        # UŻYJ PREKOMPUTOWANEJ MACIERZY - NIE LICZ ODLEGŁOŚCI OD NOWA!
        K = self.rbf_kernel_fast(lengthscale, variance)
        
        try:
            L = la.cholesky(K, lower=True)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            # UŻYJ PREKOMPUTOWANEGO y
            alpha = la.solve_triangular(L, self.y, lower=True)
            alpha = la.solve_triangular(L.T, alpha, lower=False)
            
            log_like = -0.5 * self.y.dot(alpha) - 0.5 * log_det - 0.5 * self.n_points * np.log(2 * np.pi)
            return log_like
        except la.LinAlgError:
            return -np.inf
    
    def log_posterior_fast(self, lengthscale, variance):
        """SZYBKI posterior z prekomputacją"""
        lp = self.log_prior(lengthscale, variance)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood_fast(lengthscale, variance)
        if not np.isfinite(ll):
            return -np.inf
            
        return lp + ll
    
    def sample_posterior(self, n_samples=5000, burn_in=2000, step_size=0.1):
        """ZOPTYMALIZOWANY MCMC z prekomputacją"""
        print(f"\n▶ [OPTIMIZED GP MCMC] Rozpoczynanie szybkiego MCMC...")
        print(f"   - Próbki: {n_samples}, Burn-in: {burn_in}")
        print(f"   - Step size: {step_size}")
        
        current_ls = self.ls_mu
        current_var = 1.0
        current_log_post = self.log_posterior_fast(current_ls, current_var)
        
        samples_ls = []
        samples_var = []
        accepted = 0
        
        total_iter = n_samples + burn_in
        
        print(f"📊 PARAMETRY MCMC:")
        print(f"   - Całkowite iteracje: {total_iter}")
        print(f"   - Start LP: {current_log_post:.1f}")
        print(f"   - Start lengthscale: {current_ls:.1f}")
        print(f"   - Start variance: {current_var:.3f}")
        print("-" * 50)
        
        print("🔄 ROZPOCZĘCIE ITERACJI MCMC...")
        
        for i in range(total_iter):
            # Propozycja nowych parametrów
            proposed_ls = current_ls * np.exp(np.random.normal(0, step_size))
            proposed_var = current_var * np.exp(np.random.normal(0, step_size))
            
            # ✅ TERAZ TO JEST SZYBKIE - dzięki prekomputacji!
            proposed_log_post = self.log_posterior_fast(proposed_ls, proposed_var)
            
            # Acceptance ratio
            log_alpha = proposed_log_post - current_log_post
            log_alpha += np.log(proposed_ls) - np.log(current_ls)  # Jacobian
            log_alpha += np.log(proposed_var) - np.log(current_var)  # Jacobian
            
            accept = False
            if np.log(np.random.rand()) < log_alpha:
                current_ls = proposed_ls
                current_var = proposed_var
                current_log_post = proposed_log_post
                if i >= burn_in:
                    accepted += 1
                accept = True
            
            # Zapis próbek po burn-in
            if i >= burn_in:
                samples_ls.append(current_ls)
                samples_var.append(current_var)
            
            # Oblicz aktualny acceptance rate
            current_acc_rate = accepted / max(1, i - burn_in) if i > burn_in else accepted / max(1, i)
            
            # Status co 200 iteracji
            if i % 200 == 0 or i == total_iter - 1 or (i < 100 and i % 50 == 0):
                status = "BURN-IN" if i < burn_in else "SAMPLING"
                progress = (i + 1) / total_iter * 100
                
                print(f"   🔸 Iter {i:5d}/{total_iter} [{status:8s}] | "
                      f"LP={current_log_post:8.1f} | "
                      f"ls={current_ls:8.1f} | "
                      f"var={current_var:6.3f} | "
                      f"AccRate={current_acc_rate:6.3f} | "
                      f"Progress: {progress:5.1f}%")
            
            # Status przy rozpoczęciu fazy sampling
            if i == burn_in:
                print("-" * 50)
                print("🎯 ROZPOCZĘCIE FAZY SAMPLING (zapisywanie próbek)")
                print(f"   - Dotychczasowe akceptacje: {accepted}")
                print("-" * 50)
        
        self.samples_ls = np.array(samples_ls)
        self.samples_var = np.array(samples_var)
        
        acc_rate = accepted / n_samples
        
        print("\n" + "="*70)
        print("✅ OPTIMIZED GP MCMC ZAKOŃCZONE - PODSUMOWANIE")
        print("="*70)
        print(f"📊 STATYSTYKI:")
        print(f"   ✔ Próbki: {len(samples_ls)}")
        print(f"   ✔ Akceptacje: {accepted}")
        print(f"   ✔ Acceptance rate: {acc_rate:.3f}")
        
        print(f"\n📈 PARAMETRY POSTERIOR:")
        print(f"   - Lengthscale: {np.mean(samples_ls):.1f} ± {np.std(samples_ls):.1f}")
        print(f"   - Variance:    {np.mean(samples_var):.3f} ± {np.std(samples_var):.3f}")
        
        print("="*70 + "\n")
        
        return self.samples_ls, self.samples_var
    
    def posterior_predictive(self):
        """Rozkład predykcyjny posteriora"""
        if not hasattr(self, 'samples_ls'):
            raise ValueError("Najpierw uruchom sample_posterior()")
        
        print(f"▶ [OPTIMIZED GP] Obliczanie rozkładu predykcyjnego...")
        
        # Średnie parametry z posteriora
        mean_ls = np.mean(self.samples_ls)
        mean_var = np.mean(self.samples_var)
        
        print(f"  - Używam lengthscale: {mean_ls:.1f}")
        print(f"  - Używam variance: {mean_var:.3f}")
        
        # ✅ UŻYJ PREKOMPUTOWANEJ MACIERZY - SZYBKO!
        K = self.rbf_kernel_fast(mean_ls, mean_var)
        
        # Gaussian Process regression
        L = la.cholesky(K, lower=True)
        alpha = la.solve_triangular(L, self.y, lower=True)
        alpha = la.solve_triangular(L.T, alpha, lower=False)
        
        # Predykcja
        predictive_mean = K @ alpha
        
        # Normalizacja do rozkładu prawdopodobieństwa
        predictive_probs = predictive_mean / predictive_mean.sum()
        
        print(f"  - Suma predykcji: {predictive_mean.sum():.6f}")
        print(f"  - Min predykcja: {np.min(predictive_probs):.2e}")
        print(f"  - Max predykcja: {np.max(predictive_probs):.2e}")
        
        return predictive_probs

    # Zachowaj oryginalną metodę dla kompatybilności (jeśli jest używana gdzieś indziej)
    def log_likelihood(self, lengthscale, variance, jitter=1e-6):
        """Oryginalna wiarygodność (dla kompatybilności)"""
        return self.log_likelihood_fast(lengthscale, variance)
    
    def log_posterior(self, lengthscale, variance):
        """Oryginalny posterior (dla kompatybilności)"""
        return self.log_posterior_fast(lengthscale, variance)
#  BAYESIAN SPATIAL SMOOTHING MODEL
class BayesianSpatialSmoothing:
    """Prosty Bayesowski model przestrzennego wygładzania"""
    
    def __init__(self, space_points, observed_indices, smoothing_factor=0.1):
        self.space_points = np.array(space_points)
        self.observed_indices = np.array(observed_indices)
        self.n_points = len(space_points)
        self.smoothing_factor = smoothing_factor
        
        # Zlicz obserwacje
        self.counts = np.bincount(observed_indices, minlength=self.n_points)
        self.total_obs = self.counts.sum()
        
        print(f"▶ [SPATIAL SMOOTHING] Inicjalizacja:")
        print(f"  - Punkty: {self.n_points}, Obserwacje: {self.total_obs}")
        print(f"  - Smoothing factor: {smoothing_factor}")
    
    def compute_spatial_weights(self):
        """Oblicza wagi przestrzenne na podstawie odległości"""
        print("▶ [SPATIAL SMOOTHING] Obliczanie wag przestrzennych...")
        weights = np.zeros((self.n_points, self.n_points))
        
        for i in range(self.n_points):
            for j in range(self.n_points):
                if i == j:
                    weights[i, j] = 1.0
                else:
                    dist = haversine(self.space_points[i], self.space_points[j])
                    # Waga maleje wykładniczo z odległością
                    weights[i, j] = np.exp(-dist / (self.smoothing_factor * 1000))
        
        # Normalizuj wagi wierszowo
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = weights / row_sums
        
        print(f"  - Min waga: {np.min(weights):.4f}")
        print(f"  - Max waga: {np.max(weights):.4f}")
        print(f"  - Średnia waga: {np.mean(weights):.4f}")
        
        return weights
    
    def posterior_mean(self):
        """Średnia posterior z przestrzennym wygładzaniem"""
        # Empiryczne prawdopodobieństwa
        empirical_probs = self.counts / self.total_obs
        
        print(f"  - Empiryczne probs - min: {np.min(empirical_probs):.2e}, max: {np.max(empirical_probs):.2e}")
        
        # Macierz wag przestrzennych
        spatial_weights = self.compute_spatial_weights()
        
        # Wygładzone prawdopodobieństwa
        smoothed_probs = spatial_weights @ empirical_probs
        
        # Normalizacja
        smoothed_probs = smoothed_probs / smoothed_probs.sum()
        
        print(f"  - Wygładzone probs - min: {np.min(smoothed_probs):.2e}, max: {np.max(smoothed_probs):.2e}")
        print(f"  - Suma wygładzonych: {smoothed_probs.sum():.6f}")
        
        return smoothed_probs
#endregion 
#endregion
#RESZTA KODU
#region
# IMPORTY
import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.linalg import cholesky, solve_triangular
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io import shapereader
from statsmodels.tsa.stattools import acf
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import dirichlet
import datetime
import json
import scipy.linalg as la
# METRYKA
def haversine(p1, p2, return_unit='km'):
    lon1, lat1 = map(math.radians, p1)
    lon2, lat2 = map(math.radians, p2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    if return_unit == 'km':
        return 6371.0 * c
    elif return_unit == 'rad':
        return c
    else:
        raise ValueError("return_unit must be 'km' or 'rad'")   
# WCZYTYWANIE I ZAPISYWANIE DANYCH 
# region 
def wczytaj_dane(csv_path):
    print("▶ [DATA] Wczytywanie danych z CSV...")
    df = pd.read_csv(csv_path)
    print(f"  - Wczytano {len(df)} rekordów.")
    geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")
    gdf["id"] = range(len(gdf))
    print("  ✔ Geopandas GeoDataFrame gotowe.")
    return gdf
def wczytaj_pacyfik():
    print("▶ [MAPA] Wczytywanie kształtu Pacyfiku...")
    shpfilename = shapereader.natural_earth(
        resolution="110m",
        category="physical",
        name="geography_marine_polys"
    )
    oceans = gpd.read_file(shpfilename)
    pacific = oceans[oceans["name"].str.contains("Pacific", case=False, na=False)]
    print(f"  ✔ Znaleziono {len(pacific)} poligonów Pacyfiku.")
    return pacific, pacific.unary_union
def wczytaj_lad():
    """Wczytuje kształt lądów do tła mapy"""
    print("▶ [MAPA] Wczytywanie kształtu lądów...")
    land_geoms = list(shapereader.Reader(shapereader.natural_earth(
        resolution='110m',
        category='physical',
        name='land')).geometries())
    land = gpd.GeoSeries(land_geoms, crs="EPSG:4326")
    print("  ✔ Kształt lądów gotowy.")
    return land
def filtruj_pacyfik_i_brzeg(gdf_points, cutoff_km, iqr_multiplier=1.5):
    print("\n▶ [FILTER] Filtrowanie do obszaru Pacyfiku...")
    before = len(gdf_points)
    pacific, pacific_union = wczytaj_pacyfik()

    gdf_f = gdf_points[gdf_points.within(pacific_union)].copy()
    gdf_f.reset_index(drop=True, inplace=True)
    print(f"  - Przed: {before}, Po filtrze Pacyfiku: {len(gdf_f)}")

    print("\n▶ [FILTER] Usuwanie outlierów z Species Count...")
    species_counts = gdf_f["Species Count"].values
    Q1 = np.percentile(species_counts, 25)
    Q3 = np.percentile(species_counts, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    before_outliers = len(gdf_f)
    gdf_f = gdf_f[(gdf_f["Species Count"] >= lower_bound) & 
                  (gdf_f["Species Count"] <= upper_bound)]
    gdf_f.reset_index(drop=True, inplace=True)
    
    outliers_removed = before_outliers - len(gdf_f)
    print(f"  - Przed usuwaniem outlierów: {before_outliers}")
    print(f"  - Po usunięciu outlierów: {len(gdf_f)}")
    print(f"  - Usunięto {outliers_removed} outlierów")
    print(f"  - Granice outlierów: [{lower_bound:.1f}, {upper_bound:.1f}]")

    print("\n▶ [FILTER] Usuwanie punktów blisko brzegu...")
    land = wczytaj_lad()
    land_union = land.unary_union

    cutoff_deg = cutoff_km / 111.0
    buffer = land_union.buffer(cutoff_deg)

    before_coast = len(gdf_f)
    gdf_f = gdf_f[~gdf_f.geometry.within(buffer)]
    gdf_f.reset_index(drop=True, inplace=True)
    print(f"  - Pozostało: {len(gdf_f)} (usunięto {before_coast - len(gdf_f)} przy brzegu)")

    return gdf_f
def zmniejsz_siatke(gdf, co_ktory):
    print(f"\n▶ [REDUKCJA] Redukcja siatki co {co_ktory} punkt...")
    before = len(gdf)
    if co_ktory <= 1:
        print("  - Pomijam redukcję (co_ktory<=1)")
        return gdf.copy()
    reduced = gdf.iloc[::co_ktory].copy()
    reduced.reset_index(drop=True, inplace=True)
    print(f"  - Przed: {before}, Po redukcji: {len(reduced)}")
    return reduced
def losuj_obserwacje(gdf, n_points):
    print(f"\n▶ [OBS] Losowanie obserwacji ({n_points}) wg Species Count...")
    species_counts = gdf["Species Count"].values
    probs = species_counts / species_counts.sum()
    idx = np.random.choice(len(gdf), size=n_points, p=probs, replace=True)
    print(f"  ✔ Zaliczone: wylosowano {len(idx)} indeksów.")
    return idx
def przygotuj_dane(csv_path, cutoff_km, co_ktory, n_observations):
    print("\n===================================")
    print("▶ [PIPELINE] Przygotowanie danych...")
    print("===================================")
    gdf = wczytaj_dane(csv_path)
    gdf = filtruj_pacyfik_i_brzeg(gdf, cutoff_km)
    gdf = zmniejsz_siatke(gdf, co_ktory)

    true_counts = gdf["Species Count"].values
    true_probs = true_counts / true_counts.sum()

    obs_idx = losuj_obserwacje(gdf, n_observations)

    print("\n✅ DANE GOTOWE.")
    print(f"   - Punkty po filtrach: {len(gdf)}")
    print(f"   - Obserwacje: {len(obs_idx)}")
    print(f"   - Suma prawdopodobieństw: {true_probs.sum():.6f}\n")

    return gdf, obs_idx, true_probs

#endregion
#  WIZUALIZACJE MAP
def stworz_mape_porownawcza(gdf, true_probs, pred_probs, title_suffix=""):
    """Tworzy mapę porównawczą prawdziwego i predykowanego rozkładu"""
    print(f"▶ [MAP] Tworzenie mapy porównawczej {title_suffix}...")
    
    # Oblicz wspólny zakres dla skal kolorów
    vmin = min(np.min(true_probs), np.min(pred_probs))
    vmax = max(np.max(true_probs), np.max(pred_probs))
    
    # Stwórz custom colormap - niebieski dla niskich wartości, czerwony dla wysokich
    colors = ['#1E3F66', '#2E5984', '#4682B4', '#87CEEB', '#B0E0E6', 
              '#FFE4E1', '#FFB6C1', '#FF69B4', '#DC143C', '#8B0000']
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_blue_red", colors, N=256)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                  subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Mapa 1: Prawdziwy rozkład
    sc1 = ax1.scatter(gdf["Longitude"], gdf["Latitude"], 
                     c=true_probs, cmap=cmap, s=30, alpha=0.7,
                     vmin=vmin, vmax=vmax)
    ax1.coastlines()
    ax1.set_global()
    ax1.set_title(f'Prawdziwy rozkład bogactwa gatunkowego\n{title_suffix}', 
                 fontsize=14, fontweight='bold')
    
    # Dodaj colorbar dla pierwszej mapy
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    plt.colorbar(sc1, cax=cax1, label='Prawdopodobieństwo')
    
    # Mapa 2: Predykowany rozkład
    sc2 = ax2.scatter(gdf["Longitude"], gdf["Latitude"], 
                     c=pred_probs, cmap=cmap, s=30, alpha=0.7,
                     vmin=vmin, vmax=vmax)
    ax2.coastlines()
    ax2.set_global()
    ax2.set_title(f'Predykowany rozkład bogactwa gatunkowego\n{title_suffix}', 
                 fontsize=14, fontweight='bold')
    
    # Dodaj colorbar dla drugiej mapy
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    plt.colorbar(sc2, cax=cax2, label='Prawdopodobieństwo')
    
    plt.tight_layout()
    plt.savefig(f'mapa_porownawcza_{title_suffix.replace(" ", "_").lower()}.png', 
                dpi=150, bbox_inches='tight')
    plt.show(block=False)
    
    return fig

#  BAZA DANYCH WYNIKÓW
class ResultsDatabase:
    """Baza danych do przechowywania wyników wielu testów"""
    
    def __init__(self, db_path="wyniki_testow.csv"):
        self.db_path = db_path
        self.initialize_database()
    
    def initialize_database(self):
        """Inicjalizuje bazę danych jeśli nie istnieje"""
        if not os.path.exists(self.db_path):
            columns = [
                'test_id', 'timestamp', 'n_points', 'n_observations',
                'lengthscale', 'variance', 'mcmc_samples', 
                'mcmc_burn', 'mcmc_scale', 'mcmc_seed',
                'bayesian_mse', 'bayesian_mae', 'bayesian_rmse', 'bayesian_correlation', 'bayesian_covariance',
                'dirichlet_mse', 'dirichlet_mae', 'dirichlet_rmse', 'dirichlet_correlation', 'dirichlet_covariance',
                'gp_mse', 'gp_mae', 'gp_rmse', 'gp_correlation', 'gp_covariance',
                'spatial_mse', 'spatial_mae', 'spatial_rmse', 'spatial_correlation', 'spatial_covariance',
                'mse_diff', 'mae_diff', 'correlation_diff', 'better_model_mse', 'better_model_mae',
                'bayesian_better_count', 'dirichlet_better_count', 'gp_better_count', 'spatial_better_count', 'equal_count',
                'wilcoxon_pvalue', 'test_duration_seconds'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.db_path, index=False)
            print(f"▶ [DB] Utworzono nową bazę danych: {self.db_path}")
    
    def save_test_results(self, test_params, metrics_bayesian, metrics_dirichlet, 
                         metrics_gp, metrics_spatial, comparison_stats, duration):
        """Zapisuje wyniki pojedynczego testu do bazy danych"""
        # Wczytaj istniejącą bazę
        df = pd.read_csv(self.db_path)
        
        # Generuj unikalny ID testu
        test_id = f"test_{len(df) + 1:04d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Znajdź najlepszy model wg MSE
        models_mse = {
            'Bayesian': metrics_bayesian.get('MSE', np.nan),
            'Dirichlet': metrics_dirichlet.get('MSE', np.nan),
            'GP': metrics_gp.get('MSE', np.nan),
            'Spatial': metrics_spatial.get('MSE', np.nan)
        }
        # Usuń modele z nan, aby znaleźć minimum
        valid_models_mse = {k: v for k, v in models_mse.items() if not np.isnan(v)}
        best_model_mse = min(valid_models_mse, key=valid_models_mse.get) if valid_models_mse else 'N/A'
        
        # Przygotuj nowy wiersz
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
            
            # Metryki Bayesian Field
            'bayesian_mse': float(metrics_bayesian.get('MSE', np.nan)),
            'bayesian_mae': float(metrics_bayesian.get('MAE', np.nan)),
            'bayesian_rmse': float(metrics_bayesian.get('RMSE', np.nan)),
            'bayesian_correlation': float(metrics_bayesian.get('Correlation', np.nan)),
            'bayesian_covariance': float(metrics_bayesian.get('Covariance', np.nan)),
            
            # Metryki Dirichlet
            'dirichlet_mse': float(metrics_dirichlet.get('MSE', np.nan)),
            'dirichlet_mae': float(metrics_dirichlet.get('MAE', np.nan)),
            'dirichlet_rmse': float(metrics_dirichlet.get('RMSE', np.nan)),
            'dirichlet_correlation': float(metrics_dirichlet.get('Correlation', np.nan)),
            'dirichlet_covariance': float(metrics_dirichlet.get('Covariance', np.nan)),
            
            # Metryki Gaussian Process
            'gp_mse': float(metrics_gp.get('MSE', np.nan)),
            'gp_mae': float(metrics_gp.get('MAE', np.nan)),
            'gp_rmse': float(metrics_gp.get('RMSE', np.nan)),
            'gp_correlation': float(metrics_gp.get('Correlation', np.nan)),
            'gp_covariance': float(metrics_gp.get('Covariance', np.nan)),
            
            # Metryki Spatial Smoothing
            'spatial_mse': float(metrics_spatial.get('MSE', np.nan)),
            'spatial_mae': float(metrics_spatial.get('MAE', np.nan)),
            'spatial_rmse': float(metrics_spatial.get('RMSE', np.nan)),
            'spatial_correlation': float(metrics_spatial.get('Correlation', np.nan)),
            'spatial_covariance': float(metrics_spatial.get('Covariance', np.nan)),
            
            # Różnice i najlepsze modele
            'mse_diff': float(metrics_bayesian.get('MSE', np.nan) - metrics_dirichlet.get('MSE', np.nan)),
            'mae_diff': float(metrics_bayesian.get('MAE', np.nan) - metrics_dirichlet.get('MAE', np.nan)),
            'correlation_diff': float(metrics_bayesian.get('Correlation', np.nan) - metrics_dirichlet.get('Correlation', np.nan)),
            'better_model_mse': best_model_mse,
            'better_model_mae': 'Bayesian' if metrics_bayesian.get('MAE', np.inf) < metrics_dirichlet.get('MAE', np.inf) else 'Dirichlet',
            
            # Statystyki porównania
            'bayesian_better_count': int(comparison_stats.get('bayesian_better_count', 0)),
            'dirichlet_better_count': int(comparison_stats.get('dirichlet_better_count', 0)),
            'gp_better_count': int(comparison_stats.get('gp_better_count', 0)),
            'spatial_better_count': int(comparison_stats.get('spatial_better_count', 0)),
            'equal_count': int(comparison_stats.get('equal_count', 0)),
            'wilcoxon_pvalue': float(comparison_stats.get('wilcoxon_pvalue', np.nan)),
            'test_duration_seconds': float(duration)
        }
        
        # Dodaj nowy wiersz
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Zapisz zaktualizowaną bazę
        df.to_csv(self.db_path, index=False)
        print(f"  ✔ Wyniki zapisane do bazy danych: {test_id}")
        
        return test_id
    
    def get_summary_stats(self):
        """Zwraca statystyki podsumowujące wszystkie testy"""
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
        """Wyświetla podsumowanie wszystkich testów"""
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
#FUNKCJE POMOCNICZE DO TESTOWANIA
#region 
def oblicz_metryki(true_probs, pred_probs, model_name, verbose=True):
    """Oblicza podstawowe metryki jakości predykcji"""
    if verbose:
        print(f"\n📈 METRYKI DLA {model_name}:")
    n=len(pred_probs)
    # Podstawowe metryki
    mse = float(np.mean(((pred_probs - true_probs)*n)**2))
    mae = float(np.mean(np.abs(pred_probs - true_probs)*n))
    rmse = float(np.sqrt(mse))
    corr = float(np.corrcoef(pred_probs, true_probs)[0, 1])
    covariance = float(np.cov(pred_probs, true_probs)[0, 1])
    
    if verbose:
        print(f"   - MSE:               {mse:.6f}")
        print(f"   - MAE:               {mae:.6f}")
        print(f"   - RMSE:              {rmse:.6f}")
        print(f"   - Korelacja:         {corr:.4f}")
        print(f"   - Kowariancja:       {covariance:.6f}")
    
    return {
        'MSE': mse, 'MAE': mae, 'RMSE': rmse, 
        'Correlation': corr, 'Covariance': covariance
    }


def oblicz_statystyki_porownania_wszystkich(true_probs, bayesian_pred, dirichlet_pred, gp_pred, spatial_pred):
    """Oblicza statystyki porównania między wszystkimi modelami"""
    errors_bayesian = np.abs(bayesian_pred - true_probs)
    errors_dirichlet = np.abs(dirichlet_pred - true_probs)
    errors_gp = np.abs(gp_pred - true_probs)
    errors_spatial = np.abs(spatial_pred - true_probs)
    
    # Który model jest lepszy w ilu punktach
    bayesian_better_count = 0
    dirichlet_better_count = 0
    gp_better_count = 0
    spatial_better_count = 0
    equal_count = 0
    
    for i in range(len(true_probs)):
        errors = {
            'bayesian': errors_bayesian[i],
            'dirichlet': errors_dirichlet[i],
            'gp': errors_gp[i],
            'spatial': errors_spatial[i]
        }
        min_error = min(errors.values())
        
        # Zlicz które modele mają minimalny błąd
        best_models = [model for model, error in errors.items() if error == min_error]
        
        if len(best_models) == 1:
            if best_models[0] == 'bayesian':
                bayesian_better_count += 1
            elif best_models[0] == 'dirichlet':
                dirichlet_better_count += 1
            elif best_models[0] == 'gp':
                gp_better_count += 1
            elif best_models[0] == 'spatial':
                spatial_better_count += 1
        else:
            equal_count += 1
    
    # Test Wilcoxona między Bayesian a Dirichlet (dla zachowania kompatybilności)
    from scipy.stats import wilcoxon
    try:
        stat, wilcoxon_pvalue = wilcoxon(errors_bayesian, errors_dirichlet)
        wilcoxon_pvalue = float(wilcoxon_pvalue)
    except:
        wilcoxon_pvalue = 1.0
    
    return {
        'bayesian_better_count': bayesian_better_count,
        'dirichlet_better_count': dirichlet_better_count,
        'gp_better_count': gp_better_count,
        'spatial_better_count': spatial_better_count,
        'equal_count': equal_count,
        'wilcoxon_pvalue': wilcoxon_pvalue
    }
def pokaz_punkt_referencyjny(gdf, title="Punkt referencyjny (indeks 1)"):
    """Pokazuje na mapie który punkt jest punktem referencyjnym o indeksie 1"""
    print(f"▶ [MAP] Tworzenie mapy punktu referencyjnego...")
    
    # ✅ POPRAWIONE: Punkt referencyjny to pierwszy punkt w gdf
    ref_point = (gdf.iloc[0]["Longitude"], gdf.iloc[0]["Latitude"])
    ref_idx = 0
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Wszystkie punkty szare
    ax.scatter(gdf["Longitude"], gdf["Latitude"], 
               c='lightgray', s=20, alpha=0.5, label='Wszystkie punkty')
    
    # Punkt referencyjny czerwony
    ax.scatter(ref_point[0], ref_point[1], 
               c='red', s=100, marker='*', edgecolors='black', linewidth=2,
               label=f'Punkt referencyjny (indeks 1)\n{ref_point[0]:.2f}°, {ref_point[1]:.2f}°')
    
    ax.coastlines()
    ax.set_global()
    ax.legend(loc='upper left')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Dodaj adnotację z współrzędnymi
    ax.annotate(f'({ref_point[0]:.2f}°, {ref_point[1]:.2f}°)', 
                xy=ref_point, xytext=(10, 10),
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('punkt_referencyjny.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"  ✔ Punkt referencyjny: {ref_point}")
    print(f"  ✔ Indeks: {ref_idx}")
    
    return ref_point, ref_idx

    
#endregion
#  MENADŻER TESTÓW
class TestManager:
    """Menadżer do uruchamiania wielu testów"""
    
    def __init__(self, csv_path, db_path="wyniki_testow.csv"):
        self.csv_path = csv_path
        self.db = ResultsDatabase(db_path)
        self.results = []
        self.cached_data = None
    
    def prepare_data_once(self, cutoff_km, co_ktory, max_observations):
        """Przygotowuje dane RAZ i cache'uje"""
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
             models_to_test=None, save=False):
        """
        Główna funkcja testująca
        
        Parameters:
        -----------
        test_params : dict
            Parametry testu
        test_number : int
            Numer testu
        total_tests : int
            Łączna liczba testów
        models_to_test : list
            Lista krotek (nazwa_modelu, parametry_modelu) do przetestowania.
            Np. [('bayesian', {'lengthscale': 1000}), ('dirichlet', {})]
        save : bool
            Czy zapisać wyniki do bazy danych
        """
        print(f"\n{'='*60}")
        print(f"▶ TEST {test_number}/{total_tests}")
        print(f"{'='*60}")
        
        # Domyślne modele do testowania, jeśli nie podano
        # Te domyślne parametry są tylko dla wewnętrznego użycia, 
        # prawdziwa konfiguracja powinna być w bloku if __name__ == "__main__":
        # Pamiętaj, że te domyślne wartości są niezależne od base_params.
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
        
        # Wyświetl jakie modele będą testowane
        active_models = [f"{name}_{i}" for i, (name, params) in enumerate(models_to_test)]
        print(f"🎯 TESTOWANE MODELE: {', '.join(active_models)}")
        
        start_time = datetime.datetime.now()
        
        try:
            # Przygotuj dane
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
            
            # Aktualizuj liczbę punktów
            test_params['n_points'] = len(gdf)
            
            # Losuj obserwacje
            print(f"▶ [OBS] Losowanie {test_params['n_observations']} obserwacji...")
            obs_idx = losuj_obserwacje(gdf, test_params['n_observations'])
            
            # Słowniki na wyniki
            predictions = {}
            metrics = {}
            models = {}

            # Pętla po modelach do testowania
            for i, (model_name, model_params) in enumerate(models_to_test):
                unique_model_key = f"{model_name}_{i}"
                model_display_name = f"{model_name.capitalize()} ({i})"
                
                try:
                    if model_name == 'bayesian':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': obs_idx,
                            'lengthscale': model_params.get('lengthscale', 500),
                            'variance': model_params.get('variance', 1.0),
                            'distance_unit': model_params.get('distance_unit', 'km')
                        }
                        model = BayesianFieldModel(**constructor_params)
                        model.przygotuj_apriori()
                        model.przygotuj_predykcyjny(
                            num_samples=model_params.get('mcmc_samples', 5000),
                            burn_in=model_params.get('mcmc_burn', 3000),
                            proposal_scale=model_params.get('mcmc_scale', 0.05),
                            seed=model_params.get('mcmc_seed', 42) + test_number
                        )
                        pred = model.posterior_mean()

                    elif model_name == 'bayesian_adaptive_search_binary':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        constructor_params = {
                            'space_points': points, 'metric_func': haversine, 'observed_indices': obs_idx,
                            'variance': model_params.get('variance', 1.0),
                            'distance_unit': model_params.get('distance_unit', 'km'),
                            'start_ls': model_params.get('start_ls', 1000),
                            'step_size': model_params.get('step_size', 2000),
                            'k_steps': model_params.get('k_steps', 3),
                            'num_samples_search': model_params.get("num_samples_search", 100),
                            'burn_in_search': model_params.get("burn_in_search", 50),
                            'proposal_scale_search': model_params.get("proposal_scale_search", 0.05)
                        }
                        model = BayesianFieldModelAdaptiveSearchBinary(**constructor_params)
                        model.przygotuj_apriori() # This will run the binary adaptive search
                        model.przygotuj_predykcyjny(
                            num_samples=model_params.get('mcmc_samples', 5000),
                            burn_in=model_params.get('mcmc_burn', 3000),
                            proposal_scale=model_params.get('mcmc_scale', 0.05),
                            seed=model_params.get('mcmc_seed', 42) + test_number
                        )
                        pred = model.posterior_mean()
                    elif model_name == 'bayesian_cv_gridsearch':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
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
                        model.przygotuj_apriori() # Uruchamia CV grid search
                        model.przygotuj_predykcyjny(
                            num_samples=model_params.get('mcmc_samples', 5000),
                            burn_in=model_params.get('mcmc_burn', 3000),
                            proposal_scale=model_params.get('mcmc_scale', 0.05),
                            seed=model_params.get('mcmc_seed', 42) + test_number
                        )
                        pred = model.posterior_mean()
                    elif model_name == 'dirichlet':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        model = DirichletModel(obs_idx, len(gdf))
                        pred = model.posterior_mean()

                    elif model_name == 'gaussian':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        gp_constructor_params = {
                            'space_points': points, 'observed_indices': obs_idx,
                            'lengthscale_prior': model_params.get('lengthscale_prior', (1000, 500)),
                            'variance_prior': model_params.get('variance_prior', (2, 1))
                        }
                        model = BayesianGaussianProcess(**gp_constructor_params)
                        model.sample_posterior(
                            n_samples=model_params.get('n_samples', 1000),
                            burn_in=model_params.get('burn_in', 500),
                            step_size=model_params.get('step_size', 0.1)
                        )
                        pred = model.posterior_predictive()

                    elif model_name == 'spatial':
                        print(f"\n--- MODEL: {model_display_name.upper()} ---")
                        model = BayesianSpatialSmoothing(
                            points, obs_idx, smoothing_factor=model_params.get('smoothing_factor', 0.1)
                        )
                        pred = model.posterior_mean()
                        
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
                # Aby zapewnić kompatybilność z istniejącym schematem bazy danych,
                # który oczekuje parametrów modelu bezpośrednio w `test_params`,
                # będziemy scalać parametry pierwszego modelu Bayesa (jeśli istnieje)
                # z ogólnymi parametrami testu.
                # Jest to kompromis, aby uniknąć zmiany schematu bazy danych.
                
                # Initialize with base_params
                save_test_params = test_params.copy() 
                
                # Try to find Bayesian model's params to save
                bayesian_model_params_for_save = {}
                for mn, mp in models_to_test:
                    if mn == 'bayesian':
                        # Użyj wartości z modelu, jeśli istnieją, w przeciwnym razie domyślne
                        bayesian_model_params_for_save = mp
                        break
                
                # Domyślne wartości, jeśli model Bayesian nie został znaleziony
                # lub jeśli brakuje niektórych parametrów w model_params
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
                # Merge relevant bayesian params into save_test_params for DB compatibility
                save_test_params['lengthscale'] = bayesian_model_params_for_save.get('lengthscale', 0.0)
                save_test_params['variance'] = bayesian_model_params_for_save.get('variance', 0.0)
                save_test_params['mcmc_samples'] = bayesian_model_params_for_save.get('mcmc_samples', 0)
                save_test_params['mcmc_burn'] = bayesian_model_params_for_save.get('mcmc_burn', 0)
                save_test_params['mcmc_scale'] = bayesian_model_params_for_save.get('mcmc_scale', 0.0)
                save_test_params['mcmc_seed'] = bayesian_model_params_for_save.get('mcmc_seed', 0)
                
                test_id = self._save_to_database(save_test_params, metrics, 
                                               comparison_stats, models_to_test, duration)
            
            self._print_test_summary(metrics, comparison_stats, duration)
            
            if test_number == 1:
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
        """Oblicza statystyki porównania między modelami"""
        errors = {}
        for model_name, pred in predictions.items():
            errors[model_name] = np.abs(pred - true_probs)
        
        # Który model jest lepszy w ilu punktach
        better_counts = {model_name: 0 for model_name in predictions.keys()}
        equal_count = 0
        
        n_points = len(true_probs)
        for i in range(n_points):
            point_errors = {model: errors[model][i] for model in predictions.keys()}
            min_error = min(point_errors.values())
            
            # Zlicz które modele mają minimalny błąd
            best_models = [model for model, error in point_errors.items() 
                          if error == min_error]
            
            if len(best_models) == 1:
                better_counts[best_models[0]] += 1
            else:
                equal_count += 1
        
        # Test Wilcoxona między pierwszymi dwoma modelami (jeśli są)
        wilcoxon_pvalue = None
        model_names = list(predictions.keys())
        if len(model_names) >= 2:
            from scipy.stats import wilcoxon
            try:
                stat, pvalue = wilcoxon(errors[model_names[0]], errors[model_names[1]])
                wilcoxon_pvalue = float(pvalue)
            except:
                wilcoxon_pvalue = 1.0
        
        # Znajdź najlepszy model (najmniejsze MSE)
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
        """Zapisuje wyniki do bazy danych, tworząc osobny wiersz dla każdego modelu."""
        
        test_ids = []
        
        # Stwórz kopię listy, aby można było z niej usuwać elementy
        remaining_models = list(models_to_test)

        for model_key, model_metrics in metrics.items():
            model_type = model_key.rsplit('_', 1)[0]

            # Znajdź parametry dla bieżącego modelu
            current_model_params = {}
            model_found_index = -1
            for i, (name, params) in enumerate(remaining_models):
                if name == model_type:
                    current_model_params = params
                    model_found_index = i
                    break
            
            if model_found_index != -1:
                # Usuń znaleziony model, aby uniknąć ponownego dopasowania
                remaining_models.pop(model_found_index)

            # Przygotuj parametry do zapisu
            save_params = test_params.copy()
            if model_type == 'bayesian':
                save_params['lengthscale'] = current_model_params.get('lengthscale', 0)
                save_params['variance'] = current_model_params.get('variance', 0)
                save_params['mcmc_samples'] = current_model_params.get('mcmc_samples', 0)
                save_params['mcmc_burn'] = current_model_params.get('mcmc_burn', 0)
                save_params['mcmc_scale'] = current_model_params.get('mcmc_scale', 0)
                save_params['mcmc_seed'] = current_model_params.get('mcmc_seed', 0)
            
            # Przygotuj puste metryki dla wszystkich typów modeli
            metrics_bayesian = {}
            metrics_dirichlet = {}
            metrics_gp = {}
            metrics_spatial = {}

            # Wypełnij metryki dla odpowiedniego typu modelu
            if model_type == 'bayesian':
                metrics_bayesian = model_metrics
            elif model_type == 'dirichlet':
                metrics_dirichlet = model_metrics
            elif model_type == 'gaussian':
                metrics_gp = model_metrics
            elif model_type == 'spatial':
                metrics_spatial = model_metrics

            # Statystyki porównawcze są obliczane dla całego przebiegu testu,
            # więc zapisujemy je w każdym wierszu (będą zduplikowane).
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
        """Wyświetla podsumowanie testu, obsługując unikalne klucze modeli."""
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
            print(f"\n   NAJLEPSZY MODEL: {comparison_stats['best_model'].capitalize()} "
                  f"(MSE={comparison_stats.get('best_mse', 'N/A'):.6f})")
        
        if 'better_counts' in comparison_stats:
            print(f"\n   LICZBA PUNKTÓW Z NAJLEPSZYM WYNIKIEM:")
            for model_key, count in comparison_stats['better_counts'].items():
                print(f"     - {model_key.capitalize():15s}: {count}")
            if comparison_stats.get('equal_count', 0) > 0:
                print(f"     - Równe wyniki: {comparison_stats['equal_count']}")
        
        if comparison_stats.get('wilcoxon_pvalue') is not None:
            print(f"   - Wilcoxon p-value: {comparison_stats.get('wilcoxon_pvalue'):.4f}")
    
    def clear_cache(self):
        """Czyści cache danych"""
        self.cached_data = None
        print("✅ Cache danych wyczyszczony")
    
    def run_tests(self, base_params, n_tests=1, models_to_test=None, 
                  varying_params=None, save=False):
        """
        Uruchamia serię testów
        
        Parameters:
        -----------
        base_params : dict
            Bazowe parametry testów
        n_tests : int
            Liczba testów do uruchomienia
        models_to_test : list
            Lista krotek (nazwa_modelu, parametry) do przetestowania.
        varying_params : dict
            Słownik z listami wartości do zmiany w kolejnych testach
        save : bool
            Czy zapisać wyniki do bazy
        """
        print(f"\n🎯 ROZPOCZĘCIE SERII {n_tests} TESTÓW")
        print(f"Parametry bazowe: {json.dumps({k: v for k, v in base_params.items() if k != 'csv_path'}, indent=2, default=str)}")
        
        all_results = []
        
        for i in range(n_tests):
            test_params = base_params.copy()
            
            # Zmień parametry jeśli podano varying_params
            if varying_params:
                for param_name, values in varying_params.items():
                    if i < len(values):
                        test_params[param_name] = values[i]
                    else:
                        test_params[param_name] = values[-1]
            
            result = self.test(test_params, i+1, n_tests, models_to_test, save)
            all_results.append(result)
            
            # Przerwa między testami
            if i < n_tests - 1:
                print("\n⏳ Przygotowanie do następnego testu...")
                import time
                time.sleep(0.1)
        
        # Podsumowanie serii testów
        successful_tests = [r for r in all_results if r['success']]
        print(f"\n✅ Zakończono serię testów: {len(successful_tests)}/{n_tests} udanych")
        
        # Wyświetl podsumowanie bazy danych
        if save:
            self.db.print_summary()
        
        return all_results
    
    def test_observation_length_impact(self, base_params, n_observations_list=None,
                                     models_to_test=None, save=False):
        
        if n_observations_list is None:
            n_observations_list = [100, 250, 500, 750, 1000, 1500, 2000, 
                                  2500, 3000, 3500, 5000, 7500, 10000,15000,20000
                                  ]
        
        print(f"\n🎯 BADANIE WPŁYWU LICZBY OBSERWACJI")
        print(f"{'='*60}")
        
        all_results = []
        
        # Przygotuj dane RAZ
        max_obs = max(n_observations_list)
        cached = self.prepare_data_once(
            base_params['cutoff_km'], 
            base_params['co_ktory'], 
            max_obs
        )
        self.cached_data = cached
        
        for i, n_obs in enumerate(n_observations_list):
            print(f"\n{'='*50}")
            print(f"▶ TEST {i+1}/{len(n_observations_list)} - {n_obs} obserwacji")
            print(f"{'='*50}")
            
            # Skopiuj parametry
            test_params = base_params.copy()
            test_params['n_observations'] = n_obs
            
            # Uruchom test
            result = self.test(test_params, i+1, len(n_observations_list), 
                             models_to_test, save=False)  # Nie zapisuj do głównej bazy
            
            if result['success']:
                result['n_observations'] = n_obs
                all_results.append(result)
        
        # Zapisz wyniki do osobnego pliku
        self._save_observation_length_results(all_results)
        
        # Wizualizacja
        self._plot_observation_length_results(all_results)
        
        return all_results
    
    def _save_observation_length_results(self, results):
        """Zapisuje wyniki badania wpływu liczby obserwacji"""
        if not results:
            return
        
        # Przygotuj dane do zapisu
        data = []
        for result in results:
            if not result['success']:
                continue
            
            row = {'n_observations': result.get('n_observations', 0)}
            
            # Dodaj metryki dla każdego modelu
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
        """Tworzy wykresy wyników badania wpływu liczby obserwacji"""
        if not results:
            return
            
        data = []
        for result in results:
            if not result.get('success'):
                continue
            
            n_obs = result.get('n_observations', 0)
            row = {'n_observations': n_obs}
            
            if 'metrics' in result:
                for model_name, metrics_dict in result['metrics'].items():
                    for metric_name, value in metrics_dict.items():
                        if isinstance(value, (int, float, np.number)):
                            row[f"{model_name}_{metric_name.lower()}"] = float(value)
            data.append(row)

        if not data:
            print("⚠️ Brak danych do wykreślenia")
            return

        df = pd.DataFrame(data)
        df = df.sort_values('n_observations')

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Wpływ liczby obserwacji na jakość predykcji', 
                     fontsize=16, fontweight='bold')
        
        model_colors = {'bayesian': 'b', 'dirichlet': 'r', 
                       'gaussian': 'g', 'spatial': 'm', 'bayesian_gridsearch': 'c', 'bayesian_adaptive_search_binary': 'y'}
        
        # 1. MSE vs liczba obserwacji
        ax = axes[0, 0]
        mse_plotted = False
        for col_name in df.columns:
            if col_name.endswith('_mse'):
                model_key = col_name.replace('_mse', '')
                model_base = model_key.rsplit('_', 1)[0]
                if model_base in model_colors and not df[col_name].isna().all():
                    ax.plot(df['n_observations'], df[col_name], 
                            color=model_colors[model_base], marker='o', 
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

        # 2. Różnica MSE
        ax = axes[0, 1]
        bayesian_col = next((c for c in df.columns if c.startswith('bayesian') and c.endswith('_mse')), None)
        dirichlet_col = next((c for c in df.columns if c.startswith('dirichlet') and c.endswith('_mse')), None)

        if bayesian_col and dirichlet_col and not df[bayesian_col].isna().all() and not df[dirichlet_col].isna().all():
            mse_diff = df[bayesian_col] - df[dirichlet_col]
            ax.plot(df['n_observations'], mse_diff, 'g-', marker='^', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Liczba obserwacji')
            ax.set_ylabel('Różnica MSE')
            ax.set_title(f'Różnica MSE: {bayesian_col} vs {dirichlet_col}')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
        else:
            ax.text(0.5, 0.5, 'Brak danych do porównania MSE', ha='center', va='center', transform=ax.transAxes)

        # 3. MAE vs liczba obserwacji
        ax = axes[1, 0]
        mae_plotted = False
        for col_name in df.columns:
            if col_name.endswith('_mae'):
                model_key = col_name.replace('_mae', '')
                model_base = model_key.rsplit('_', 1)[0]
                if model_base in model_colors and not df[col_name].isna().all():
                    ax.plot(df['n_observations'], df[col_name], 
                            color=model_colors[model_base], marker='s', 
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

        # 4. Korelacja vs liczba obserwacji
        ax = axes[1, 1]
        corr_plotted = False
        for col_name in df.columns:
            if col_name.endswith('_correlation'):
                model_key = col_name.replace('_correlation', '')
                model_base = model_key.rsplit('_', 1)[0]
                if model_base in model_colors and not df[col_name].isna().all():
                    ax.plot(df['n_observations'], df[col_name], 
                            color=model_colors[model_base], marker='x', 
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


#Uruchomienie programu
# Przykład użycia
def program(base_params, n_tests, csv_path, models_to_test=None, save=False, xd={"rt":False,"it":False}, impact_models_to_test=None):
    print("================================================")
    print("🎯 SYSTEM TESTOWANIA MODELI BAYESOWSKICH")
    print("================================================")
    
    # Domyślne modele do testowania, jeśli nie podano
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
    
    # Wyświetl jakie modele będą testowane
    active = [name for name, params in models_to_test]
    print(f"TESTOWANE MODELE: {', '.join(active)}")
    print("================================================\n")
    
    test_manager = TestManager(csv_path)
    
    # 1. Uruchom standardowe testy
    print("🎯 ETAP 1: STANDARDOWE TESTY")
    if xd["rt"]:
        results = test_manager.run_tests(
            base_params, n_tests=n_tests, 
            models_to_test=models_to_test, save=save
        )
    
    # 2. Test wpływu liczby obserwacji
    print("\n🎯 ETAP 2: TEST WPŁYWU LICZBY OBSERWACJI")
    if xd["it"]:
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
            base_params, models_to_test=impact_models_to_test, save=save
        )
    
    print("\n================================================")
    print("✅ WSZYSTKIE TESTY ZAKOŃCZONE")
    print("================================================")
#endregion
#Parametry
if __name__ == "__main__":
    # ==================================================================
    # --- KONFIGURACJA TESTÓW ---
    # ==================================================================
    
    # Ścieżka do pliku CSV z danymi
    csv_path = r"C:\Users\User\Downloads\Global_2020_MarineSpeciesRichness_AquaMaps.csv"
    
    # Podstawowe parametry dla wszystkich testów (ogólne, nie specyficzne dla modelu)
    base_params = {
        'cutoff_km': 1000,      # Odcięcie od brzegu w km
        'co_ktory': 200,        # Redukcja siatki (co n-ty punkt)
        'n_observations': 50000,# Domyślna liczba obserwacji
        'n_points': 0           # (nie edytować, ustawiane automatycznie)
    }
    
    # Modele do uruchomienia w standardowej serii testów (`run_tests`)
    # Format: lista krotek [('nazwa_modelu', {parametry_modelu}), ...]
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
        ('bayesian', {
            'lengthscale': 5000,
            'variance': 1.0,
            'distance_unit': "km",
            'mcmc_samples': 5000,
            'mcmc_burn': 3000,
            'mcmc_scale': 0.05,
            'mcmc_seed': 42
        }),
        ('dirichlet', {}),
        ('spatial', {'smoothing_factor': 0.1})
        # ('gaussian', {
        #     'lengthscale_prior': (1000, 500),
        #     'variance_prior': (2, 1),
        #     'n_samples': 1000,
        #     'burn_in': 500,
        #     'step_size': 0.1
        # }) # Odkomentuj, aby dodać model GP (wolny)
    ]
    
    # Modele do uruchomienia w teście wpływu liczby obserwacji (`test_observation_length_impact`)
    impact_models_to_test = [
        
        ('bayesian', {
            'lengthscale': 5000,
            'variance': 1.0,
            'distance_unit': "km",
            'mcmc_samples': 5000,
            'mcmc_burn': 3000,
            'mcmc_scale': 0.05,
            'mcmc_seed': 42
        }),
       ('bayesian_adaptive_search_binary', {
        'variance': 1.0,
        'distance_unit': "km",
        'start_ls': 2000,
        'step_size': 2000,
        'k_steps': 5,
        'num_samples_search': 300,
        'burn_in_search': 50,
        'proposal_scale_search': 0.05,
        'mcmc_samples': 5000,
        'mcmc_burn': 3000,
        'mcmc_scale': 0.05,
        'mcmc_seed': 42
    }),
        
        ('dirichlet', {}),
        ('spatial', {'smoothing_factor': 0.1})
    ]

    # --- USTAWIENIA URUCHOMIENIA ---
    
    n_tests = 2  # Liczba testów w standardowej serii
    save_results = False  # Czy zapisać wyniki do pliku CSV
    
    # Które części programu uruchomić?
    # "rt": True -> uruchom `run_tests`
    # "it": True -> uruchom `test_observation_length_impact`
    run_options = {"rt": False, "it": True}

    # ==================================================================
    # --- URUCHOMIENIE PROGRAMU ---
    # ==================================================================
    program(base_params, 
            n_tests=n_tests, 
            csv_path=csv_path, 
            models_to_test=models_to_test,
            impact_models_to_test=impact_models_to_test,
            save=save_results, 
            xd=run_options)