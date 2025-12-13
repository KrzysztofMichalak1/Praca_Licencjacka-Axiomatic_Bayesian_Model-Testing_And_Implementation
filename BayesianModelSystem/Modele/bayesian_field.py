"This module contains the BayesianFieldModel and related classes."
from numba import njit
import numpy as np
from scipy.linalg import cholesky, solve_triangular
import math

@njit(fastmath=True)
def x_from_w(w, clip_value=10.0):
    u = w 
    u_clipped = np.clip(u, -clip_value, clip_value)
    
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
    """Transformation from x to w with standardization."""
    if np.any(x <= 0):
        return None
    x1 = x[0]
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
    """Optimized log-posterior with precomputation."""
    u = w
    max_u = np.max(u)
    exp_u = np.exp(u - max_u)
    denom = np.exp(-max_u) + np.sum(exp_u)
    x0=1/denom
    x1 = np.exp(-max_u)*x0
    x_rest = exp_u * x0

    if np.any(x_rest <= 0.0) or x1 <= 0.0:
        return -1e300

    y = solve_lower_triangular(L, u)
    lp = log_norm_const - 0.5 * np.dot(y, y)

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
            except np.linalg.LinAlgError:
                print("  ⚠️ Macierz kowariancji nie jest dodatnio określona, używam diagonalnej")
                cov_prop = (proposal_scale**2) * np.eye(self.m)
                L_prop = np.sqrt(cov_prop)
        else:
            cov_prop = (proposal_scale**2) * np.eye(self.m)
            L_prop = np.sqrt(cov_prop)

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
        """Evaluates a given 'lengthscale' using K-fold cross-validation."""
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
@njit(fastmath=True)
def z_to_x_softmax(Z):
    """Stable softmax transformation."""
    max_Z = np.max(Z)
    exp_Z = np.exp(Z - max_Z)
    sum_exp_Z = np.sum(exp_Z)
    if sum_exp_Z == 0:
        # Avoid division by zero, return uniform distribution
        return np.full_like(Z, 1.0 / Z.size)
    return exp_Z / sum_exp_Z

@njit(fastmath=True)
def log_posterior_logistic_normal_fast(Z, counts, N, K_inv, log_prior_norm_const, mu_prior):
    """Computes the log-posterior for the Logistic-Normal model."""
    # Log-likelihood part
    max_Z = np.max(Z)
    log_sum_exp_Z = max_Z + np.log(np.sum(np.exp(Z - max_Z)))
    log_lik = np.dot(counts, Z) - N * log_sum_exp_Z

    if not np.isfinite(log_lik):
        return -np.inf

    # Log-prior part
    Z_centered = Z - mu_prior
    mat_vec_prod = np.dot(K_inv, Z_centered)
    quad_form = np.dot(Z_centered, mat_vec_prod)
    log_prior = log_prior_norm_const - 0.5 * quad_form
    
    if not np.isfinite(log_prior):
        return -np.inf
        
    return log_prior + log_lik


class LogisticNormalMCMC:
    """
    Logistic-Normal model for spatial point process data, implemented
    with an interface consistent with other models in this module.
    """
    def __init__(self, space_points, metric_func, observed_indices,
                 lengthscale=1000.0, variance=1.0, distance_unit='km',
                 mu_prior=0.0):
        print("\n▶ [LOGISTIC-NORMAL-MCMC] Inicjalizacja modelu...")
        self.space_points = list(space_points)
        self.metric = metric_func
        self.n = len(self.space_points)
        
        self.observed_indices = np.asarray(observed_indices)
        self.counts = np.bincount(self.observed_indices, minlength=self.n).astype(np.float64)
        self.N = self.counts.sum()

        print(f"  - Liczba punktów: {self.n}")
        print(f"  - Liczba obserwacji: {self.N}")
        
        # Hyperparameters
        self.lengthscale = lengthscale
        self.variance = variance
        self.distance_unit = distance_unit
        self.mu_prior = mu_prior
        
        self.K = None
        self.K_inv = None
        self.log_prior_norm_const = None
        
        self.samples_Z = None
        self.samples_x = None
        print("  ✔ Model gotowy.\n")

    def przygotuj_apriori(self):
        print("▶ [APRIORI] Budowa macierzy kowariancji dla procesu Gaussa...")
        
        dist_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                dist = self.metric(self.space_points[i], self.space_points[j], return_unit=self.distance_unit)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        self.K = self.variance * np.exp(-(dist_matrix / self.lengthscale)**2)
        self.K += 1e-8 * np.eye(self.n) # Jitter
        
        try:
            L = cholesky(self.K, lower=True)
            L_inv = solve_triangular(L, np.eye(self.n), lower=True)
            self.K_inv = L_inv.T @ L_inv
            logdet = 2 * np.sum(np.log(np.diag(L)))
            self.log_prior_norm_const = -0.5 * (self.n * np.log(2 * np.pi) + logdet)
            print("  ✔ Cholesky i prekomputacja stałych dla priora zakończona.")
        except np.linalg.LinAlgError:
            print("  ⚠️ Błąd Cholesky, używam bezpośredniej inwersji macierzy K.")
            self.K_inv = np.linalg.inv(self.K)
            sign, logdet = np.linalg.slogdet(self.K)
            if sign != 1:
                raise ValueError("Macierz kowariancji nie jest dodatnio określona.")
            self.log_prior_norm_const = -0.5 * (self.n * np.log(2 * np.pi) + logdet)

    def przygotuj_predykcyjny(self, num_samples, burn_in, proposal_scale, seed=None):
        print("\n" + "="*70)
        print("▶ [MCMC] START SAMPLERA DLA MODELU LOGISTIC-NORMAL")
        print("="*70)

        if seed is not None: np.random.seed(seed)
        
        if self.K_inv is None:
            raise RuntimeError("Należy najpierw uruchomić 'przygotuj_apriori'.")

        Z_current = np.full(self.n, self.mu_prior)
        logpost_current = log_posterior_logistic_normal_fast(
            Z_current, self.counts, self.N, self.K_inv, self.log_prior_norm_const, self.mu_prior
        )

        cov_prop = (proposal_scale**2) * self.K
        try:
            L_prop = np.linalg.cholesky(cov_prop)
        except np.linalg.LinAlgError:
            print("  ⚠️ Macierz kowariancji propozycji nie jest dodatnio określona, używam diagonalnej.")
            cov_prop = (proposal_scale**2) * np.eye(self.n)
            L_prop = np.sqrt(cov_prop)

        samples = []
        total_iterations = num_samples + burn_in
        accepted = 0
        
        print(f"📊 PARAMETRY MCMC:")
        print(f"   - Całkowita liczba iteracji: {total_iterations}")
        print(f"   - Burn-in: {burn_in}")
        print(f"   - Próbki posteriora: {num_samples}")
        print(f"   - Proposal scale: {proposal_scale}")
        print(f"   - Start LP: {logpost_current:.1f}")
        print("-" * 50)
        print("🔄 ROZPOCZĘCIE ITERACJI MCMC...")

        for i in range(total_iterations):
            z_norm = np.random.normal(0, 1, self.n)
            Z_prop = Z_current + L_prop @ z_norm

            logpost_prop = log_posterior_logistic_normal_fast(
                Z_prop, self.counts, self.N, self.K_inv, self.log_prior_norm_const, self.mu_prior
            )

            accept = False
            if np.isfinite(logpost_prop):
                log_alpha = logpost_prop - logpost_current
                if log_alpha >= 0 or np.log(np.random.uniform()) < log_alpha:
                    accept = True
            
            if accept:
                Z_current = Z_prop
                logpost_current = logpost_prop
                if i >= burn_in: accepted += 1
        
            if i >= burn_in:
                samples.append(Z_current.copy())
            
            if i % 500 == 0 or i == total_iterations - 1:
                status = "BURN-IN" if i < burn_in else "SAMPLING"
                acc_rate_display = accepted / max(1, i - burn_in + 1)
                print(f"   🔸 Iter {i:5d}/{total_iterations} [{status:8s}] | "
                      f"LP={logpost_current:8.1f} | AccRate={acc_rate_display:.3f}")
        
        self.samples_Z = np.array(samples)
        if len(self.samples_Z) > 0:
            self.samples_x = np.array([z_to_x_softmax(z_sample) for z_sample in self.samples_Z])
        else:
            self.samples_x = np.array([])
        
        final_acc_rate = accepted / max(1, num_samples)
        print("\n" + "="*70)
        print("✅ MCMC ZAKOŃCZONE - PODSUMOWANIE")
        print(f"   ✔ Akceptacje (sampling): {accepted}/{num_samples} ({final_acc_rate:.4f})")
        print("="*70 + "\n")

    def posterior_mean(self):
        if self.samples_x is None or len(self.samples_x) == 0:
            raise ValueError("Brak próbek posteriora. Uruchom 'przygotuj_predykcyjny'.")
        return np.mean(self.samples_x, axis=0)
    
    def posterior_quantiles(self, q=(0.025, 0.975)):
        if self.samples_x is None or len(self.samples_x) == 0:
            raise ValueError("Brak próbek posteriora. Uruchom 'przygotuj_predykcyjny'.")
        return np.quantile(self.samples_x, q, axis=0)