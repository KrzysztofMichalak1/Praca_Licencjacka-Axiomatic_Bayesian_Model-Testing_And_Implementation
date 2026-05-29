import numpy as np
from .bayesian_helpers import MultivariateNormalCholesky, log_posterior_fast, x_from_w

class ModelAksjomatycznyPreparamed:
    """
    Główny model implementujący Bayesowskie pole losowe z wykorzystaniem Procesu
    Gaussowskiego (GP) do modelowania przestrzennego.
    """
    def __init__(self, space_points, metric_func, observed_indices,
                 lengthscale, variance, distance_unit='km', p=1):
        print("\n▶ [MODEL-AKSJOMATYCZNY-PREPARAMED] Inicjalizacja modelu...")
        self.space_points = list(space_points)
        self.metric = metric_func
        self.n = len(self.space_points)
        self.m = self.n - 1
        self.p = p
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
        print("▶ [APRIORI] Budowa macierzy kowariancji (wektorowo)...")
        from .bayesian_helpers import vectorized_haversine
        
        # Punkty p2...pn (wszystkie poza pierwszym)
        pts_rest = np.array(self.space_points[1:])
        p1 = np.array([self.space_points[0]])
        
        # 1. Odległości od punktu referencyjnego p1
        if self.metric.__name__ == 'haversine':
            d_i1 = (vectorized_haversine(pts_rest, p1, return_unit=self.distance_unit).flatten() / self.lengthscale)**(self.p)
            # 2. Odległości między wszystkimi parami (poza p1)
            dist_matrix_rest = vectorized_haversine(pts_rest, pts_rest, return_unit=self.distance_unit)
        else:
            # Fallback (wolniejszy)
            d_i1 = np.array([(self.metric(pi, self.space_points[0], return_unit=self.distance_unit)/self.lengthscale)**(self.p) for pi in pts_rest])
            from scipy.spatial.distance import cdist
            dist_matrix_rest = cdist(pts_rest, pts_rest, metric='euclidean')

        d_ij = (dist_matrix_rest / self.lengthscale)**(self.p)
        
        # 3. Formuła kowariancji: cov(wi, wj) = 0.5 * sigma^2 * (d_i1 + d_j1 - d_ij)
        # Zgodnie z teorią pól aksjomatycznych sigma^2 (variance) skaluje całą macierz.
        D_i1 = d_i1[:, np.newaxis]
        D_j1 = d_i1[np.newaxis, :]
        
        self.Sigma_u = self.variance * 0.5 * (D_i1 + D_j1 - d_ij)
        
        # 4. Stabilizacja (nugget) - zwiększona dla stabilności przy dużych macierzach
        self.Sigma_u += np.eye(self.m) * 1e-8
        
        self.mvn_u = MultivariateNormalCholesky(self.Sigma_u)
        self.counts_f = self.counts.astype(np.float64)
        print("  ✔ Prekomputacja stałych zakończona.")

    def znajdz_dobry_punkt_startowy(self, n_trials=20):
        return np.zeros(self.n-1)

    def przygotuj_predykcyjny(self, num_samples, burn_in, proposal_scale, seed=None):
        print("\n" + "="*70)
        print("▶ [MCMC] START SAMPLERA ADAPTACYJNEGO")
        print("="*70)

        if seed is not None:
            np.random.seed(seed)

        w_current = self.znajdz_dobry_punkt_startowy()
        logpost_current = log_posterior_fast(w_current, self.counts_f, self.mvn_u.L,
                                           self.mvn_u.log_norm_const)

        adaptation_period = min(1000, burn_in // 2)
        if self.Sigma_u is not None:
            cov_prop = (proposal_scale**2) * self.Sigma_u
            try:
                L_prop = np.linalg.cholesky(cov_prop)
            except np.linalg.LinAlgError:
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
    
        for i in range(total_iterations):
            z = np.random.normal(0, 1, self.m)
            w_prop = w_current + L_prop @ z

            try:
                logpost_prop = log_posterior_fast(w_prop, self.counts_f, self.mvn_u.L,
                                                self.mvn_u.log_norm_const)
            except:
                logpost_prop = -np.inf

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

            if i < adaptation_period and i >= 100 and i % 50 == 0:
                current_acc_rate = np.mean(acceptance_history)
                old_scale = proposal_scale
                if current_acc_rate < 0.15: proposal_scale *= 0.8
                elif current_acc_rate > 0.35: proposal_scale *= 1.2
                
                if proposal_scale != old_scale:
                    cov_prop = (proposal_scale**2) * self.Sigma_u
                    try:
                        L_prop = np.linalg.cholesky(cov_prop)
                    except:
                        L_prop = np.sqrt((proposal_scale**2) * np.eye(self.m))
            
            if i % 1000 == 0 or i == total_iterations - 1:
                status = "BURN-IN" if i < burn_in else "SAMPLING"
                current_acc_display = (accepted_burnin / max(1, i + 1)) if i < burn_in else (accepted / max(1, i - burn_in + 1))
                print(f"   🔸 Iter {i:5d}/{total_iterations} [{status:8s}] | LP={logpost_current:8.1f} | Acc={current_acc_display:.3f}")
        
        self.samples_w = np.array(samples)
        if len(self.samples_w) > 0:
            self.samples_x = np.array([x_from_w(w) for w in self.samples_w])
        
        print(f"✔ MCMC ZAKOŃCZONE (Acc: {accepted/max(1, num_samples):.3f})\n")

    def posterior_mean(self):
        if self.samples_x is None: raise ValueError("Brak próbek posteriora.")
        return np.mean(self.samples_x, axis=0)
