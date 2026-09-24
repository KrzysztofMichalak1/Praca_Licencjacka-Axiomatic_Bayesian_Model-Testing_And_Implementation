import numpy as np
from .model_lenka_preparamed import ModelLenkaPreparamed, log_posterior_logistic_normal_fast
from scipy.linalg import solve_triangular

class ModelLenka(ModelLenkaPreparamed):
    """
    Rozszerzenie modelu `ModelLenkaPreparamed`, które automatycznie wyszukuje
    optymalną wartość hiperparametru `lengthscale`.
    """
    def __init__(self, space_points, metric_func, observed_indices,
                 variance=1.0, distance_unit='km', mu_prior=0.0,
                 start_ls=50.0, step_size=25.0, k_steps=3,
                 num_samples_search=100, burn_in_search=50, 
                 proposal_scale_search=0.05):
        
        # Enforce positive search params for numerical sanity
        start_ls = max(1e-5, start_ls)
        step_size = max(1e-5, step_size)
        
        super().__init__(space_points, metric_func, observed_indices, 
                         lengthscale=1.0, variance=variance, 
                         distance_unit=distance_unit, mu_prior=mu_prior)
        
        self.start_ls = start_ls
        self.step_size = step_size
        self.k_steps = k_steps
        self.num_samples_search = num_samples_search
        self.burn_in_search = burn_in_search
        self.proposal_scale_search = proposal_scale_search
        
        print(f"\n▶ [MODEL-LENKA] Inicjalizacja Optymalizatora:")
        print(f"   - Start lengthscale: {start_ls}")
        print(f"   - Step size: {step_size}")
        print(f"   - Kroki bisekcji (k): {k_steps}")

    def _evaluate_lengthscale(self, ls):
        print(f"    [EVAL] ls={ls:.2f}")
        try:
            # Ujednolicona, prekomputowana macierz odległości z klasą bazową dla spójności i wydajności
            if not hasattr(self, 'dist_matrix') or self.dist_matrix is None:
                self.dist_matrix = np.zeros((self.n, self.n))
                for i in range(self.n):
                    for j in range(i, self.n):
                        try:
                            dist = self.metric(self.space_points[i], self.space_points[j], return_unit=self.distance_unit)
                        except TypeError:
                            dist = self.metric(self.space_points[i], self.space_points[j])
                        self.dist_matrix[i, j] = self.dist_matrix[j, i] = dist
            
            distances = self.dist_matrix
            
            # SPÓJNOŚĆ MATEMATYCZNA: Zgodnie z równaniem 41
            K = self.variance * np.exp(-(distances / ls)**2)
            n = K.shape[0]
            
            # STABILNY ROZKŁAD CHOLESKIEGO: Adaptacyjna stabilizacja przekątnej (jitter/nugget)
            L = None
            jitter = 1e-8 * self.variance
            for attempt in range(10):
                try:
                    L = np.linalg.cholesky(K + np.eye(n) * jitter)
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0
            
            if L is None:
                # Ostateczny fallback w przypadku ekstremalnych błędów
                L = np.linalg.cholesky(K + np.eye(n) * (1e-2 * self.variance))
            
            # Wysoce stabilne obliczenie odwrócenia K_inv za pomocą solve_triangular (znacznie bezpieczniejsze niż np.linalg.inv)
            L_inv = solve_triangular(L, np.eye(n), lower=True)
            K_inv = L_inv.T @ L_inv
            
            log_det_K = 2 * np.sum(np.log(np.diag(L)))
            log_prior_norm_const = -0.5 * n * np.log(2 * np.pi) - 0.5 * log_det_K
            
            samples_lp = []
            current_z = np.random.randn(n) * 0.1
            log_post_current = log_posterior_logistic_normal_fast(
                current_z, self.counts, self.N, K_inv, log_prior_norm_const, self.mu_prior
            )
            for _ in range(self.num_samples_search + self.burn_in_search):
                proposal = current_z + np.random.randn(n) * self.proposal_scale_search
                log_post_proposal = log_posterior_logistic_normal_fast(
                    proposal, self.counts, self.N, K_inv, log_prior_norm_const, self.mu_prior
                )
                
                # Stabilne przejście MCMC dla stanów nieskończonych/not-finite
                if np.isfinite(log_post_proposal):
                    if not np.isfinite(log_post_current) or np.log(np.random.rand()) < (log_post_proposal - log_post_current):
                        current_z = proposal
                        log_post_current = log_post_proposal
                
                if _ >= self.burn_in_search:
                    samples_lp.append(log_post_current)
            
            if not samples_lp: 
                return -np.inf
            
            # Obliczamy średnią ze skończonych log-posteriorów
            finite_lps = [lp for lp in samples_lp if np.isfinite(lp)]
            mean_log_post = np.mean(finite_lps) if finite_lps else -np.inf
            print(f"    [EVAL-OK] ls={ls:.2f}, score={mean_log_post:.2f}")
            return mean_log_post
        except Exception as e:
            print(f"    [EVAL-ERR] ls={ls:.2f}, błąd: {e}")
            return -np.inf

    def przygotuj_apriori(self):
        print("▶ [MODEL-LENKA] Rozpoczynanie poszukiwania optymalnego 'lengthscale'...")
        
        # Obliczenie macierzy odległości raz na początku dla spójności i optymalizacji wydajności
        self.dist_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                try:
                    dist = self.metric(self.space_points[i], self.space_points[j], return_unit=self.distance_unit)
                except TypeError:
                    dist = self.metric(self.space_points[i], self.space_points[j])
                self.dist_matrix[i, j] = self.dist_matrix[j, i] = dist

        current_ls = self.start_ls
        current_score = self._evaluate_lengthscale(current_ls)
        
        for i in range(10):
            next_ls = current_ls + self.step_size
            next_score = self._evaluate_lengthscale(next_ls)
            if next_score > current_score:
                current_ls, current_score = next_ls, next_score
            else: break
        
        best_ls, best_score = current_ls, current_score
        if self.k_steps > 0:
            current_step = self.step_size
            for k in range(1, self.k_steps + 1):
                current_step /= 2.0
                test_points = sorted(list(set([max(1e-5, best_ls - current_step), best_ls, best_ls + current_step])))
                scores = {ls: self._evaluate_lengthscale(ls) for ls in test_points}
                new_best_ls = max(scores, key=scores.get)
                if scores[new_best_ls] > best_score:
                    best_ls, best_score = new_best_ls, scores[new_best_ls]

        self.lengthscale = best_ls
        print(f"🎯 KONIEC OPTYMALIZACJI (Finalny LS: {self.lengthscale:.2f})\n")
        
        # Wywołanie metody bazowej z odnalezionym najlepszym hiperparametrem
        super().przygotuj_apriori()
