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
        print(f"    [EVAL] ls={ls:.0f}")
        try:
            from scipy.spatial.distance import cdist
            if self.distance_unit == 'km':
                distances = cdist(self.space_points, self.space_points, metric='euclidean')
            else:
                # Ujednolicona logika mierzenia dystansu z klasą bazową
                distances = np.zeros((len(self.space_points), len(self.space_points)))
                for i in range(len(self.space_points)):
                    for j in range(len(self.space_points)):
                        try:
                            distances[i, j] = self.metric(self.space_points[i], self.space_points[j], return_unit=self.distance_unit)
                        except TypeError:
                            distances[i, j] = self.metric(self.space_points[i], self.space_points[j])
            
            # SPÓJNOŚĆ MATEMATYCZNA: Zgodnie z równaniem 41
            K = self.variance * np.exp(-(distances / ls)**2)
            n = K.shape[0]
            K = K + np.eye(n) * 1e-8
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K = K + np.eye(n) * 1e-6
                L = np.linalg.cholesky(K)
            
            K_inv = np.linalg.inv(K)
            log_det_K = 2 * np.sum(np.log(np.diag(L)))
            log_prior_norm_const = -0.5 * n * np.log(2 * np.pi) - 0.5 * log_det_K
            
            samples = []
            current_z = np.random.randn(n) * 0.1
            for _ in range(self.num_samples_search + self.burn_in_search):
                proposal = current_z + np.random.randn(n) * self.proposal_scale_search
                log_post_current = log_posterior_logistic_normal_fast(
                    current_z, self.counts, self.N, K_inv, log_prior_norm_const, self.mu_prior
                )
                log_post_proposal = log_posterior_logistic_normal_fast(
                    proposal, self.counts, self.N, K_inv, log_prior_norm_const, self.mu_prior
                )
                if np.log(np.random.rand()) < (log_post_proposal - log_post_current):
                    current_z = proposal
                if _ >= self.burn_in_search:
                    samples.append(current_z.copy())
            
            if not samples: return -np.inf
            n_samples_for_eval = min(50, len(samples))
            log_post_samples = [log_posterior_logistic_normal_fast(z, self.counts, self.N, K_inv, log_prior_norm_const, self.mu_prior) for z in samples[:n_samples_for_eval]]
            mean_log_post = np.mean([lp for lp in log_post_samples if np.isfinite(lp)]) if any(np.isfinite(log_post_samples)) else -np.inf
            print(f"    [EVAL-OK] ls={ls:.0f}, score={mean_log_post:.2f}")
            return mean_log_post
        except Exception as e:
            print(f"    [EVAL-ERR] ls={ls:.0f}, błąd: {e}")
            return -np.inf

    def przygotuj_apriori(self):
        print("▶ [MODEL-LENKA] Rozpoczynanie poszukiwania optymalnego 'lengthscale'...")
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