
import numpy as np
from .Lenks_model import LogisticNormalMCMC, log_posterior_logistic_normal_fast

import numpy as np
from scipy.linalg import solve_triangular
from .Lenks_model import LogisticNormalMCMC, log_posterior_logistic_normal_fast

class LenkAdaptiveSearchModel(LogisticNormalMCMC):
    def __init__(self, space_points, metric_func, observed_indices,
                 variance=1.0, distance_unit='km', mu_prior=0.0,
                 start_ls=3000, step_size=2000, k_steps=3,
                 num_samples_search=100, burn_in_search=50, 
                 proposal_scale_search=0.05):
        
        # Initialize the parent with a placeholder lengthscale; it will be optimized.
        super().__init__(space_points, metric_func, observed_indices, 
                         lengthscale=1.0, variance=variance, 
                         distance_unit=distance_unit, mu_prior=mu_prior)
        
        # Parameters for the adaptive search
        self.start_ls = start_ls
        self.step_size = step_size
        self.k_steps = k_steps
        self.num_samples_search = num_samples_search
        self.burn_in_search = burn_in_search
        self.proposal_scale_search = proposal_scale_search
        
        print(f"\n▶ [LENK-ADAPTIVE-SEARCH] Inicjalizacja:")
        print(f"   - Start lengthscale: {start_ls}")
        print(f"   - Step size: {step_size}")
        print(f"   - Kroki bisekcji (k): {k_steps}")

    def _evaluate_lengthscale(self, ls):
        """Quickly evaluates a lengthscale for the LogisticNormalMCMC model."""
        print(f"    [EVAL] ls={ls:.0f}")
        try:
            # Create covariance matrix directly
            from scipy.spatial.distance import cdist
            
            # Calculate distances
            if self.distance_unit == 'km':
                distances = cdist(self.space_points, self.space_points, metric='euclidean')
            elif self.distance_unit == 'latlon':
                distances = np.zeros((len(self.space_points), len(self.space_points)))
                for i in range(len(self.space_points)):
                    for j in range(len(self.space_points)):
                        distances[i, j] = self.metric(self.space_points[i], self.space_points[j])
            else:
                distances = cdist(self.space_points, self.space_points, metric=self.metric)
            
            # Create covariance matrix with jitter for numerical stability
            K = self.variance * np.exp(-distances / ls)
            n = K.shape[0]
            jitter = 1e-8  # Small jitter for numerical stability
            
            # Add jitter to diagonal
            K = K + np.eye(n) * jitter
            
            # Try Cholesky decomposition to check positive definiteness
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                print(f"    [EVAL-WARN] ls={ls:.0f}, Cholesky failed - increasing jitter")
                K = K + np.eye(n) * 1e-6  # Add more jitter
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    print(f"    [EVAL-FAIL] ls={ls:.0f}, matrix not positive definite")
                    return -np.inf
            
            # Calculate inverse using Cholesky decomposition
            K_inv = np.linalg.inv(K)
            
            # Calculate log-determinant using Cholesky
            log_det_K = 2 * np.sum(np.log(np.diag(L)))
            
            # Calculate normalization constant for prior
            log_prior_norm_const = -0.5 * n * np.log(2 * np.pi) - 0.5 * log_det_K
            
            # Run a short MCMC with this covariance structure
            samples = []
            current_z = np.random.randn(n) * 0.1  # Start near zero
            
            for _ in range(self.num_samples_search + self.burn_in_search):
                # Simple Metropolis-Hastings step
                proposal = current_z + np.random.randn(n) * self.proposal_scale_search
                
                # Calculate log posterior for current and proposal
                log_post_current = log_posterior_logistic_normal_fast(
                    current_z, self.counts, self.N, 
                    K_inv, log_prior_norm_const, self.mu_prior
                )
                log_post_proposal = log_posterior_logistic_normal_fast(
                    proposal, self.counts, self.N, 
                    K_inv, log_prior_norm_const, self.mu_prior
                )
                
                # Accept/reject
                if np.log(np.random.rand()) < (log_post_proposal - log_post_current):
                    current_z = proposal
                
                # Store after burn-in
                if _ >= self.burn_in_search:
                    samples.append(current_z.copy())
            
            if not samples:
                print(f"    [EVAL-FAIL] ls={ls:.0f}, no samples collected")
                return -np.inf
                
            # Calculate the mean log-posterior of the generated samples
            n_samples_for_eval = min(50, len(samples))
            log_post_samples = []
            
            for z_sample in samples[:n_samples_for_eval]:
                log_post = log_posterior_logistic_normal_fast(
                    z_sample, self.counts, self.N, 
                    K_inv, log_prior_norm_const, self.mu_prior
                )
                if np.isfinite(log_post):
                    log_post_samples.append(log_post)
            
            mean_log_post = np.mean(log_post_samples) if log_post_samples else -np.inf
            print(f"    [EVAL-OK] ls={ls:.0f}, score={mean_log_post:.2f}")
            return mean_log_post
            
        except Exception as e:
            print(f"    [EVAL-ERROR] ls={ls:.0f}: {str(e)[:100]}...")
            return -np.inf

    def przygotuj_apriori(self):
        """
        Overrides the parent method to perform an adaptive search for the optimal
        lengthscale before calling the final prior preparation.
        """
        print("▶ [ADAPTIVE SEARCH] Rozpoczynanie poszukiwania optymalnego 'lengthscale'...")
        
        # Ensure we have counts data
        if not hasattr(self, 'counts') or self.counts is None:
            raise ValueError("Counts data must be set before preparing prior")
        
        # --- PHASE 1: Coarse Grid Search ---
        print(f"\n📌 FAZA 1: Szukanie zgrubnego optimum (krok = {self.step_size})")
        print("-" * 50)
        
        current_ls = self.start_ls
        current_score = self._evaluate_lengthscale(current_ls)
        
        # Search upwards
        for i in range(10):  # Limit to 10 steps for safety
            next_ls = current_ls + self.step_size
            next_score = self._evaluate_lengthscale(next_ls)
            
            if next_score > current_score:
                current_ls = next_ls
                current_score = next_score
                print(f"  ↑ Lepsze ls={current_ls:.0f}, score={current_score:.2f}")
            else:
                # Score dropped, we probably passed the peak
                break
        
        best_ls = current_ls
        best_score = current_score

        print(f"\n✅ ZGRUBNE OPTIMUM: ls={best_ls:.0f}, score={best_score:.2f}")
        
        # --- PHASE 2: Bisection/Fine-tuning Search ---
        if self.k_steps > 0:
            print(f"\n📌 FAZA 2: {self.k_steps} kroków bisekcji dla 'lengthscale'")
            print("-" * 50)
            
            current_step = self.step_size
            for k in range(1, self.k_steps + 1):
                current_step /= 2.0
                print(f"\n  🔄 KROK BISEKCJI {k}/{self.k_steps} (step = {current_step:.0f}):")
                
                # Test left, center, and right points
                test_points = [
                    max(100, best_ls - current_step),  # Ensure ls > 0
                    best_ls,
                    best_ls + current_step
                ]
                
                # Remove duplicates
                test_points = list(set(test_points))
                test_points.sort()
                
                print(f"    Testowane punkty: {[f'{p:.0f}' for p in test_points]}")
                
                # Evaluate all test points
                scores = {}
                for ls in test_points:
                    scores[ls] = self._evaluate_lengthscale(ls)
                
                # Find the best lengthscale among the tested points
                new_best_ls = max(scores, key=scores.get)
                new_best_score = scores[new_best_ls]
                
                if new_best_score > best_score:
                    improvement = new_best_score - best_score
                    print(f"    ✅ Znaleziono lepszy ls: {new_best_ls:.0f} "
                          f"(poprawa: {improvement:.2f})")
                    best_ls = new_best_ls
                    best_score = new_best_score
                else:
                    print(f"    ℹ️  Najlepszy ls pozostaje: {best_ls:.0f}")

        # Set the found optimal lengthscale
        self.lengthscale = best_ls
        print("\n" + "=" * 60)
        print(f"🎯 KONIEC OPTYMALIZACJI 'lengthscale'")
        print(f"   Finalny 'lengthscale': {self.lengthscale:.0f}")
        print(f"   Finalny score: {best_score:.2f}")
        print("=" * 60 + "\n")
        
        # Now, call the original przygotuj_apriori from the parent class
        # with the optimized lengthscale.
        print("▶ [APRIORI] Finalne przygotowanie modelu z optymalnym 'lengthscale'...")
        super().przygotuj_apriori()