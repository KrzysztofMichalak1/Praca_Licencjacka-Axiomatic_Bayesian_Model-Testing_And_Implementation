"""This module contains the BayesianGaussianProcess class."""
import numpy as np
import scipy.linalg as la
from ..data.metric import haversine

class BayesianGaussianProcess:
    """Bayesian Gaussian Process model for comparison."""
    
    def __init__(self, space_points, observed_indices, 
                 lengthscale_prior=(1000, 500), variance_prior=(2, 1)):
        self.space_points = np.array(space_points)
        self.observed_indices = np.array(observed_indices)
        self.n_points = len(space_points)
        
        self.counts = np.bincount(observed_indices, minlength=self.n_points)
        self.total_obs = self.counts.sum()
        
        self.ls_mu, self.ls_sigma = lengthscale_prior
        self.var_alpha, self.var_beta = variance_prior
        
        print("▶ [OPTIMIZED BAYESIAN GP] Prekomputacja macierzy odległości...")
        self.distance_matrix = self._precompute_distance_matrix()
        self.y = self.counts / self.total_obs
        
        print(f"  - Punkty: {self.n_points}, Obserwacje: {self.total_obs}")
        print(f"  - Counts: min={self.counts.min()}, max={self.counts.max()}")
    
    def _precompute_distance_matrix(self):
        """Precomputes the distance matrix."""
        dist_matrix = np.zeros((self.n_points, self.n_points))
        for i in range(self.n_points):
            for j in range(i, self.n_points):
                dist = haversine(self.space_points[i], self.space_points[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix
    
    def rbf_kernel_fast(self, lengthscale, variance, jitter=1e-6):
        """Fast RBF kernel with precomputation."""
        K = variance * np.exp(-0.5 * (self.distance_matrix / lengthscale)**2)
        K += np.eye(self.n_points) * jitter
        return K
    
    def log_prior(self, lengthscale, variance):
        """Log prior for the parameters."""
        if lengthscale <= 0 or variance <= 0:
            return -np.inf
            
        lp_ls = -0.5 * ((np.log(lengthscale) - np.log(self.ls_mu)) / self.ls_sigma)**2 - np.log(lengthscale)
        lp_var = -(self.var_alpha + 1) * np.log(variance) - self.var_beta / variance
        
        return lp_ls + lp_var
    
    def log_likelihood_fast(self, lengthscale, variance):
        """Fast likelihood with precomputation."""
        K = self.rbf_kernel_fast(lengthscale, variance)
        
        try:
            L = la.cholesky(K, lower=True)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            alpha = la.solve_triangular(L, self.y, lower=True)
            alpha = la.solve_triangular(L.T, alpha, lower=False)
            
            log_like = -0.5 * self.y.dot(alpha) - 0.5 * log_det - 0.5 * self.n_points * np.log(2 * np.pi)
            return log_like
        except la.LinAlgError:
            return -np.inf
    
    def log_posterior_fast(self, lengthscale, variance):
        """Fast posterior with precomputation."""
        lp = self.log_prior(lengthscale, variance)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood_fast(lengthscale, variance)
        if not np.isfinite(ll):
            return -np.inf
            
        return lp + ll
    
    def sample_posterior(self, n_samples=5000, burn_in=2000, step_size=0.1):
        """Optimized MCMC with precomputation."""
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
            proposed_ls = current_ls * np.exp(np.random.normal(0, step_size))
            proposed_var = current_var * np.exp(np.random.normal(0, step_size))
            
            proposed_log_post = self.log_posterior_fast(proposed_ls, proposed_var)
            
            log_alpha = proposed_log_post - current_log_post
            log_alpha += np.log(proposed_ls) - np.log(current_ls)
            log_alpha += np.log(proposed_var) - np.log(current_var)
            
            accept = False
            if np.log(np.random.rand()) < log_alpha:
                current_ls = proposed_ls
                current_var = proposed_var
                current_log_post = proposed_log_post
                if i >= burn_in:
                    accepted += 1
                accept = True
            
            if i >= burn_in:
                samples_ls.append(current_ls)
                samples_var.append(current_var)
            
            current_acc_rate = accepted / max(1, i - burn_in) if i > burn_in else accepted / max(1, i)
            
            if i % 200 == 0 or i == total_iter - 1 or (i < 100 and i % 50 == 0):
                status = "BURN-IN" if i < burn_in else "SAMPLING"
                progress = (i + 1) / total_iter * 100
                
                print(f"   🔸 Iter {i:5d}/{total_iter} [{status:8s}] | "
                      f"LP={current_log_post:8.1f} | "
                      f"ls={current_ls:8.1f} | "
                      f"var={current_var:6.3f} | "
                      f"AccRate={current_acc_rate:6.3f} | "
                      f"Progress: {progress:5.1f}%")
            
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
        """Posterior predictive distribution."""
        if not hasattr(self, 'samples_ls'):
            raise ValueError("Najpierw uruchom sample_posterior()")
        
        print(f"▶ [OPTIMIZED GP] Obliczanie rozkładu predykcyjnego...")
        
        mean_ls = np.mean(self.samples_ls)
        mean_var = np.mean(self.samples_var)
        
        print(f"  - Używam lengthscale: {mean_ls:.1f}")
        print(f"  - Używam variance: {mean_var:.3f}")
        
        K = self.rbf_kernel_fast(mean_ls, mean_var)
        
        L = la.cholesky(K, lower=True)
        alpha = la.solve_triangular(L, self.y, lower=True)
        alpha = la.solve_triangular(L.T, alpha, lower=False)
        
        predictive_mean = K @ alpha
        
        predictive_probs = predictive_mean / predictive_mean.sum()
        
        print(f"  - Suma predykcji: {predictive_mean.sum():.6f}")
        print(f"  - Min predykcja: {np.min(predictive_probs):.2e}")
        print(f"  - Max predykcja: {np.max(predictive_probs):.2e}")
        
        return predictive_probs

    def log_likelihood(self, lengthscale, variance, jitter=1e-6):
        """Original likelihood for compatibility."""
        return self.log_likelihood_fast(lengthscale, variance)
    
    def log_posterior(self, lengthscale, variance):
        """Original posterior for compatibility."""
        return self.log_posterior_fast(lengthscale, variance)
