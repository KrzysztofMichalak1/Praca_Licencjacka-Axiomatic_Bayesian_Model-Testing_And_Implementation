
from numba import njit
import numpy as np
from scipy.linalg import cholesky, solve_triangular
import math

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
    Model logistyczno-normalny dla danych w postaci przestrzennego procesu punktowego.
    Jest to wariant modelu z polem Gaussowskim, gdzie do transformacji latentnego
    pola `Z` na prawdopodobieństwa `p` używana jest funkcja softmax.

    Algorytm działania:
    1. Inicjalizacja (`__init__` i `przygotuj_apriori`):
       - Model przyjmuje geometrię, obserwowane indeksy i hiperparametry dla GP
         (długość skali `lengthscale`, wariancja `variance`).
       - Zliczane są obserwacje w każdej lokalizacji (`counts`).
       - Obliczana jest macierz kowariancji `K` dla latentnego pola `Z` oraz jej
         odwrotność `K_inv` i stała normalizacyjna dla priora.
         Zakładamy, że `Z` ma rozkład a priori `N(mu_prior, K)`.

    2. Główna pętla MCMC (`przygotuj_predykcyjny`):
       - Cel: Próbkowanie z rozkładu a posteriori `p(Z | counts)`.
       - Używany jest algorytm Metropolis-Hastings.
       - W każdej iteracji:
         a) Proponowany jest nowy stan `Z_proposal`.
         b) Obliczana jest gęstość log-posterior `log p(Z_proposal | counts)`, która jest
            sumą gęstości log-prior `log p(Z_proposal)` i log-wiarygodności
            `log p(counts | Z_proposal)`. Wiarygodność bazuje na rozkładzie
            wielomianowym (Multinomial), gdzie p-stwa `p` są wynikiem
            transformacji `Z` funkcją softmax.
         c) Nowy stan `Z_proposal` jest akceptowany na podstawie stosunku gęstości.
       - Po fazie burn-in, próbki `Z` są zapisywane.

    3. Obliczenie predykcji (`posterior_mean`):
       - Każda zapisana próbka `Z` jest transformowana do próbki prawdopodobieństw `p`
         za pomocą funkcji softmax.
       - Finalna predykcja jest średnią arytmetyczną tych próbek `p`.
    """
    def __init__(self, space_points, metric_func, observed_indices,
                 lengthscale=50.0, variance=1.0, distance_unit='km',
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

    def _log_posterior(self, Z):
        """
        Poprawione log-posterior dla multinomial logistic-normal.
        
        Prior: Z ~ N(mu_prior, K)
        Wiara: counts ~ Multinomial(N, softmax(Z))
        """
        # Term priora (Gauss)
        diff = Z - self.mu_prior
        prior_term = -0.5 * diff.T @ self.K_inv @ diff + self.log_prior_norm_const
        
        # Term wiarygodności (multinomial)
        p = np.exp(Z - np.max(Z))  # stabilność numeryczna
        p = p / np.sum(p)
        likelihood_term = np.sum(self.counts * np.log(p + 1e-12))
        
        return prior_term + likelihood_term

    def przygotuj_predykcyjny(self, num_samples, burn_in, proposal_scale, seed=None):
        print("\n" + "="*70)
        print("▶ [MCMC] START SAMPLERA DLA MODELU LOGISTIC-NORMAL")
        print("="*70)

        if seed is not None: 
            np.random.seed(seed)
        
        if self.K_inv is None:
            raise RuntimeError("Należy najpierw uruchomić 'przygotuj_apriori'.")

        Z_current = np.full(self.n, self.mu_prior)
        logpost_current = self._log_posterior(Z_current)

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

            logpost_prop = self._log_posterior(Z_prop)

            accept = False
            if np.isfinite(logpost_prop):
                log_alpha = logpost_prop - logpost_current
                if log_alpha >= 0 or np.log(np.random.uniform()) < log_alpha:
                    accept = True
            
            if accept:
                Z_current = Z_prop
                logpost_current = logpost_prop
                if i >= burn_in: 
                    accepted += 1
        
            if i >= burn_in:
                samples.append(Z_current.copy())
            
            if i % 500 == 0 or i == total_iterations - 1:
                status = "BURN-IN" if i < burn_in else "SAMPLING"
                acc_rate_display = accepted / max(1, i - burn_in + 1)
                print(f"   🔸 Iter {i:5d}/{total_iterations} [{status:8s}] | "
                      f"LP={logpost_current:8.1f} | AccRate={acc_rate_display:.3f}")
        
        self.samples_Z = np.array(samples)
        if len(self.samples_Z) > 0:
            # Transformacja Z -> p (softmax)
            self.samples_x = []
            for z_sample in self.samples_Z:
                p = np.exp(z_sample - np.max(z_sample))
                p = p / np.sum(p)
                self.samples_x.append(p)
            self.samples_x = np.array(self.samples_x)
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
