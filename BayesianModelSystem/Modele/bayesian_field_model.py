import numpy as np
from .bayesian_helpers import MultivariateNormalCholesky, log_posterior_fast, x_from_w

class BayesianFieldModel:
    """
    Główny model implementujący Bayesowskie pole losowe z wykorzystaniem Procesu
    Gaussowskiego (GP) do modelowania przestrzennego. Prawdopodobieństwa w poszczególnych
    lokalizacjach są wynikiem transformacji latentnego (ukrytego) pola `w` za pomocą
    funkcji probit.

    Algorytm działania:
    1. Inicjalizacja (`__init__` i `przygotuj_apriori`):
       - Model przyjmuje geometrię, obserwowane indeksy i hiperparametry dla GP
         (długość skali `lengthscale`, wariancja `variance`).
       - Na podstawie `lengthscale` i odległości między punktami obliczana jest
         macierz kowariancji `Sigma` dla latentnego pola Gaussowskiego `w`.
         Zakładamy, że `w` ma rozkład a priori `N(0, Sigma)`.

    2. Główna pętla MCMC (`przygotuj_predykcyjny`):
       - Cel: Próbkowanie z rozkładu a posteriori `p(w | dane)`.
       - Używany jest algorytm Metropolis-Hastings do generowania łańcucha próbek `w`.
       - W każdej iteracji:
         a) Proponowany jest nowy stan `w_proposal` poprzez dodanie szumu do `w_current`.
         b) Obliczana jest wiarygodność (likelihood) `p(dane | w_proposal)`.
            Wiarygodność bazuje na modelu Bernoulli'ego, gdzie p-stwo sukcesu `p`
            jest wynikiem transformacji `w` funkcją probit (`p = Phi(w)`).
            Wiarygodność jest niezerowa tylko dla obserwowanych punktów.
         c) Obliczany jest stosunek gęstości `posterior` (acceptance ratio), łączący
            wiarygodność i gęstość `prior` dla `w`.
         d) Nowy stan `w_proposal` jest akceptowany z prawdopodobieństwem równym
            temu stosunkowi.
       - Pierwsze `burn_in` próbek jest odrzucanych, a reszta tworzy empiryczną
         reprezentację rozkładu `posterior`.

    3. Obliczenie predykcji (`posterior_mean`):
       - Zebrane próbki `w` są transformowane do próbek prawdopodobieństw `p`
         poprzez zastosowanie funkcji probit do każdej próbki.
       - Finalna predykcja jest średnią arytmetyczną tych próbek `p` dla każdego
         punktu siatki.
    """
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
