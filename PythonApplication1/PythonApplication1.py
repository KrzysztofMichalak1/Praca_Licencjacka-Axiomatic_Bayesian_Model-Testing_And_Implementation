#MODELE
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

        self.Sigma_u = None  # Macierz kowariancji dla znormalizowanych zmiennych
        self.mvn_u = None     # Rozkład MVN dla u
        self.samples_w = None # Próbki dla w (w_i / d(p1,pi))
        self.samples_x = None # Próbki dla x
        
        # ✅ OPTYMALIZACJA 1: Prekomputacja stałych
        self.counts_f = None  # counts jako float64
        
        print("  ✔ Model gotowy.\n")

    def przygotuj_apriori(self):
        print("▶ [APRIORI] Budowa macierzy kowariancji dla ZNORMALIZOWANYCH u...")
    
        p1 = self.space_points[0]
    
        # 1. Buduj macierz kowariancji Σ_w dla oryginalnych w
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
        """Znajduje punkt startowy który unika trywialnych rozwiązań"""
        return np.zeros(self.n-1)

    def przygotuj_predykcyjny(self, num_samples, burn_in, proposal_scale, seed=None):
        print("\n" + "="*70)
        print("▶ [MCMC] START SAMPLERA Z ADAPTACYJNYM MCMC (POPRAWIONA WERSJA)")
        print("="*70)

        # Inicjalizacja timerów
        import time
        timers = {
            'log_posterior': 0.0,
            'proposal_generation': 0.0,
            'transform_w_to_x': 0.0,
            'adaptation': 0.0,
            'other': 0.0
        }
        total_start_time = time.time()

        if seed is not None:
            np.random.seed(seed)

        # Znajdź dobry punkt startowy
        w_current = self.znajdz_dobry_punkt_startowy()
    
        # Oblicz początkową wartość log-posterior
        logpost_current = log_posterior_fast(w_current, self.counts_f, self.mvn_u.L, 
                                           self.mvn_u.log_norm_const)

        # Inicjalizacja adaptacyjnego MCMC
        adaptation_period = min(1000, burn_in // 2)
        adaptation_updates = 0
    
        # Inicjalizacja macierzy kowariancji propozycji
        # Używamy przeskalowanej macierzy apriori jako punktu startowego
        if self.Sigma_u is not None:
            # Upewnij się, że macierz jest dodatnio określona
            cov_prop = (proposal_scale**2) * self.Sigma_u
            try:
                # Spróbuj rozkład Cholesky'ego dla generowania propozycji
                L_prop = np.linalg.cholesky(cov_prop)
                use_cholesky = True
            except np.linalg.LinAlgError:
                print("  ⚠️ Macierz kowariancji nie jest dodatnio określona, używam diagonalnej")
                cov_prop = (proposal_scale**2) * np.eye(self.m)
                L_prop = np.sqrt(cov_prop)  # Dla macierzy diagonalnej
                use_cholesky = True
        else:
            cov_prop = (proposal_scale**2) * np.eye(self.m)
            L_prop = np.sqrt(cov_prop)
            use_cholesky = True

        samples = []
        total_iterations = num_samples + burn_in
        accepted = 0
        accepted_burnin = 0
    
        # Śledzenie statystyk akceptacji dla adaptacji
        acceptance_history = []
    
        print(f"📊 PARAMETRY MCMC:")
        print(f"   - Całkowita liczba iteracji: {total_iterations}")
        print(f"   - Burn-in: {burn_in}")
        print(f"   - Próbki posteriora: {num_samples}")
        print(f"   - Adaptacja przez: {adaptation_period} iteracji")
        print(f"   - Wymiar przestrzeni stanów: {self.m}")
        print(f"   - Początkowy proposal scale: {proposal_scale}")
        print(f"   - Start LP: {logpost_current:.1f}")
        print(f"   - Używam Cholesky: {use_cholesky}")
        print("-" * 50)

        print("🔄 ROZPOCZĘCIE ITERACJI MCMC...")
    
        for i in range(total_iterations):
            iter_start_time = time.time()
        
            # GENEROWANIE PROPOZYCJI - POPRAWIONA WERSJA
            prop_start = time.time()
            try:
                if use_cholesky:
                    # Generuj propozycję używając rozkładu Cholesky'ego
                    z = np.random.normal(0, 1, self.m)
                    w_prop = w_current + L_prop @ z
                else:
                    # Fallback: niezależne propozycje dla każdej składowej
                    print("gówno się odpaliło")
                    w_prop = w_current + np.random.normal(0, proposal_scale, self.m)
            except Exception as e:
                # Awaryjne generowanie propozycji
                print("ekstra gówno się odpaliło")
                w_prop = w_current + np.random.normal(0, 0.1, self.m)
            timers['proposal_generation'] += time.time() - prop_start

            # OBLICZENIE LOG-POSTERIOR DLA PROPOZYCJI
            lp_start = time.time()
            try:
                logpost_prop = log_posterior_fast(w_prop, self.counts_f, self.mvn_u.L, 
                                                self.mvn_u.log_norm_const)
            except (ValueError, RuntimeError) as e:
                # Jeśli obliczenia się nie powiodą, odrzuć propozycję
                logpost_prop = -np.inf
            timers['log_posterior'] += time.time() - lp_start

            # DECYZJA O AKCEPTACJI - SYMETRYCZNY RANDOM WALK
            accept = False
            if np.isfinite(logpost_prop):
                log_alpha = logpost_prop - logpost_current
            
                # Symetryczny random walk - nie ma dodatkowych członów
                if log_alpha >= 0 or np.log(np.random.uniform()) < log_alpha:
                    accept = True
                    

            # AKTUALIZACJA STANU
            if accept:
                w_current = w_prop
                logpost_current = logpost_prop
                if i >= burn_in:
                    accepted += 1
                else:
                    accepted_burnin += 1
        
            # ZAPIS PRÓBEK PO BURN-IN
            if i >= burn_in:
                samples.append(w_current.copy())
            
            # ŚLEDZENIE HISTORII AKCEPTACJI DLA ADAPTACJI
            acceptance_history.append(1 if accept else 0)
            if len(acceptance_history) > 100:
                acceptance_history.pop(0)

            adaptation_start = time.time()
            if i < adaptation_period and i >= 100:
                current_acc_rate = np.mean(acceptance_history[-100:]) if len(acceptance_history) >= 100 else np.mean(acceptance_history)
            
                # Adaptuj co 50 iteracji
                if i % 50 == 0:
                    old_scale = proposal_scale
                
                    if current_acc_rate < 0.15:
                        # Za mało akceptacji - zmniejsz krok
                        proposal_scale *= 0.8
                        adaptation_updates += 1
                    elif current_acc_rate > 0.35:
                        # Za dużo akceptacji - zwiększ krok  
                        proposal_scale *= 1.2
                        adaptation_updates += 1
                    # Dla 0.15-0.35 zostaw bez zmian (optymalny zakres)
                
                    # Aktualizuj macierz kowariancji jeśli zmienił się scale
                    if proposal_scale != old_scale:
                        if self.Sigma_u is not None and use_cholesky:
                            cov_prop = (proposal_scale**2) * self.Sigma_u
                            try:
                                L_prop = np.linalg.cholesky(cov_prop)
                            except np.linalg.LinAlgError:
                                # Fallback na diagonalną
                                cov_prop = (proposal_scale**2) * np.eye(self.m)
                                L_prop = np.sqrt(cov_prop)
                        else:
                            cov_prop = (proposal_scale**2) * np.eye(self.m)
                            L_prop = np.sqrt(cov_prop)
                    
                        if i % 200 == 0:  # Rzadziej wypisuj informacje
                            print(f"   🔄 ADAPTACJA: scale {old_scale:.4f} → {proposal_scale:.4f} (acc: {current_acc_rate:.3f})")
        
            timers['adaptation'] += time.time() - adaptation_start

            # WYPISYWANIE STATUSU - POPRAWIONA WERSJA
            if i % 500 == 0 or i == total_iterations - 1 or (i < 100 and i % 50 == 0):
                transform_start = time.time()
                try:
                    x_current = x_from_w(w_current)
                    x_min, x_max = np.min(x_current), np.max(x_current)
                except:
                    x_min, x_max = np.nan, np.nan
                timers['transform_w_to_x'] += time.time() - transform_start
            
                status = "BURN-IN" if i < burn_in else "SAMPLING"
                progress = (i + 1) / total_iterations * 100
            
                # Oblicz aktualny wskaźnik akceptacji
                if i < burn_in:
                    current_acc_display = accepted_burnin / max(1, i + 1)
                else:
                    samples_so_far = i - burn_in + 1
                    current_acc_display = accepted / max(1, samples_so_far)
            
                print(f"   🔸 Iter {i:5d}/{total_iterations} [{status:8s}] | "
                      f"LP={logpost_current:8.1f} | "
                      f"AccRate={current_acc_display:6.3f} | "
                      f"Scale={proposal_scale:6.4f} | "
                      f"x_range=[{x_min:.2e},{x_max:.2e}] | "
                      f"Progress: {progress:5.1f}%")
        
            timers['other'] += time.time() - iter_start_time - (
                timers['log_posterior'] + timers['proposal_generation'] + 
                timers['transform_w_to_x'] + timers['adaptation']
            )

        # TRANSFORMACJA PRÓBEK NA X - OPTYMALIZACJA
        transform_samples_start = time.time()
        self.samples_w = np.array(samples)
    
        # Zoptymalizowana transformacja wszystkich próbek na raz
        if len(self.samples_w) > 0:
            # Oblicz x dla wszystkich próbek jednocześnie
            x_samples = []
            for w_sample in self.samples_w:
                try:
                    x_sample = x_from_w(w_sample)
                    x_samples.append(x_sample)
                except (ValueError, RuntimeError):
                    # W przypadku błędu, użyj jednorodnego rozkładu jako fallback
                    x_samples.append(np.ones(self.n) / self.n)
            self.samples_x = np.array(x_samples)
        else:
            self.samples_x = np.array([])
        
        timers['transform_w_to_x'] += time.time() - transform_samples_start

        total_time = time.time() - total_start_time
        final_acc_rate = accepted / max(1, num_samples)
        burnin_acc_rate = accepted_burnin / max(1, burn_in)

        print("\n" + "="*70)
        print("✅ MCMC ZAKOŃCZONE - PODSUMOWANIE")
        print("="*70)
        print(f"📊 STATYSTYKI:")
        print(f"   ✔ Próbek posteriora:           {len(self.samples_x)}")
        print(f"   ✔ Akceptacje (burn-in):        {accepted_burnin}/{burn_in} ({burnin_acc_rate:.4f})")
        print(f"   ✔ Akceptacje (sampling):       {accepted}/{num_samples} ({final_acc_rate:.4f})")
        print(f"   ✔ Aktualizacji adaptacyjnych:  {adaptation_updates}")
        print(f"   ✔ Końcowy proposal scale:      {proposal_scale:.4f}")
    
        # DIAGNOSTYKA PRÓBEK
        if len(self.samples_x) > 0:
            x_mean = np.mean(self.samples_x, axis=0)
            print(f"   ✔ Średnia posterior - min:     {np.min(x_mean):.2e}")
            print(f"   ✔ Średnia posterior - max:     {np.max(x_mean):.2e}")
            print(f"   ✔ Średnia posterior - suma:    {np.sum(x_mean):.6f}")

        print(f"\n⏱️  CZAS WYKONANIA FUNKCJI:")
        print(f"   • log_posterior:           {timers['log_posterior']:.2f}s ({timers['log_posterior']/total_time*100:.1f}%)")
        print(f"   • proposal_generation:     {timers['proposal_generation']:.2f}s ({timers['proposal_generation']/total_time*100:.1f}%)")
        print(f"   • transform_w_to_x:        {timers['transform_w_to_x']:.2f}s ({timers['transform_w_to_x']/total_time*100:.1f}%)")
        print(f"   • adaptation:              {timers['adaptation']:.2f}s ({timers['adaptation']/total_time*100:.1f}%)")
        print(f"   • other:                   {timers['other']:.2f}s ({timers['other']/total_time*100:.1f}%)")
        print(f"   • CAŁKOWITY CZAS:          {total_time:.2f}s")

        print(f"\n📈 WYDAJNOŚĆ:")
        iterations_per_second = total_iterations / total_time if total_time > 0 else 0
        print(f"   • Iteracje na sekundę:     {iterations_per_second:.1f}")
        print(f"   • Czas na iterację:        {total_time/total_iterations*1000:.1f}ms")
        print("="*70 + "\n")
    def posterior_mean(self):
        """Oblicza średnią posterior"""
        if self.samples_x is None:
            raise ValueError("Brak próbek posteriora. Uruchom najpierw przygotuj_predykcyjny().")
        print("▶ [POST] Liczę średnią posterior...")
        return np.mean(self.samples_x, axis=0)
    def posterior_quantiles(self, q=(0.025, 0.975)):
        """Oblicza kwantyle posterior"""
        if self.samples_x is None:
            raise ValueError("Brak próbek posteriora.")
        return np.quantile(self.samples_x, q, axis=0)
    def get_u_samples(self):
        """Zwraca próbki dla znormalizowanych zmiennych u"""
        if self.samples_w is None:
            raise ValueError("Brak próbek posteriora.")
        return self.samples_w 
    def get_w_samples(self):
        """Zwraca próbki dla oryginalnych zmiennych w (w_i / d(p1,pi))"""
        return self.samples_w
#region
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
            'Bayesian': metrics_bayesian['MSE'],
            'Dirichlet': metrics_dirichlet['MSE'],
            'GP': metrics_gp['MSE'],
            'Spatial': metrics_spatial['MSE']
        }
        best_model_mse = min(models_mse, key=models_mse.get)
        
        # Przygotuj nowy wiersz
        new_row = {
            'test_id': test_id,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_points': test_params['n_points'],
            'n_observations': test_params['n_observations'],
            'lengthscale': test_params['lengthscale'],
            'variance': test_params['variance'],
            'mcmc_samples': test_params['mcmc_samples'],
            'mcmc_burn': test_params['mcmc_burn'],
            'mcmc_scale': test_params['mcmc_scale'],
            'mcmc_seed': test_params['mcmc_seed'],
            
            # Metryki Bayesian Field
            'bayesian_mse': float(metrics_bayesian['MSE']),
            'bayesian_mae': float(metrics_bayesian['MAE']),
            'bayesian_rmse': float(metrics_bayesian['RMSE']),
            'bayesian_correlation': float(metrics_bayesian['Correlation']),
            'bayesian_covariance': float(metrics_bayesian['Covariance']),
            
            # Metryki Dirichlet
            'dirichlet_mse': float(metrics_dirichlet['MSE']),
            'dirichlet_mae': float(metrics_dirichlet['MAE']),
            'dirichlet_rmse': float(metrics_dirichlet['RMSE']),
            'dirichlet_correlation': float(metrics_dirichlet['Correlation']),
            'dirichlet_covariance': float(metrics_dirichlet['Covariance']),
            
            # Metryki Gaussian Process
            'gp_mse': float(metrics_gp['MSE']),
            'gp_mae': float(metrics_gp['MAE']),
            'gp_rmse': float(metrics_gp['RMSE']),
            'gp_correlation': float(metrics_gp['Correlation']),
            'gp_covariance': float(metrics_gp['Covariance']),
            
            # Metryki Spatial Smoothing
            'spatial_mse': float(metrics_spatial['MSE']),
            'spatial_mae': float(metrics_spatial['MAE']),
            'spatial_rmse': float(metrics_spatial['RMSE']),
            'spatial_correlation': float(metrics_spatial['Correlation']),
            'spatial_covariance': float(metrics_spatial['Covariance']),
            
            # Różnice i najlepsze modele
            'mse_diff': float(metrics_bayesian['MSE'] - metrics_dirichlet['MSE']),
            'mae_diff': float(metrics_bayesian['MAE'] - metrics_dirichlet['MAE']),
            'correlation_diff': float(metrics_bayesian['Correlation'] - metrics_dirichlet['Correlation']),
            'better_model_mse': best_model_mse,
            'better_model_mae': 'Bayesian' if metrics_bayesian['MAE'] < metrics_dirichlet['MAE'] else 'Dirichlet',
            
            # Statystyki porównania
            'bayesian_better_count': int(comparison_stats['bayesian_better_count']),
            'dirichlet_better_count': int(comparison_stats['dirichlet_better_count']),
            'gp_better_count': int(comparison_stats['gp_better_count']),
            'spatial_better_count': int(comparison_stats['spatial_better_count']),
            'equal_count': int(comparison_stats['equal_count']),
            'wilcoxon_pvalue': float(comparison_stats['wilcoxon_pvalue']),
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
        if models_to_test is None:
            models_to_test = [
                ('bayesian', {
                    'lengthscale': test_params['lengthscale'],
                    'variance': test_params['variance'],
                    'distance_unit': test_params.get('distance_unit', 'km')
                }),
                ('dirichlet', {}),
                ('spatial', {'smoothing_factor': 0.1})
            ]
        
        # Wyświetl jakie modele będą testowane
        active_models = [name for name, params in models_to_test]
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
            for model_name, model_params in models_to_test:
                model_full_name = "Unknown"
                
                try:
                    if model_name == 'bayesian':
                        model_full_name = "Bayesian Field Model"
                        print(f"\n--- MODEL: {model_full_name.upper()} ---")
                        
                        constructor_params = {
                            'space_points': points,
                            'metric_func': haversine,
                            'observed_indices': obs_idx,
                            **model_params
                        }
                        
                        model = BayesianFieldModel(**constructor_params)
                        model.przygotuj_apriori()
                        model.przygotuj_predykcyjny(
                            num_samples=test_params['mcmc_samples'],
                            burn_in=test_params['mcmc_burn'],
                            proposal_scale=test_params['mcmc_scale'],
                            seed=test_params['mcmc_seed'] + test_number
                        )
                        pred = model.posterior_mean()

                    elif model_name == 'dirichlet':
                        model_full_name = "Dirichlet Model"
                        print(f"\n--- MODEL: {model_full_name.upper()} ---")
                        model = DirichletModel(obs_idx, len(gdf))
                        pred = model.posterior_mean()

                    elif model_name == 'gaussian':
                        model_full_name = "Gaussian Process Model"
                        print(f"\n--- MODEL: {model_full_name.upper()} ---")
                        
                        gp_params = {
                            'lengthscale_prior': (1000, 500),
                            'variance_prior': (2, 1),
                            **model_params
                        }
                        
                        model = BayesianGaussianProcess(points, obs_idx, **gp_params)
                        model.sample_posterior(n_samples=1000, burn_in=500, step_size=0.1)
                        pred = model.posterior_predictive()

                    elif model_name == 'spatial':
                        model_full_name = "Spatial Smoothing Model"
                        print(f"\n--- MODEL: {model_full_name.upper()} ---")
                        model = BayesianSpatialSmoothing(points, obs_idx, **model_params)
                        pred = model.posterior_mean()
                        
                    else:
                        print(f"⚠️ Nieznany model: {model_name}")
                        continue
                        
                    predictions[model_name] = pred
                    models[model_name] = model
                    metrics[model_name] = oblicz_metryki(true_probs, pred, model_full_name, verbose=True)

                except Exception as e:
                    print(f"❌ Błąd podczas uruchamiania modelu {model_full_name}: {e}")
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
                test_id = self._save_to_database(test_params, metrics, 
                                               comparison_stats, duration)
            
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
    
    def _save_to_database(self, test_params, metrics, comparison_stats, duration):
        """Zapisuje wyniki do bazy danych"""
        # Przygotuj słownik z metrykami
        result_dict = {
            'bayesian': metrics.get('bayesian', {}),
            'dirichlet': metrics.get('dirichlet', {}),
            'gaussian': metrics.get('gaussian', {}),
            'spatial': metrics.get('spatial', {})
        }
        
        # Utwórz kompletne metryki dla wszystkich modeli
        complete_metrics = {}
        for model_name in ['bayesian', 'dirichlet', 'gaussian', 'spatial']:
            model_metrics = result_dict.get(model_name, {})
            for metric_name in ['MSE', 'MAE', 'RMSE', 'Correlation', 'Covariance']:
                key = f'{model_name}_{metric_name.lower()}'
                complete_metrics[key] = float(model_metrics.get(metric_name, 0.0))
        
        # Przygotuj porównanie
        better_counts = comparison_stats.get('better_counts', {})
        
        return self.db.save_test_results(
            test_params, 
            metrics.get('bayesian', {}),
            metrics.get('dirichlet', {}),
            metrics.get('gaussian', {}),
            metrics.get('spatial', {}),
            {
                'bayesian_better_count': better_counts.get('bayesian', 0),
                'dirichlet_better_count': better_counts.get('dirichlet', 0),
                'gp_better_count': better_counts.get('gaussian', 0),
                'spatial_better_count': better_counts.get('spatial', 0),
                'equal_count': comparison_stats.get('equal_count', 0),
                'wilcoxon_pvalue': comparison_stats.get('wilcoxon_pvalue', 1.0)
            },
            duration
        )
    
    def _print_test_summary(self, metrics, comparison_stats, duration):
        """Wyświetla podsumowanie testu"""
        print(f"\n📈 PODSUMOWANIE TESTU:")
        print(f"   - Czas trwania: {duration:.1f}s")
        
        if metrics:
            print(f"\n   MSE:")
            for model_name, model_metrics in metrics.items():
                print(f"     - {model_name.capitalize():15s}: {model_metrics.get('MSE', 'N/A'):.6f}")
            
            print(f"\n   MAE:")
            for model_name, model_metrics in metrics.items():
                print(f"     - {model_name.capitalize():15s}: {model_metrics.get('MAE', 'N/A'):.6f}")
        
        if 'best_model' in comparison_stats:
            print(f"\n   NAJLEPSZY MODEL: {comparison_stats['best_model'].capitalize()} "
                  f"(MSE={comparison_stats.get('best_mse', 'N/A'):.6f})")
        
        if 'better_counts' in comparison_stats:
            print(f"\n   LICZBA PUNKTÓW Z NAJLEPSZYM WYNIKIEM:")
            for model_name, count in comparison_stats['better_counts'].items():
                print(f"     - {model_name.capitalize():15s}: {count}")
            if comparison_stats.get('equal_count', 0) > 0:
                print(f"     - Równe wyniki: {comparison_stats['equal_count']}")
        
        if comparison_stats.get('wilcoxon_pvalue') is not None:
            print(f"   - Wilcoxon p-value: {comparison_stats['wilcoxon_pvalue']:.4f}")
    
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
                                  2500, 3000, 3500, 5000, 7500, 10000,15000,20000]
        
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
    
        # DEBUG: Sprawdź co jest w wynikach
        print(f"\n🔍 DEBUG: Analiza wyników do wykresów")
        print(f"Liczba wyników: {len(results)}")
    
        for i, result in enumerate(results):
            if result['success']:
                print(f"\nWynik {i+1}: n_obs={result.get('n_observations', 'brak')}")
                if 'metrics' in result:
                    for model_name, metrics in result['metrics'].items():
                        print(f"  {model_name}: {list(metrics.keys())}")
    
        # Przygotuj dane - ZBIERZ WSZYSTKIE METRYKI
        data = []
        for result in results:
            if not result['success']:
                continue
        
            n_obs = result.get('n_observations', 0)
            row = {'n_observations': n_obs}
        
            # Dodaj WSZYSTKIE metryki dla każdego modelu
            if 'metrics' in result:
                for model_name, metrics in result['metrics'].items():
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            row[f"{model_name}_{metric_name.lower()}"] = float(value)
                        elif isinstance(value, np.ndarray):
                            # Pomijaj tablice
                            pass
        
            data.append(row)
    
        if not data:
            print("⚠️ Brak danych do wykreślenia")
            return
    
        df = pd.DataFrame(data)
        df = df.sort_values('n_observations')
    
        # DEBUG: Pokaż dostępne kolumny
        print(f"\n📊 Dostępne kolumny w danych:")
        print(df.columns.tolist())
    
        # Utwórz wykresy 2x2
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Wpływ liczby obserwacji na jakość predykcji', 
                    fontsize=16, fontweight='bold')
    
        model_colors = {'bayesian': 'b', 'dirichlet': 'r', 
                       'gaussian': 'g', 'spatial': 'm'}
    
        # 1. MSE vs liczba obserwacji
        ax = axes[0, 0]
        mse_plotted = False
        mse_models = []
    
        for model_name in df.columns:
            if model_name.endswith('_mse'):
                model = model_name.replace('_mse', '')
                if model in model_colors:
                    # Sprawdź czy są dane (nie wszystkie NaN)
                    if not df[model_name].isna().all():
                        ax.plot(df['n_observations'], df[model_name], 
                               color=model_colors[model], marker='o', 
                               label=model.capitalize(), linewidth=2)
                        mse_plotted = True
                        mse_models.append(model)
    
        if mse_plotted:
            ax.set_xlabel('Liczba obserwacji')
            ax.set_ylabel('MSE')
            ax.set_title(f'MSE vs liczba obserwacji\n(modele: {", ".join(mse_models)})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_xscale('log')
        else:
            ax.text(0.5, 0.5, 'Brak danych MSE', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('MSE - brak danych')
    
        # 2. Różnica MSE między Bayesian a Dirichlet (jeśli oba są dostępne)
        ax = axes[0, 1]
        if 'bayesian_mse' in df.columns and 'dirichlet_mse' in df.columns:
            # Sprawdź czy są prawidłowe dane
            if not df['bayesian_mse'].isna().all() and not df['dirichlet_mse'].isna().all():
                mse_diff = df['bayesian_mse'] - df['dirichlet_mse']
                ax.plot(df['n_observations'], mse_diff, 'g-', marker='^', linewidth=2)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax.set_xlabel('Liczba obserwacji')
                ax.set_ylabel('Różnica MSE (Bayesian - Dirichlet)')
                ax.set_title('Różnica MSE: Bayesian vs Dirichlet')
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
            
                # Dodaj informację o trendzie
                if len(mse_diff) > 1:
                    trend = "malejący" if mse_diff.iloc[-1] < mse_diff.iloc[0] else "rosnący"
                    ax.text(0.05, 0.95, f'Trend: {trend}', 
                           transform=ax.transAxes, fontsize=10,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'Brak kompletnych danych\nBayesian/Dirichlet MSE', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Różnica MSE - niekompletne dane')
        else:
            ax.text(0.5, 0.5, 'Brak danych dla porównania\nBayesian/Dirichlet', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Różnica MSE - brak danych')
    
        # 3. MAE vs liczba obserwacji
        ax = axes[1, 0]
        mae_plotted = False
        mae_models = []
    
        for model_name in df.columns:
            if model_name.endswith('_mae'):
                model = model_name.replace('_mae', '')
                if model in model_colors:
                    # Sprawdź czy są dane
                    if model_name in df.columns and not df[model_name].isna().all():
                        ax.plot(df['n_observations'], df[model_name], 
                               color=model_colors[model], marker='s', 
                               label=model.capitalize(), linewidth=2)
                        mae_plotted = True
                        mae_models.append(model)
    
        if mae_plotted:
            ax.set_xlabel('Liczba obserwacji')
            ax.set_ylabel('MAE')
            ax.set_title(f'MAE vs liczba obserwacji\n(modele: {", ".join(mae_models)})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_xscale('log')
        else:
            # Sprawdź czy może są inne metryki do pokazania
            alt_metric = None
            for metric in ['rmse', 'covariance']:
                metric_models = []
                for model_name in df.columns:
                    if model_name.endswith(f'_{metric}'):
                        model = model_name.replace(f'_{metric}', '')
                        if model in model_colors:
                            metric_models.append(model)
            
                if metric_models:
                    alt_metric = metric
                    break
        
            if alt_metric:
                # Rysuj alternatywną metrykę
                alt_plotted = False
                for model_name in df.columns:
                    if model_name.endswith(f'_{alt_metric}'):
                        model = model_name.replace(f'_{alt_metric}', '')
                        if model in model_colors:
                            if not df[model_name].isna().all():
                                label = f"{model.capitalize()} ({alt_metric.upper()})"
                                ax.plot(df['n_observations'], df[model_name], 
                                       color=model_colors[model], marker='d', 
                                       label=label, linewidth=2)
                                alt_plotted = True
            
                if alt_plotted:
                    ax.set_xlabel('Liczba obserwacji')
                    ax.set_ylabel(alt_metric.upper())
                    ax.set_title(f'{alt_metric.upper()} vs liczba obserwacji')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_xscale('log')
                else:
                    ax.text(0.5, 0.5, 'Brak danych MAE', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title('MAE - brak danych')
            else:
                ax.text(0.5, 0.5, 'Brak danych MAE', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('MAE - brak danych')
    
        # 4. Korelacja vs liczba obserwacji
        ax = axes[1, 1]
        corr_plotted = False
        corr_models = []
    
        for model_name in df.columns:
            if model_name.endswith('_correlation'):
                model = model_name.replace('_correlation', '')
                if model in model_colors:
                    # Sprawdź czy są dane
                    if model_name in df.columns and not df[model_name].isna().all():
                        ax.plot(df['n_observations'], df[model_name], 
                               color=model_colors[model], marker='x', 
                               label=model.capitalize(), linewidth=2)
                        corr_plotted = True
                        corr_models.append(model)
    
        if corr_plotted:
            ax.set_xlabel('Liczba obserwacji')
            ax.set_ylabel('Korelacja')
            ax.set_title(f'Korelacja vs liczba obserwacji\n(modele: {", ".join(corr_models)})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
        
            # Dodaj linię na poziomie 0
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        else:
            # Sprawdź czy może jest kowariancja
            cov_plotted = False
            cov_models = []
        
            for model_name in df.columns:
                if model_name.endswith('_covariance'):
                    model = model_name.replace('_covariance', '')
                    if model in model_colors:
                        if model_name in df.columns and not df[model_name].isna().all():
                            ax.plot(df['n_observations'], df[model_name], 
                                   color=model_colors[model], marker='*', 
                                   label=f"{model.capitalize()} (kow.)", linewidth=2)
                            cov_plotted = True
                            cov_models.append(model)
        
            if cov_plotted:
                ax.set_xlabel('Liczba obserwacji')
                ax.set_ylabel('Kowariancja')
                ax.set_title(f'Kowariancja vs liczba obserwacji\n(modele: {", ".join(cov_models)})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
            
                # Dodaj linię na poziomie 0
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Brak danych korelacji/kowariancji', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Korelacja/Kowariancja - brak danych')
    
        plt.tight_layout()
    
        # Zapisz wykres
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"observation_length_impact_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.show()
    
        print(f"\n  ✔ Wykresy zapisane do: {plot_file}")
    
        # Dodatkowo wypisz statystyki
        print(f"\n📈 PODSUMOWANIE DANYCH:")
        for col in df.columns:
            if col != 'n_observations':
                if not df[col].isna().all():
                    non_nan = df[col].dropna()
                    if len(non_nan) > 0:
                        model_metric = col.split('_')
                        if len(model_metric) >= 2:
                            model = model_metric[0]
                            metric = '_'.join(model_metric[1:])
                            print(f"  - {model}.{metric}: "
                                  f"n={len(non_nan)}, "
                                  f"min={non_nan.min():.3e}, "
                                  f"max={non_nan.max():.3e}")
def convert_to_serializable(obj):
        """Konwertuje obiekt na format możliwy do zapisania w JSON"""
        if hasattr(obj, 'dtype'):  # Sprawdź czy to obiekt numpy
            if np.issubdtype(obj.dtype, np.integer):
                return int(obj)
            elif np.issubdtype(obj.dtype, np.floating):
                return float(obj)
            elif np.issubdtype(obj.dtype, np.bool_):
                return bool(obj)
            else:
                return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
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
    csv_path = r"C:\Users\User\Downloads\Global_2020_MarineSpeciesRichness_AquaMaps.csv"
    
    base_params = {
        'cutoff_km': 1000,
        'co_ktory': 100,
        'n_observations': 50000,
        'lengthscale': 5000,
        'variance': 1.0,
        'distance_unit': "km",
        'mcmc_samples': 5000,
        'mcmc_burn': 3000,
        'mcmc_scale': 0.05,
        'mcmc_seed': 42,
        'n_points': 0 
    }
    
    # Określ które modele testować
    program(base_params, n_tests=2, csv_path=csv_path, 
            save=False, xd={"rt":False,"it":True})