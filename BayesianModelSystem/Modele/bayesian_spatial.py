import numpy as np
from scipy import stats, special
from scipy.spatial.distance import cdist, pdist, squareform
import warnings

class BayesianSpatialModel:
    """Bazowa klasa dostosowana do istniejacej struktury danych."""
    
    def __init__(self, space_points, metric_func, observed_indices, 
                 counts=None, N=None, distance_unit='km'):
        """
        space_points: array (n x d) wspolrzednych przestrzennych
        metric_func: funkcja metryki odleglosci
        observed_indices: indeksy obserwowanych punktow
        counts: liczby sukcesow/obserwacji
        N: calkowite liczby prob (dla binomial) lub exposure (dla poisson)
        distance_unit: 'km' lub 'latlon'
        """
        self.space_points = np.asarray(space_points)
        self.metric_func = metric_func
        self.observed_indices = np.asarray(observed_indices, dtype=int)
        self.distance_unit = distance_unit
        
        # Dane obserwowane
        self.counts = counts
        self.N = N
        
        # Oblicz wszystkie odleglosci
        self._compute_all_distances()
        
        print(f"? [BAYES-SPATIAL] Inicjalizacja modelu")
        print(f"   - Liczba punktow: {self.space_points.shape[0]}")
        print(f"   - Obserwowane: {len(self.observed_indices)}")
        print(f"   - Unit: {distance_unit}")
    
    def _compute_all_distances(self):
        """Oblicz macierz odleglosci dla wszystkich punktow."""
        n_points = len(self.space_points)
        self.distance_matrix = np.zeros((n_points, n_points))
        
        if self.distance_unit == 'km':
            # Euclidean distance for km
            self.distance_matrix = cdist(self.space_points, self.space_points, metric='euclidean')
        elif self.distance_unit == 'latlon':
            # Use provided metric function for lat/lon
            for i in range(n_points):
                for j in range(n_points):
                    self.distance_matrix[i, j] = self.metric_func(
                        self.space_points[i], 
                        self.space_points[j]
                    )
        else:
            # Use provided metric
            self.distance_matrix = cdist(
                self.space_points, 
                self.space_points, 
                metric=self.metric_func
            )
    
    def _compute_spatial_weights(self, phi, weight_type='exponential'):
        """
        Oblicz wagi przestrzenne.
        
        phi: parameter zasiegu przestrzennego (lengthscale)
        weight_type: 'exponential', 'gaussian'
        """
        D = self.distance_matrix
        
        if weight_type == 'exponential':
            W = np.exp(-D / phi)
        elif weight_type == 'gaussian':
            W = np.exp(-(D**2) / (2 * phi**2))
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
            
        # Upewnij sie, ze macierz jest symetryczna
        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 1.0)
        
        return W


class GaussianSpatialModelConjugate(BayesianSpatialModel):
    """
    Gaussowski model przestrzenny z priorami sprzezonymi, implementujący Proces Gaussowski (GP).
    Dla danych ciaglych.

    Algorytm działania:
    1. Inicjalizacja:
       - Model przyjmuje współrzędne punktów, obserwowane indeksy oraz wartości w tych punktach (`counts`).
       - Obliczana jest pełna macierz odległości `D` między wszystkimi punktami.

    2. Metoda `fit`:
       - Cel: Estymacja parametrów `posterior` dla średniej (`mu`) i wariancji (`sigma^2`)
         latentnego pola Gaussowskiego.
       - Kroki:
         a) Optymalizacja `phi` (lengthscale): Jeśli włączone, model szuka `phi`, które
            najlepiej opisuje korelację przestrzenną, maksymalizując profilowaną wiarygodność.
         b) Obliczenie macierzy kowariancji: Na podstawie `phi` i odległości między
            obserwowanymi punktami tworzona jest macierz kowariancji (wag) `W_obs`.
         c) Transformacja danych: Obserwowane dane `y_obs` są transformowane z użyciem
            rozkładu Cholesky'ego macierzy `W_obs`, co "wybiela" dane, usuwając korelacje.
         d) Wyznaczenie parametrów `posterior`: Na podstawie transformowanych danych i
            zdefiniowanych priorów (Normal dla średniej, Inverse-Gamma dla wariancji),
            model oblicza analitycznie parametry rozkładów `posterior`.

    3. Metoda `predict`:
       - Cel: Predykcja wartości pola Gaussowskiego na całej siatce.
       - Kroki:
         a) Próbkowanie z `posterior`: Generowane są próbki `mu` i `sigma^2` z ich
            rozkładów `posterior` obliczonych w metodzie `fit`.
         b) Predykcja dla nieobserwowanych punktów:
            - Dla każdej próbki `(mu, sigma^2)`, obliczana jest średnia warunkowa i
              kowariancja warunkowa dla nieobserwowanych punktów, bazując na
              obserwowanych danych i strukturze korelacji (standardowa predykcja GP).
            - Generowane są próbki z wynikowego wielowymiarowego rozkładu normalnego.
         c) Predykcja dla obserwowanych punktów: Wartości są po prostu kopiowane z
            danych wejściowych (`y_obs`).
         d) Agregacja wyników: Predykcje ze wszystkich próbek są uśredniane, aby
            uzyskać finalną wartość oczekiwaną i przedziały ufności.
    """
    
    def __init__(self, space_points, metric_func, observed_indices, 
                 counts=None, N=None, distance_unit='km',
                 mu_prior=0.0, sigma_prior=1.0):
        """
        mu_prior: prior dla sredniej
        sigma_prior: prior dla odchylenia standardowego
        """
        super().__init__(space_points, metric_func, observed_indices, 
                        counts, N, distance_unit)
        
        # Dane musza byc ustawione jako counts (dla spojnosci)
        if counts is None:
            raise ValueError("'counts' must be provided for Gaussian model")
        
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        
        # Dla Gaussowskiego modelu, counts to obserwacje y
        self.y = self.counts  # alias dla czytelnosci
        
        print(f"   - Model: Gaussian (conjugate)")
        print(f"   - Prior: m ~ N({mu_prior}, {sigma_prior})")
    
    def fit(self, phi=4, optimize_phi=True):
        """
        Dopasuj model.
        
        phi: parameter zasiegu przestrzennego
        optimize_phi: czy optymalizowac phi automatycznie
        """
        print(f"\n? [GAUSSIAN-CONJUGATE] Dopasowywanie modelu...")
        
        # Obserwowane dane
        y_obs = self.y
        n_obs = len(y_obs)
        
        # Znajdz optymalny phi jesli potrzebne
        if phi is None and optimize_phi:
            phi = self._optimize_phi(y_obs)
        elif phi is None:
            # Uzyj heurystyki: 1/3 maksymalnej odleglosci
            phi = np.max(self.distance_matrix) / 3
            print(f"   - Phi (heurystyka): {phi:.0f}")
        
        # Oblicz wagi dla obserwowanych punktow
        D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
        W_obs = np.exp(-D_obs / phi)
        
        # Dodaj jitter dla stabilnosci
        W_obs = W_obs + np.eye(n_obs) * 1e-8
        
        # Transformacja Cholesky'ego
        try:
            L = np.linalg.cholesky(W_obs)
        except np.linalg.LinAlgError:
            W_obs = W_obs + np.eye(n_obs) * 1e-6
            L = np.linalg.cholesky(W_obs)
        
        # Transformacja danych
        y_transformed = np.linalg.solve(L, y_obs)
        
        # Parametry posterior dla wariancji (Inverse-Gamma)
        # Prior: s2 ~ InvGamma(a0, b0)
        alpha0 = 0.001  # weak prior
        beta0 = 0.001   # weak prior
        
        y_bar = np.mean(y_transformed)
        SS = np.sum((y_transformed - y_bar)**2)
        
        alpha_n = alpha0 + n_obs / 2
        beta_n = beta0 + 0.5 * SS
        
        # Posterior dla sredniej (Normal)
        # Prior: m ~ N(m0, t0*s2)
        mu0 = self.mu_prior
        sigma0 = self.sigma_prior
        kappa0 = 1.0 / (sigma0**2)
        
        # Precyzje (odwrotnosci wariancji)
        sigma2_est = beta_n / (alpha_n - 1) if alpha_n > 1 else beta_n / alpha0
        kappa_n = kappa0 + n_obs / sigma2_est
        mu_n = (kappa0 * mu0 + (n_obs / sigma2_est) * y_bar) / kappa_n
        
        # Zapisz parametry
        self.posterior_params = {
            'mu_n': mu_n,
            'sigma_n': 1.0 / np.sqrt(kappa_n),
            'alpha_n': alpha_n,
            'beta_n': beta_n,
            'phi': phi,
            'W_obs': W_obs,
            'y_obs': y_obs
        }
        
        print(f"? Model dopasowany:")
        print(f"   - Phi: {phi:.0f}")
        print(f"   - Posterior m: N({mu_n:.2f}, {1.0/np.sqrt(kappa_n):.2f})")
        print(f"   - Posterior s2: IG({alpha_n:.3f}, {beta_n:.3f})")
        
        return self
    
    def _optimize_phi(self, y_obs):
        """Ulepszona optymalizacja phi dla modelu Gaussa oparta na profilowanej wiarygodności."""
        print("   ?? Optymalizacja phi dla modelu Gaussa...")

        max_dist = np.max(self.distance_matrix)
        if max_dist == 0: max_dist = 1000 # fallback

        # Dynamiczna siatka logarytmiczna dla phi
        phi_values = np.logspace(
            np.log10(max(1.0, 0.01 * max_dist)),
            np.log10(max(10.0, 2.0 * max_dist)),
            15
        )
        print(f"   - Testowane phi: {np.round(phi_values, 0)}")

        scores = []
        n_obs = len(y_obs)

        for phi in phi_values:
            try:
                # Oblicz macierz kowariancji (wag)
                D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
                W_obs = np.exp(-D_obs / phi) + np.eye(n_obs) * 1e-6 # Jitter dla stabilności

                # Log-determinant
                sign, logdet = np.linalg.slogdet(W_obs)
                if sign <= 0: # Macierz nie jest dodatnio określona
                    scores.append(np.inf)
                    continue
                
                # Oblicz residua
                y_mean = np.mean(y_obs)
                residuals = y_obs - y_mean
                
                # Rozwiąż układ równań zamiast odwracać macierz
                # To oblicza W_inv @ residuals
                solved_res = np.linalg.solve(W_obs, residuals)
                
                # Oblicz RSS (Residual Sum of Squares)
                rss = np.dot(residuals, solved_res)

                if rss <= 0:
                    scores.append(np.inf)
                    continue

                # Profilowana log-wiarygodność (do minimalizacji)
                score = n_obs * np.log(rss) + logdet
                
                if not np.isfinite(score):
                    scores.append(np.inf)
                else:
                    scores.append(score)
            except np.linalg.LinAlgError:
                scores.append(np.inf)
            except Exception:
                scores.append(np.inf)
        
        if np.all(np.isinf(scores)):
             best_phi = max_dist / 3 # Heurystyka
             warnings.warn(f"Wszystkie wartości phi dały nieprawidłowy wynik. Używam heurystyki phi={best_phi:.0f}")
        else:
            best_idx = np.nanargmin(scores)
            best_phi = phi_values[best_idx]
        
        print(f"   - Najlepsze phi: {best_phi:.0f}")
        return best_phi
    
    def predict(self, num_samples=1000, return_samples=False):
        """
        Predykcja dla wszystkich punktow z poprawnym warunkowaniem GP.
        """
        if not hasattr(self, 'posterior_params'):
            raise ValueError("Model must be fitted first")

        n_total = len(self.space_points)
        n_obs = len(self.observed_indices)
        
        print(f"\n? [GAUSSIAN-PREDICTION-GP] Generowanie predykcji...")

        # --- Pobierz parametry z modelu ---
        mu_samples = np.random.normal(loc=self.posterior_params['mu_n'], scale=self.posterior_params['sigma_n'], size=num_samples)
        sigma2_samples = stats.invgamma.rvs(a=self.posterior_params['alpha_n'], scale=self.posterior_params['beta_n'], size=num_samples)
        phi = self.posterior_params['phi']
        y_obs = self.posterior_params['y_obs']

        # --- Podzial na indeksy ---
        unobs_indices = np.setdiff1d(np.arange(n_total), self.observed_indices, assume_unique=True)
        n_unobs = len(unobs_indices)

        if n_unobs == 0: # Jeśli wszystko obserwowane
            mean_pred = np.mean(np.tile(y_obs, (num_samples, 1)), axis=0)
            if return_samples:
                 return y_obs, (y_obs, y_obs), np.tile(y_obs, (num_samples, 1))
            else:
                 return y_obs, (y_obs, y_obs)

        # --- Oblicz macierze kowariancji ---
        W_obs = self.posterior_params['W_obs']
        D_obs_unobs = self.distance_matrix[np.ix_(self.observed_indices, unobs_indices)]
        W_obs_unobs = np.exp(-D_obs_unobs / phi)
        D_unobs = self.distance_matrix[np.ix_(unobs_indices, unobs_indices)]
        W_unobs = np.exp(-D_unobs / phi)

        # --- Pre-komputacja dla wydajności ---
        try:
            L_obs = np.linalg.cholesky(W_obs)
            # alpha = (W_obs)^-1 * y_obs
            alpha = np.linalg.solve(L_obs.T, np.linalg.solve(L_obs, y_obs))
            # v = (W_obs)^-1 * 1
            v = np.linalg.solve(L_obs.T, np.linalg.solve(L_obs, np.ones(n_obs)))
            
            # Pre-oblicz części średniej warunkowej
            term1 = np.dot(W_obs_unobs.T, alpha)
            term2 = np.dot(W_obs_unobs.T, v)
        except np.linalg.LinAlgError:
            warnings.warn("Macierz kowariancji obserwowanych niestabilna. Zwracam jednorodną predykcję.")
            mean_pred = np.full(n_total, np.mean(y_obs))
            return mean_pred, (mean_pred, mean_pred), None

        # --- Kowariancja warunkowa (bez sigma^2) ---
        W_obs_inv_W_ou = np.linalg.solve(W_obs, W_obs_unobs)
        Cov_cond_norm = W_unobs - np.dot(W_obs_unobs.T, W_obs_inv_W_ou)
        Cov_cond_norm = 0.5 * (Cov_cond_norm + Cov_cond_norm.T)
        Cov_cond_norm += np.eye(n_unobs) * 1e-6

        # --- Generuj próbki ---
        predictions = np.zeros((num_samples, n_total))
        predictions[:, self.observed_indices] = np.tile(y_obs, (num_samples, 1))

        for i in range(num_samples):
            mu_i = mu_samples[i]
            
            # Średnia warunkowa
            mu_cond = mu_i + (term1 - mu_i * term2)

            # Kowariancja warunkowa
            cov_cond = sigma2_samples[i] * Cov_cond_norm
            
            try:
                pred_unobs = np.random.multivariate_normal(mu_cond, cov_cond, check_valid='warn', tol=1e-8)
                predictions[i, unobs_indices] = pred_unobs
            except (ValueError, np.linalg.LinAlgError) as e:
                warnings.warn(f"Błąd próbkowania (próbka {i}): {e}. Używam średniej warunkowej.")
                predictions[i, unobs_indices] = mu_cond
        
        # Srednie i przedzialy ufnosci
        mean_pred = np.mean(predictions, axis=0)
        lower_pred = np.percentile(predictions, 2.5, axis=0)
        upper_pred = np.percentile(predictions, 97.5, axis=0)
        
        print(f"? Predykcje wygenerowane:")
        print(f"   - Srednia predykcji: {np.mean(mean_pred):.3f}")
        print(f"   - Zakres: [{np.min(mean_pred):.3f}, {np.max(mean_pred):.3f}]")
        
        if return_samples:
            return mean_pred, (lower_pred, upper_pred), predictions
        else:
            return mean_pred, (lower_pred, upper_pred)


class SpatialPoissonConjugate(BayesianSpatialModel):
    """
    Przestrzenny model Poissona z priorami Gamma (sprzężonymi), uwzględniający
    wygładzanie przestrzenne. Model ten jest przeznaczony dla danych w postaci zliczeń.

    Algorytm działania:
    1. Inicjalizacja:
       - Model przyjmuje współrzędne punktów, obserwowane indeksy, zliczenia (`counts`)
         oraz opcjonalnie wektor ekspozycji (`N`).
       - Przechowuje parametry prioru Gamma (`alpha_prior`, `beta_prior`).

    2. Metoda `fit`:
       - Cel: Estymacja parametrów `posterior` rozkładu Gamma dla intensywności (`lambda`)
         w każdym z obserwowanych punktów.
       - Kroki:
         a) Optymalizacja `phi`: Jeśli włączone, model szuka optymalnego `phi` (lengthscale),
            które maksymalizuje log-wiarygodność Poissona.
         b) Obliczenie wag: Tworzona jest macierz wag przestrzennych `W_obs` dla
            obserwowanych punktów na podstawie `phi`.
         c) Obliczenie parametrów `posterior` (Gamma): Dla każdego obserwowanego punktu `i`,
            parametry `alpha_n` i `beta_n` rozkładu `posterior` są obliczane przez połączenie:
            - Priora (Gamma(`alpha_prior`, `beta_prior`)).
            - Danych z punktu `i` (zliczenia `y_counts[i]` i ekspozycja `y_exposure[i]`).
            - Informacji z sąsiedztwa: tworzone są "pseudo-obserwacje" na podstawie
              ważonej przestrzennie średniej intensywności z sąsiednich punktów.

    3. Metoda `predict`:
       - Cel: Predykcja intensywności `lambda` na całej siatce.
       - Kroki:
         a) Próbkowanie z `posterior`: Dla każdego obserwowanego punktu generowane są
            próbki intensywności `lambda` z jego rozkładu `posterior` Gamma.
         b) Interpolacja przestrzenna: Dla każdego punktu `j` na całej siatce (obserwowanego
            i nieobserwowanego), wartość `lambda` jest estymowana jako średnia ważona
            próbek `lambda` z obserwowanych lokalizacji. Wagi zależą od odległości
            przestrzennej do punktu `j`.
         c) Agregacja wyników: Predykcje ze wszystkich próbek są uśredniane, aby
            uzyskać finalną oczekiwaną intensywność `lambda` i przedziały ufności.
    """
    
    def __init__(self, space_points, metric_func, observed_indices,
                 counts, N=None, distance_unit='km',
                 alpha_prior=0.5, beta_prior=0.5, smoothing_strength=1.0):
        """
        counts: obserwowane liczniki
        N: exposure/czas obserwacji (jesli None, zakladamy = 1)
        alpha_prior, beta_prior: parametry priora Gamma
        """
        super().__init__(space_points, metric_func, observed_indices,
                        counts, N, distance_unit)
        
        if counts is None:
            raise ValueError("'counts' must be provided for Poisson model")
        
        if N is None:
            self.exposure = np.ones(len(space_points))
        else:
            self.exposure = N

        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.smoothing_strength = smoothing_strength
        
        # Obserwowane dane
        self.y_counts = self.counts[self.observed_indices]
        self.y_exposure = self.exposure[self.observed_indices]
        
        print(f"   - Model: Poisson (conjugate)")
        print(f"   - Prior: l ~ Gamma({alpha_prior}, {beta_prior})")
    
    def fit(self, phi=4, optimize_phi=True):
        """
        Dopasuj model Poissona.
        """
        print(f"\n? [POISSON-CONJUGATE] Dopasowywanie modelu...")
        
        n_obs = len(self.observed_indices)
        
        # Znajdz optymalny phi
        if phi is None and optimize_phi:
            phi = self._optimize_phi_poisson()
        elif phi is None:
            phi = np.max(self.distance_matrix) / 3
            print(f"   - Phi (heurystyka): {phi:.0f}")
        
        # Oblicz wagi dla obserwowanych punktow
        D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
        W_obs = np.exp(-D_obs / phi)
        
        # Parametry posterior Gamma dla kazdego punktu
        alpha_n = np.zeros(n_obs)
        beta_n = np.zeros(n_obs)
        
        for i in range(n_obs):
            # Oblicz przestrzenne priors na podstawie sąsiadów
            weights = W_obs[i, :]
            
            # Lokalna, ważona przestrzennie intensywność (lambda)
            neighbor_counts = np.sum(weights * self.y_counts)
            neighbor_exposure = np.sum(weights * self.y_exposure)

            if neighbor_exposure > 1e-9:
                neighbor_rate = neighbor_counts / neighbor_exposure
            else:
                # Fallback do średniej globalnej
                if np.sum(self.y_exposure) > 1e-9:
                    neighbor_rate = np.sum(self.y_counts) / np.sum(self.y_exposure)
                else:
                    neighbor_rate = 1.0

            # Zdefiniuj pseudo-obserwacje z priora przestrzennego
            spatial_pseudo_counts = self.smoothing_strength * neighbor_rate
            spatial_pseudo_exposure = self.smoothing_strength

            alpha_n[i] = self.alpha_prior + self.y_counts[i] + spatial_pseudo_counts
            beta_n[i] = self.beta_prior + self.y_exposure[i] + spatial_pseudo_exposure
        
        # Zapisz parametry
        self.posterior_params = {
            'alpha_n': alpha_n,
            'beta_n': beta_n,
            'phi': phi,
            'W_obs': W_obs,
            'y_counts': self.y_counts,
            'y_exposure': self.y_exposure
        }
        
        print(f"? Model dopasowany:")
        print(f"   - Phi: {phi:.0f}")
        print(f"   - a posterior: [{np.min(alpha_n):.2f}, {np.max(alpha_n):.2f}]")
        print(f"   - b posterior: [{np.min(beta_n):.2f}, {np.max(beta_n):.2f}]")
        
        return self
    
    def _optimize_phi_poisson(self):
        """Ulepszona optymalizacja phi dla modelu Poissona z dynamiczną siatką."""
        print("   ?? Optymalizacja phi dla modelu Poissona...")

        max_dist = np.max(self.distance_matrix)
        if max_dist == 0: max_dist = 1000 # fallback

        phi_values = np.logspace(
            np.log10(max(1.0, 0.01 * max_dist)),
            np.log10(max(10.0, 2.0 * max_dist)),
            15
        )
        print(f"   - Testowane phi: {np.round(phi_values, 0)}")

        scores = []

        for phi in phi_values:
            try:
                D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
                W_obs = np.exp(-D_obs / phi)
                
                log_likelihood = 0
                for i in range(len(self.y_counts)):
                    # Estymuj lambda uwzgledniajac sasiadow
                    weights = W_obs[i, :]
                    sum_weights = np.sum(weights)
                    
                    if sum_weights <= 1e-9:
                        lambda_i = np.mean(self.y_counts) # Fallback
                    else:
                        lambda_i = np.sum(weights * self.y_counts) / sum_weights
                    
                    lambda_i = max(lambda_i, 1e-9) # Unikaj log(0)

                    # Poissono-wa log-likelihood
                    log_likelihood += (self.y_counts[i] * np.log(lambda_i) - 
                                      lambda_i * self.y_exposure[i])

                if not np.isfinite(log_likelihood):
                    scores.append(np.inf)
                else:
                    scores.append(-log_likelihood)
            except Exception:
                scores.append(np.inf)
        
        if np.all(np.isinf(scores)):
             best_phi = max_dist / 3 # Heurystyka
             warnings.warn(f"Wszystkie wartości phi dały nieprawidłowy wynik. Używam heurystyki phi={best_phi:.0f}")
        else:
            best_idx = np.nanargmin(scores)
            best_phi = phi_values[best_idx]
        
        print(f"   - Najlepsze phi: {best_phi:.0f}")
        return best_phi
    
    def predict(self, num_samples=1000, return_samples=False):
        """
        Predykcja intensywnosci l dla wszystkich punktow.
        """
        if not hasattr(self, 'posterior_params'):
            raise ValueError("Model must be fitted first")
        
        n_total = len(self.space_points)
        n_obs = len(self.observed_indices)
        
        print(f"\n? [POISSON-PREDICTION] Generowanie predykcji...")
        
        # 1. Probkuj l dla obserwowanych punktow z posteriora Gamma
        lambda_samples_obs = np.zeros((num_samples, n_obs))
        
        for i in range(n_obs):
            lambda_samples_obs[:, i] = stats.gamma.rvs(
                a=self.posterior_params['alpha_n'][i],
                scale=1.0 / self.posterior_params['beta_n'][i],
                size=num_samples
            )
        
        # 2. Interpoluj dla nieobserwowanych punktow
        phi = self.posterior_params['phi']
        
        # Dla kazdego punktu (obserwowanego i nie)
        predictions = np.zeros((num_samples, n_total))
        
        for j in range(n_total):
            # Oblicz odleglosci do obserwowanych punktow
            distances = self.distance_matrix[j, self.observed_indices]
            weights = np.exp(-distances / phi)
            weights = weights / np.sum(weights)  # normalizuj
            
            # Srednia wazona
            predictions[:, j] = np.dot(lambda_samples_obs, weights)
        
        # Srednie i przedzialy ufnosci
        mean_pred = np.mean(predictions, axis=0)
        lower_pred = np.percentile(predictions, 2.5, axis=0)
        upper_pred = np.percentile(predictions, 97.5, axis=0)
        
        print(f"? Predykcje wygenerowane:")
        print(f"   - Srednia l: {np.mean(mean_pred):.3f}")
        print(f"   - Zakres l: [{np.min(mean_pred):.3f}, {np.max(mean_pred):.3f}]")
        
        if return_samples:
            return mean_pred, (lower_pred, upper_pred), predictions
        else:
            return mean_pred, (lower_pred, upper_pred)
    
    def predict_counts(self, num_samples=1000):
        """
        Predykcja liczby zdarzen (a nie tylko intensywnosci).
        """
        lambda_pred, ci, lambda_samples = self.predict(num_samples, return_samples=True)
        
        # Generuj liczby z Poissona
        count_samples = np.random.poisson(lambda_samples * self.exposure)
        
        mean_counts = np.mean(count_samples, axis=0)
        lower_counts = np.percentile(count_samples, 2.5, axis=0)
        upper_counts = np.percentile(count_samples, 97.5, axis=0)
        
        return mean_counts, (lower_counts, upper_counts), count_samples


class SpatialBinomialConjugate(BayesianSpatialModel):
    """
    Przestrzenny model dwumianowy z priorami Beta (sprzężonymi), który uwzględnia
    wygładzanie przestrzenne. Model jest przeznaczony dla danych o liczbie sukcesów i prób.

    Algorytm działania:
    1. Inicjalizacja:
       - Model przyjmuje współrzędne punktów, obserwowane indeksy, liczbę sukcesów (`counts`)
         oraz liczbę prób (`N`).
       - Przechowuje parametry prioru Beta (`alpha_prior`, `beta_prior`).

    2. Metoda `fit`:
       - Cel: Estymacja parametrów `posterior` rozkładu Beta dla prawdopodobieństwa
         sukcesu (`p`) w każdym z obserwowanych punktów.
       - Kroki:
         a) Optymalizacja `phi`: Jeśli włączone, model szuka optymalnego `phi` (lengthscale),
            które maksymalizuje log-wiarygodność dwumianową.
         b) Obliczenie wag: Tworzona jest macierz wag przestrzennych `W_obs` dla
            obserwowanych punktów na podstawie `phi`.
         c) Obliczenie parametrów `posterior` (Beta): Dla każdego obserwowanego punktu `i`,
            parametry `alpha_n` i `beta_n` rozkładu `posterior` są obliczane przez połączenie:
            - Priora (Beta(`alpha_prior`, `beta_prior`)).
            - Danych z punktu `i` (liczba sukcesów i porażek).
            - Informacji z sąsiedztwa: tworzone są "pseudo-obserwacje" na podstawie
              ważonego przestrzennie prawdopodobieństwa sukcesu z sąsiednich punktów.

    3. Metoda `predict`:
       - Cel: Predykcja prawdopodobieństwa sukcesu `p` na całej siatce.
       - Kroki:
         a) Próbkowanie z `posterior`: Dla każdego obserwowanego punktu generowane są
            próbki prawdopodobieństwa `p` z jego rozkładu `posterior` Beta.
         b) Interpolacja przestrzenna: Dla każdego punktu `j` na całej siatce, wartość `p`
            jest estymowana jako średnia ważona próbek `p` z obserwowanych lokalizacji.
            Wagi zależą od odległości przestrzennej do punktu `j`.
         c) Agregacja wyników: Predykcje ze wszystkich próbek są uśredniane, aby
            uzyskać finalne oczekiwane prawdopodobieństwo `p` i przedziały ufności.
    """
    
    def __init__(self, space_points, metric_func, observed_indices,
                 counts, N, distance_unit='km',
                 alpha_prior=0.5, beta_prior=0.5, smoothing_strength=0.1):
        """
        counts: liczba sukcesow
        N: liczba prob
        alpha_prior, beta_prior: parametry priora Beta
        """
        super().__init__(space_points, metric_func, observed_indices,
                        counts, N, distance_unit)
        
        if counts is None or N is None:
            raise ValueError("Both 'counts' and 'N' must be provided for Binomial model")
        
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.smoothing_strength = smoothing_strength
        
        # Obserwowane dane
        self.y_success = self.counts[self.observed_indices]
        self.y_trials = self.N[self.observed_indices]
        self.y_failure = self.y_trials - self.y_success
        
        print(f"   - Model: Binomial (conjugate)")
        print(f"   - Prior: p ~ Beta({alpha_prior}, {beta_prior})")
    
    def fit(self, phi=4, optimize_phi=True):
        """
        Dopasuj model dwumianowy.
        """
        print(f"\n? [BINOMIAL-CONJUGATE] Dopasowywanie modelu...")
        
        n_obs = len(self.observed_indices)
        
        # Znajdz optymalny phi
        if phi is None and optimize_phi:
            phi = self._optimize_phi_binomial()
        elif phi is None:
            phi = np.max(self.distance_matrix) / 3
            print(f"   - Phi (heurystyka): {phi:.0f}")
        
        # Oblicz wagi dla obserwowanych punktow
        D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
        W_obs = np.exp(-D_obs / phi)
        
        # Parametry posterior Beta dla kazdego punktu
        alpha_n = np.zeros(n_obs)
        beta_n = np.zeros(n_obs)
        
        for i in range(n_obs):
            # Oblicz lokalne prawdopodobienstwo p_hat_i na podstawie sasiadow
            weights = W_obs[i, :]
            
            # Suma wazona sukcesow i prob
            weighted_success_sum = np.sum(weights * self.y_success)
            weighted_trials_sum = np.sum(weights * self.y_trials)
            
            if weighted_trials_sum > 0:
                p_hat_i = weighted_success_sum / weighted_trials_sum
            else:
                # Fallback: uzyj sredniej globalnej, jesli brak sasiadow
                p_hat_i = np.mean(self.y_success) / np.mean(self.y_trials) if np.mean(self.y_trials) > 0 else 0.5

            p_hat_i = np.clip(p_hat_i, 1e-9, 1 - 1e-9)
            
            # Dodaj pseudo-obserwacje z priora przestrzennego
            spatial_alpha = self.smoothing_strength * p_hat_i
            spatial_beta = self.smoothing_strength * (1 - p_hat_i)
            
            alpha_n[i] = self.alpha_prior + self.y_success[i] + spatial_alpha
            beta_n[i] = self.beta_prior + self.y_failure[i] + spatial_beta
        
        # Zapisz parametry
        self.posterior_params = {
            'alpha_n': alpha_n,
            'beta_n': beta_n,
            'phi': phi,
            'W_obs': W_obs,
            'y_success': self.y_success,
            'y_trials': self.y_trials
        }
        
        print(f"? Model dopasowany:")
        print(f"   - Phi: {phi:.0f}")
        print(f"   - a posterior: [{np.min(alpha_n):.2f}, {np.max(alpha_n):.2f}]")
        print(f"   - b posterior: [{np.min(beta_n):.2f}, {np.max(beta_n):.2f}]")
        
        return self
    
    def _optimize_phi_binomial(self):
        """Optymalizacja phi dla modelu dwumianowego."""
        print("   ?? Optymalizacja phi...")

        # Rozszerz zakres poszukiwan i uzyj skali logarytmicznej
        max_dist = np.max(self.distance_matrix)
        if max_dist == 0: max_dist = 1000 # fallback

        # Szukaj od 1% maksymalnej odleglosci do 200%
        phi_values = np.logspace(
            np.log10(max(1.0, 0.01 * max_dist)), 
            np.log10(max(10.0, 2.0 * max_dist)), 
            15
        )
        print(f"   - Testowane phi: {np.round(phi_values, 0)}")
        
        scores = []
        
        for phi in phi_values:
            D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
            W_obs = np.exp(-D_obs / phi)
            
            try:
                log_likelihood = 0
                for i in range(len(self.y_success)):
                    # Estymuj p uwzgledniajac sasiadow
                    weighted_success = np.sum(W_obs[i, :] * self.y_success)
                    weighted_trials = np.sum(W_obs[i, :] * self.y_trials)
                    
                    if weighted_trials <= 1e-9: # Unikaj dzielenia przez zero
                        p_i = 0.5 # Niewiele informacji, uzyj priora 50/50
                    else:
                        p_i = weighted_success / weighted_trials

                    p_i = np.clip(p_i, 1e-9, 1 - 1e-9)
                    
                    # Log-wiarygodnosc dwumianowa (bez stalej)
                    log_likelihood += (self.y_success[i] * np.log(p_i) + 
                                      self.y_failure[i] * np.log(1 - p_i))

                if not np.isfinite(log_likelihood):
                    scores.append(np.inf)
                else:
                    scores.append(-log_likelihood)
            except Exception:
                scores.append(np.inf)
        
        if np.all(np.isinf(scores)):
             best_phi = max_dist / 3 # Heurystyka jesli wszystko inne zawiedzie
             warnings.warn(f"All phi values resulted in an invalid score. Using heuristic phi={best_phi:.0f}")
        else:
            best_idx = np.nanargmin(scores)
            best_phi = phi_values[best_idx]
        
        print(f"   - Najlepsze phi: {best_phi:.0f}")
        return best_phi
    
    def predict(self, num_samples=1000, return_samples=False):
        """
        Predykcja prawdopodobienstwa sukcesu p dla wszystkich punktow.
        """
        if not hasattr(self, 'posterior_params'):
            raise ValueError("Model must be fitted first")
        
        n_total = len(self.space_points)
        n_obs = len(self.observed_indices)
        
        print(f"\n? [BINOMIAL-PREDICTION] Generowanie predykcji...")
        
        # 1. Probkuj p dla obserwowanych punktow z posteriora Beta
        p_samples_obs = np.zeros((num_samples, n_obs))
        
        for i in range(n_obs):
            p_samples_obs[:, i] = stats.beta.rvs(
                a=self.posterior_params['alpha_n'][i],
                b=self.posterior_params['beta_n'][i],
                size=num_samples
            )
        
        # 2. Interpoluj dla nieobserwowanych punktow
        phi = self.posterior_params['phi']
        
        predictions = np.zeros((num_samples, n_total))
        
        for j in range(n_total):
            distances = self.distance_matrix[j, self.observed_indices]
            weights = np.exp(-distances / phi)
            weights = weights / np.sum(weights)
            
            predictions[:, j] = np.dot(p_samples_obs, weights)
        
        # Srednie i przedzialy ufnosci
        mean_pred = np.mean(predictions, axis=0)
        lower_pred = np.percentile(predictions, 2.5, axis=0)
        upper_pred = np.percentile(predictions, 97.5, axis=0)
        
        print(f"? Predykcje wygenerowane:")
        print(f"   - Srednie p: {np.mean(mean_pred):.3f}")
        print(f"   - Zakres p: [{np.min(mean_pred):.3f}, {np.max(mean_pred):.3f}]")
        
        if return_samples:
            return mean_pred, (lower_pred, upper_pred), predictions
        else:
            return mean_pred, (lower_pred, upper_pred)
    
    def predict_counts(self, num_samples=1000):
        """
        Predykcja liczby sukcesow (dla zadanych N).
        """
        p_pred, ci, p_samples = self.predict(num_samples, return_samples=True)
        
        # Generuj liczby sukcesow z Binomial
        success_samples = np.zeros((num_samples, len(self.space_points)))
        
        for i in range(num_samples):
            for j in range(len(self.space_points)):
                # Jesli N jest dostepne dla tego punktu
                if j < len(self.N):
                    n_trials = self.N[j]
                    success_samples[i, j] = np.random.binomial(
                        n=n_trials,
                        p=p_samples[i, j]
                    )
        
        mean_success = np.mean(success_samples, axis=0)
        lower_success = np.percentile(success_samples, 2.5, axis=0)
        upper_success = np.percentile(success_samples, 97.5, axis=0)
        
        return mean_success, (lower_success, upper_success), success_samples


class ReferencePriorSpatialModel(BayesianSpatialModel):
    """
    Priory referencyjne wedlug Berger et al. (2009).
    Mozna uzywac z istniejacymi modelami.
    """
    
    def __init__(self, space_points, metric_func, observed_indices,
                 counts=None, N=None, distance_unit='km'):
        super().__init__(space_points, metric_func, observed_indices,
                        counts, N, distance_unit)
        
        print(f"   - Model: Reference Prior (Berger et al. 2009)")
    
    def compute_jeffreys_prior(self, model_type='binomial'):
        """
        Oblicz prior Jeffreysa dla roznych modeli.
        """
        if model_type == 'binomial':
            # Jeffreys prior for binomial: Beta(1/2, 1/2)
            return 0.5, 0.5
        
        elif model_type == 'poisson':
            # Jeffreys prior for Poisson: Gamma(1/2, 0) 
            # (improper, use Gamma(1/2, epsilon))
            return 0.5, 1e-10
        
        elif model_type == 'gaussian':
            # Jeffreys prior for Gaussian: p(m, s2) ~ 1/s2
            return None  # Rozklad niewlasciwy
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def compute_reference_prior_spatial(self, phi, model_type='binomial'):
        """
        Oblicz prior referencyjny uwzgledniajacy strukture przestrzenna.
        """
        # Oblicz macierz kowariancji
        D_obs = self.distance_matrix[np.ix_(self.observed_indices, self.observed_indices)]
        K = np.exp(-D_obs / phi)
        
        n_obs = len(self.observed_indices)
        
        if model_type == 'gaussian':
            # Oblicz wyznacznik informacji Fishera
            try:
                K_inv = np.linalg.inv(K)
                # Informacja Fishera dla s2 przy zalozeniu przestrzennym
                info_sigma2 = 0.5 * np.trace(np.dot(K_inv, K_inv))
                prior_value = np.sqrt(info_sigma2)
            except:
                prior_value = 1.0
        
        elif model_type == 'poisson':
            # Dla Poissona, informacja Fishera zalezy od intensywnosci
            if self.counts is not None:
                lambda_est = np.mean(self.counts[self.observed_indices])
                info_lambda = n_obs / lambda_est
                prior_value = np.sqrt(info_lambda)
            else:
                prior_value = 1.0
        
        elif model_type == 'binomial':
            # Dla dwumianowego, informacja Fishera dla p
            if self.counts is not None and self.N is not None:
                p_est = np.mean(self.counts[self.observed_indices] / 
                               self.N[self.observed_indices])
                info_p = n_obs / (p_est * (1 - p_est))
                prior_value = np.sqrt(info_p)
            else:
                prior_value = 1.0
        
        return prior_value


# Funkcje pomocnicze dla kompatybilnosci z istniejacym kodem
def prepare_conjugate_model(model_type, space_points, metric_func, observed_indices,
                           counts, N=None, **kwargs):
    """
    Przygotuj odpowiedni model sprzezony.
    
    model_type: 'gaussian', 'poisson', 'binomial'
    """
    if model_type == 'gaussian':
        return GaussianSpatialModelConjugate(
            space_points, metric_func, observed_indices,
            counts=counts, N=N, **kwargs
        )
    
    elif model_type == 'poisson':
        return SpatialPoissonConjugate(
            space_points, metric_func, observed_indices,
            counts=counts, N=N, **kwargs
        )
    
    elif model_type == 'binomial':
        return SpatialBinomialConjugate(
            space_points, metric_func, observed_indices,
            counts=counts, N=N, **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
