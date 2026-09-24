import numpy as np
from scipy.spatial.distance import cdist

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
        
        print(f"▶ [SPATIAL-BASE] Inicjalizacja modelu bazowego")
        print(f"   - Liczba punktow: {self.space_points.shape[0]}")
        print(f"   - Obserwowane: {len(self.observed_indices)}")
    
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
        """
        D = self.distance_matrix
        
        if weight_type == 'exponential':
            W = np.exp(-D / phi)
        elif weight_type == 'gaussian':
            W = np.exp(-(D**2) / (2 * phi**2))
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
            
        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 1.0)
        
        return W
