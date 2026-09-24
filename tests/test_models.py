import pytest
import numpy as np
from BayesianModelSystem.Wczytywanie_danych.metric import haversine
from BayesianModelSystem.Modele import ModelDirichleta, ModelGausowskiSprzezony, ModelAksjomatyczny, ModelAksjomatycznyPreparamed

@pytest.fixture(scope="module")
def small_dataset():
    side = 4
    n_points = side * side
    space_points = [(float(i * 10), float(j * 10)) for i in range(side) for j in range(side)]
    observed_indices = np.array([2, 3, 3, 5, 5, 5, 6, 7, 9, 9, 10, 10, 11, 13, 14, 14, 14, 14, 15, 15])
    return {"space_points": space_points, "n_points": n_points, "observed_indices": observed_indices}

def test_axiomatic_model_smoke(small_dataset):
    space_points, obs_idx = small_dataset["space_points"], small_dataset["observed_indices"]
    model = ModelAksjomatyczny(
        space_points=space_points, 
        metric_func=haversine, 
        observed_indices=obs_idx,
        variance=1.0, 
        distance_unit='km'
    )
    model.przygotuj_apriori()
    assert model.lengthscale > 0
    
    # Very few samples for smoke test
    model.przygotuj_predykcyjny(num_samples=10, burn_in=5, proposal_scale=0.1)
    pred = model.posterior_mean()
    assert pred.shape == (len(space_points),)
    assert np.all(np.isfinite(pred))

def test_axiomatic_preparamed_model_smoke(small_dataset):
    space_points, obs_idx = small_dataset["space_points"], small_dataset["observed_indices"]
    model = ModelAksjomatycznyPreparamed(
        space_points=space_points, 
        metric_func=haversine, 
        observed_indices=obs_idx,
        lengthscale=1000.0,
        variance=1.0, 
        distance_unit='km'
    )
    model.przygotuj_apriori()
    model.przygotuj_predykcyjny(num_samples=10, burn_in=5, proposal_scale=0.1)
    pred = model.posterior_mean()
    assert pred.shape == (len(space_points),)
    assert np.all(np.isfinite(pred))

def test_dirichlet_model_smoke(small_dataset):
    model = ModelDirichleta(small_dataset["observed_indices"], small_dataset["n_points"])
    pred = model.posterior_mean()
    assert pred.shape == (small_dataset["n_points"],)
    assert np.all(pred >= 0)
    assert np.isclose(np.sum(pred), 1.0)

def test_gaussian_spatial_model_smoke(small_dataset):
    space_points, n_points, obs_idx = small_dataset["space_points"], small_dataset["n_points"], small_dataset["observed_indices"]
    observed_counts = np.bincount(obs_idx, minlength=n_points)
    empirical_probs = observed_counts / len(obs_idx)
    unique_indices = np.where(observed_counts > 0)[0]
    observed_values = empirical_probs[unique_indices]

    model = ModelGausowskiSprzezony(
        space_points=space_points, metric_func=haversine, observed_indices=unique_indices,
        counts=observed_values, distance_unit='km',
        mu_prior=np.mean(observed_values),
        sigma_prior=np.std(observed_values) if np.std(observed_values) > 0 else 1.0
    )
    model.fit(phi=100, optimize_phi=False)
    pred, (lower, upper) = model.predict(num_samples=10)
    
    assert pred.shape == (n_points,)
    assert np.all(np.isfinite(pred))
    assert np.allclose(pred[unique_indices], observed_values)
