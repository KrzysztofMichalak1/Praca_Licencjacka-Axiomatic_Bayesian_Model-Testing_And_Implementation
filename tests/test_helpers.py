#
# Plik: test_helpers.py
#
# Opis:
# Ten plik zawiera testy jednostkowe dla małych, niezależnych funkcji pomocniczych
# używanych w różnych częściach systemu.
#
# Cel testów:
# Weryfikacja poprawności działania funkcji dla znanych danych wejściowych i
# oczekiwanych wyników. Testy sprawdzają również przypadki brzegowe i
# stabilność numeryczną.
#
# Testowane funkcje:
# - `haversine`: Sprawdza obliczanie odległości między dwoma znanymi punktami
#   geograficznymi.
# - `z_to_x_softmax`: Weryfikuje, czy transformacja softmax działa poprawnie,
#   czy jej wyniki sumują się do 1 i czy jest stabilna numerycznie dla
#   dużych wartości wejściowych.
#

import numpy as np
import pytest
from BayesianModelSystem.Wczytywanie_danych.metric import haversine
from BayesianModelSystem.Modele.model_lenka_preparamed import z_to_x_softmax

def test_haversine_known_values():
    """
    Testuje funkcję haversine dla znanych punktów.
    """
    # Równik do równika w odległości 90 stopni longitudy
    p1 = (0, 0)
    p2 = (0, 90)
    # Obwód ziemi to ok 40075 km, 1/4 tego to ok 10018 km
    distance = haversine(p1, p2)
    assert 10000 < distance < 10025

    # Biegun Północny do Południowego
    p_north = (0, 90)
    p_south = (0, -90)
    distance_poles = haversine(p_north, p_south)
    # Połowa obwodu ziemi
    assert 19900 < distance_poles < 20050

def test_z_to_x_softmax_basic():
    """
    Testuje podstawowe działanie softmax.
    """
    z = np.array([1.0, 2.0, 3.0])
    x = z_to_x_softmax(z)
    assert x.shape == z.shape
    assert np.isclose(np.sum(x), 1.0)
    # Wartości powinny być w dobrej kolejności
    assert x[0] < x[1] < x[2]

def test_z_to_x_softmax_stability():
    """
    Testuje stabilność numeryczną softmax dla dużych wartości.
    """
    z = np.array([1000.0, 1001.0, 1002.0])
    x = z_to_x_softmax(z)
    assert not np.isnan(x).any()
    assert np.isclose(np.sum(x), 1.0)

def test_z_to_x_softmax_zero_sum_exp():
    """
    Testuje przypadek, gdy suma exp(Z) może być zero (bardzo małe wartości).
    """
    # Numba może optymalizować inaczej, ale testujemy logikę fallback
    z = np.array([-10000.0, -10000.0, -10000.0])
    x = z_to_x_softmax(z)
    # Powinien zwrócić rozkład jednostajny
    assert np.allclose(x, np.array([1/3.0, 1/3.0, 1/3.0]))
