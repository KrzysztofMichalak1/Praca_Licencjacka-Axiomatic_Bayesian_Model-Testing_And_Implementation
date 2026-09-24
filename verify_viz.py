import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.getcwd())

from BayesianModelSystem.Pomocnicze.visualization import stworz_mape_porownawcza
from BayesianModelSystem.Wczytywanie_danych.preprocessing import przygotuj_dane

def verify():
    csv_path = "Global_2020_MarineSpeciesRichness_AquaMaps (4).csv"
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found!")
        return

    print("Loading data...")
    # Loading a small subset for speed
    gdf, _, true_probs = przygotuj_dane(csv_path, cutoff_km=1000, co_ktory=500, n_observations=100)
    
    # Create some dummy prediction data (true + noise)
    pred_probs = true_probs + np.random.normal(0, 0.01, size=len(true_probs))
    pred_probs = np.clip(pred_probs, 0, 1)
    
    print("Generating comparison map...")
    stworz_mape_porownawcza(gdf, true_probs, pred_probs, title_suffix="VERIFICATION_TEST")
    
    print("Generating reference point map...")
    from BayesianModelSystem.Pomocnicze.visualization import pokaz_punkt_referencyjny
    pokaz_punkt_referencyjny(gdf, title="PUNKT_REFERENCYJNY_TEST")
    
    print("Done. Please check generated PNG files.")

if __name__ == "__main__":
    verify()
