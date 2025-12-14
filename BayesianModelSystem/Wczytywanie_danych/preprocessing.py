"""This module contains data preprocessing functions."""
import numpy as np
from .loaders import wczytaj_pacyfik, wczytaj_lad, losuj_obserwacje, wczytaj_dane

def filtruj_pacyfik_i_brzeg(gdf_points, cutoff_km, iqr_multiplier=1.5):
    """Filters points to the Pacific area and removes outliers and coastal points."""
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
    land_union = land.union_all()

    cutoff_deg = cutoff_km / 111.0
    buffer = land_union.buffer(cutoff_deg)

    before_coast = len(gdf_f)
    gdf_f = gdf_f[~gdf_f.geometry.within(buffer)]
    gdf_f.reset_index(drop=True, inplace=True)
    print(f"  - Pozostało: {len(gdf_f)} (usunięto {before_coast - len(gdf_f)} przy brzegu)")

    return gdf_f

def zmniejsz_siatke(gdf, co_ktory):
    """Reduces the grid by taking every n-th point."""
    print(f"\n▶ [REDUKCJA] Redukcja siatki co {co_ktory} punkt...")
    before = len(gdf)
    if co_ktory <= 1:
        print("  - Pomijam redukcję (co_ktory<=1)")
        return gdf.copy()
    reduced = gdf.iloc[::co_ktory].copy()
    reduced.reset_index(drop=True, inplace=True)
    print(f"  - Przed: {before}, Po redukcji: {len(reduced)}")
    return reduced

def przygotuj_dane(csv_path, cutoff_km, co_ktory, n_observations):
    """Data preparation pipeline."""
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
