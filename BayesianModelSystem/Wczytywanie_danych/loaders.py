"""This module contains functions for loading data."""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from cartopy.io import shapereader
import numpy as np

def wczytaj_dane(csv_path):
    """Loads data from a CSV file and returns a GeoDataFrame."""
    print("▶ [DATA] Wczytywanie danych z CSV...")
    df = pd.read_csv(csv_path)
    print(f"  - Wczytano {len(df)} rekordów.")
    geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")
    gdf["id"] = range(len(gdf))
    print("  ✔ Geopandas GeoDataFrame gotowe.")
    return gdf

def wczytaj_pacyfik():
    """Loads the shape of the Pacific Ocean."""
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
    """Loads the shape of land for the map background."""
    print("▶ [MAPA] Wczytywanie kształtu lądów...")
    land_geoms = list(shapereader.Reader(shapereader.natural_earth(
        resolution='110m',
        category='physical',
        name='land')).geometries())
    land = gpd.GeoSeries(land_geoms, crs="EPSG:4326")
    print("  ✔ Kształt lądów gotowy.")
    return land

def losuj_obserwacje(gdf, n_points):
    """Draws observations based on Species Count."""
    print(f"\n▶ [OBS] Losowanie obserwacji ({n_points}) wg Species Count...")
    species_counts = gdf["Species Count"].values
    probs = species_counts / species_counts.sum()
    idx = np.random.choice(len(gdf), size=n_points, p=probs, replace=True)
    print(f"  ✔ Zaliczone: wylosowano {len(idx)} indeksów.")
    return idx
