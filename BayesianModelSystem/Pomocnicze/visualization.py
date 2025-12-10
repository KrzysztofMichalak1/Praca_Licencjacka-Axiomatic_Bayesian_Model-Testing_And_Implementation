"This module contains functions for data visualization."
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def stworz_mape_porownawcza(gdf, true_probs, pred_probs, title_suffix=""):
    """Creates a comparative map of true and predicted distributions."""
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

def pokaz_punkt_referencyjny(gdf, title="Punkt referencyjny (indeks 1)"):
    """Shows the reference point on the map."""
    print(f"▶ [MAP] Tworzenie mapy punktu referencyjnego...")
    
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
