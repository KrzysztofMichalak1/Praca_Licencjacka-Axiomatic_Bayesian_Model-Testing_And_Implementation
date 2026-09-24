"This module contains functions for data visualization."
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.colors as mcolors
import re

def stworz_mape_porownawcza(gdf, true_probs, pred_probs, title_suffix=""):
    """Creates a comparative map of true and predicted distributions."""
    print(f"▶ [MAP] Tworzenie mapy porównawczej {title_suffix}...")
    
    # Oblicz wspólny zakres dla skal kolorów (ignorując NaN)
    combined = np.concatenate([true_probs, pred_probs])
    vmin = np.nanmin(combined)
    vmax = np.nanmax(combined)
    print(f"    - Zakres prawdopodobieństwa: [{vmin:.2e}, {vmax:.2e}]")
    
    # Tworzenie figury
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11), dpi=300,
                                  subplot_kw={'projection': ccrs.PlateCarree()})

    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13})
    cmap = 'magma'

    # Funkcja pomocnicza do rysowania pojedynczej mapy
    def rysuj_mape(ax, data, title):
        sc = ax.scatter(gdf["Longitude"], gdf["Latitude"],
                         c=data, cmap=cmap, s=75, alpha=0.95,
                         vmin=vmin, vmax=vmax, edgecolors='none', transform=ccrs.PlateCarree())
        
        # Dodawanie cech geograficznych
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':', zorder=3)
        ax.add_feature(cfeature.LAND, facecolor='#f9f9f9', alpha=1.0, zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor='#e0f2ff', alpha=1.0, zorder=0)
        
        # Dodawanie siatki (gridlines)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                         alpha=0.1, linestyle='--', color='gray', zorder=2)
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = (ax == ax2) # Etykiety dolne tylko na dolnej mapie
        
        # Ustawianie zasięgu na podstawie danych
        lon_min, lon_max = gdf["Longitude"].min(), gdf["Longitude"].max()
        lat_min, lat_max = gdf["Latitude"].min(), gdf["Latitude"].max()
        padding = 3.0
        ax.set_extent([lon_min - padding, lon_max + padding, 
                        lat_min - padding, lat_max + padding], crs=ccrs.PlateCarree())
        
        ax.set_title(title, fontweight='bold', pad=5)
        return sc

    # Rysowanie obu map
    sc1 = rysuj_mape(ax1, true_probs, f'Prawdziwy rozkład bogactwa gatunkowego ({title_suffix})')
    sc2 = rysuj_mape(ax2, pred_probs, f'Predykowany rozkład bogactwa gatunkowego ({title_suffix})')

    # Dodanie wspólnego paska koloru (colorbar) na dole
    fig.subplots_adjust(bottom=0.12, hspace=0.02, top=0.96, left=0.05, right=0.95)
    cbar_ax = fig.add_axes([0.3, 0.06, 0.4, 0.02]) # [left, bottom, width, height]
    cbar = fig.colorbar(sc2, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Prawdopodobieństwo występowania', fontweight='bold', labelpad=10)
    
    # Usuwamy tight_layout bo ręcznie ustawiliśmy subplots_adjust i colorbar
    # plt.tight_layout()
    
    # Generowanie bezpiecznej nazwy pliku
    safe_suffix = re.sub(r'[^\w\s-]', '', title_suffix).strip().replace(' ', '_').lower()
    file_name = f'mapa_porownawcza_{safe_suffix}.png'
    
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close(fig) # Zwolnienie pamięci
    
    print(f"  ✔ Mapa została zapisana w pliku: {file_name}")
    return fig

def pokaz_punkt_referencyjny(gdf, title="Punkt referencyjny (indeks 1)"):
    """Shows the reference point on the map."""
    print(f"▶ [MAP] Tworzenie mapy punktu referencyjnego...")
    
    ref_point = (gdf.iloc[0]["Longitude"], gdf.iloc[0]["Latitude"])
    ref_idx = 0
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Wszystkie punkty szare
    ax.scatter(gdf["Longitude"], gdf["Latitude"], 
               c='lightgray', s=20, alpha=0.5, label='Wszystkie punkty', transform=ccrs.PlateCarree())
    
    # Punkt referencyjny czerwony
    ax.scatter(ref_point[0], ref_point[1], 
               c='red', s=100, marker='*', edgecolors='black', linewidth=1.5,
               label=f'Punkt referencyjny (indeks 1)\n{ref_point[0]:.2f}°, {ref_point[1]:.2f}°',
               transform=ccrs.PlateCarree(), zorder=5)
    
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.1)
    
    # Ustawianie zasięgu globalnego lub lokalnego (tu globalny z lekkim zoomem jeśli trzeba)
    ax.set_global()
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Dodaj adnotację z współrzędnymi
    ax.annotate(f'({ref_point[0]:.2f}°, {ref_point[1]:.2f}°)', 
                xy=(ref_point[0], ref_point[1]), xytext=(10, 10),
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                transform=ccrs.PlateCarree())
    
    plt.tight_layout()
    plt.savefig('punkt_referencyjny.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✔ Punkt referencyjny: {ref_point}")
    print(f"  ✔ Indeks: {ref_idx}")
    
    return ref_point, ref_idx
