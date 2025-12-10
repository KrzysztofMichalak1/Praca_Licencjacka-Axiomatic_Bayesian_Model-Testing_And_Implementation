"""This module contains functions for calculating prediction quality metrics."""
import numpy as np

def oblicz_metryki(true_probs, pred_probs, model_name, verbose=True):
    """Calculates basic prediction quality metrics."""
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

