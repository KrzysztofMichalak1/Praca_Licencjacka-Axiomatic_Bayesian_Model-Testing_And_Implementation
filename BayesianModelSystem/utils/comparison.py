"""This module contains functions for comparing models."""
import numpy as np
from scipy.stats import wilcoxon

def oblicz_statystyki_porownania_wszystkich(true_probs, bayesian_pred, dirichlet_pred, gp_pred, spatial_pred):
    """Calculates comparison statistics between all models."""
    errors_bayesian = np.abs(bayesian_pred - true_probs)
    errors_dirichlet = np.abs(dirichlet_pred - true_probs)
    errors_gp = np.abs(gp_pred - true_probs)
    errors_spatial = np.abs(spatial_pred - true_probs)
    
    # Który model jest lepszy w ilu punktach
    bayesian_better_count = 0
    dirichlet_better_count = 0
    gp_better_count = 0
    spatial_better_count = 0
    equal_count = 0
    
    for i in range(len(true_probs)):
        errors = {
            'bayesian': errors_bayesian[i],
            'dirichlet': errors_dirichlet[i],
            'gp': errors_gp[i],
            'spatial': errors_spatial[i]
        }
        min_error = min(errors.values())
        
        # Zlicz które modele mają minimalny błąd
        best_models = [model for model, error in errors.items() if error == min_error]
        
        if len(best_models) == 1:
            if best_models[0] == 'bayesian':
                bayesian_better_count += 1
            elif best_models[0] == 'dirichlet':
                dirichlet_better_count += 1
            elif best_models[0] == 'gp':
                gp_better_count += 1
            elif best_models[0] == 'spatial':
                spatial_better_count += 1
        else:
            equal_count += 1
    
    # Test Wilcoxona między Bayesian a Dirichlet (dla zachowania kompatybilności)
    try:
        stat, wilcoxon_pvalue = wilcoxon(errors_bayesian, errors_dirichlet)
        wilcoxon_pvalue = float(wilcoxon_pvalue)
    except:
        wilcoxon_pvalue = 1.0
    
    return {
        'bayesian_better_count': bayesian_better_count,
        'dirichlet_better_count': dirichlet_better_count,
        'gp_better_count': gp_better_count,
        'spatial_better_count': spatial_better_count,
        'equal_count': equal_count,
        'wilcoxon_pvalue': wilcoxon_pvalue
    }
