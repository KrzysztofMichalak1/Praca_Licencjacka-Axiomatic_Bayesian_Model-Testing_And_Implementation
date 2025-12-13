"""Default configuration for the Bayesian Model System."""

def get_configuration():
    """Returns the default configuration dictionary."""
    return {
        'csv_path': r"Global_2020_MarineSpeciesRichness_AquaMaps (4).csv",
        'base_params': {
            'cutoff_km': 1000,
            'co_ktory': 200,
            'n_observations': 2000,
            'n_points': 0
        },
        'models_to_test': [
             ('logistic_normal_mcmc', {
                'lengthscale': 1750,
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
                'mcmc_seed': 42
            }),
            ('bayesian', {
                'lengthscale': 5000,
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
                'mcmc_seed': 42
            }),
            #('dirichlet', {}),
            #('spatial', {'smoothing_factor': 0.1})
        ],
        'impact_models_to_test': [
            ('bayesian', {
                'lengthscale': 5000,
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
            }),
           # ('bayesian_adaptive_search_binary', {
           #      'variance': 1.0,
           #      'distance_unit': "km",
           #      'start_ls': 8000,
           #      'step_size': 8000,
           #      'k_steps': 6,
           #      'num_samples_search': 300,
           #      'burn_in_search': 150,
           #      'proposal_scale_search': 0.05,
           #      'mcmc_samples': 5000,
           #      'mcmc_burn': 3000,
           #      'mcmc_scale': 0.05,
           #      'mcmc_seed': 42
           #  }),
            ('logistic_normal_mcmc', {
                'lengthscale': 750,
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
            }),
            #('dirichlet', {}),
            #('spatial', {'smoothing_factor': 0.1})
        ],
        'n_tests': 2,
        'save_results': False,
        'run_options': {"rt": False, "it": True}
    }
