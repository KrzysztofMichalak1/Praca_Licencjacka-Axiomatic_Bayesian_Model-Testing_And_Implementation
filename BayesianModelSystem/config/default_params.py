"""Default configuration for the Bayesian Model System."""

def get_configuration():
    """Returns the default configuration dictionary."""
    return {
        'csv_path': r"Global_2020_MarineSpeciesRichness_AquaMaps (4).csv",
        'base_params': {
            'cutoff_km': 1000,
            'co_ktory': 100,
            'n_observations': 100,
            'n_points': 0
        },
        'models_to_test': [
             ('spatial_binomial', {
                'alpha_prior': 0.5,
                'beta_prior': 0.5,
                'smoothing_strength': 1,
                'optimize_phi': False,
                "phi":4
             }),
             
             ('spatial_poisson', {
                'alpha_prior': 0.5,
                'beta_prior': 0.5,
                'optimize_phi':False,
                "phi":4,
             }),
             ('lenk_adaptive', {
                'variance': 1.0,
                'distance_unit': "km",
                'start_ls': 800,
                'step_size': 800,
                'k_steps': 6,
                'num_samples_search': 300,
                'burn_in_search': 150,
                'proposal_scale_search': 0.05,
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
                'mcmc_seed': 42
             }),
             ('logistic_normal_mcmc', {
                'lengthscale': 1250,
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
                'mcmc_seed': 42
            }),
            ('bayesian', {
                'lengthscale': 12000,
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
                'mcmc_seed': 42
            }),
            ('dirichlet', {}),
            ('spatial', {'smoothing_factor': 0.1})
        ],
        'impact_models_to_test': [
            # ('lenk_adaptive', {
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
            
            # ('bayesian', {
            #     'lengthscale': 11000,
            #     'variance': 1.0,
            #     'distance_unit': "km",
            #     'mcmc_samples': 4000,
            #     'mcmc_burn': 3000,
            #     'mcmc_scale': 0.05,
            # }),
            #  ('spatial_binomial', {
            #     'alpha_prior': 0.5,
            #     'beta_prior': 0.5,
            #     'smoothing_strength': 1,
            #     'optimize_phi': False,
            #     "phi":4
            #  }),
            #  ('spatial_gaussian', {
            #     'mu_prior': 0.0,
            #     'sigma_prior': 1.0,
            #     'optimize_phi': False,
            #     "phi":5,
            #  }),
            #  ('spatial_poisson', {
            #     'alpha_prior': 0.5,
            #     'beta_prior': 0.5,
            #     'optimize_phi':False,
            #     "phi":4,
            #  }),
           ('bayesian_adaptive_search_binary', {
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
                'mcmc_seed': 42,
                'p':0.9
            }),
            ('bayesian_adaptive_search_binary', {
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
                'mcmc_seed': 42,
                'p':1
            }),
            ('bayesian_adaptive_search_binary', {
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
                'mcmc_seed': 42,
                'p':1.5
            }),
            # ('logistic_normal_mcmc', {
            #     'lengthscale': 1150,
            #     'variance': 1.0,
            #     'distance_unit': "km",
            #     'mcmc_samples': 5000,
            #     'mcmc_burn': 3000,
            #     'mcmc_scale': 0.05,
            # }),
            # ('dirichlet', {}),
            # ('spatial', {'smoothing_factor': 0.1}),
            ('spatial_gaussian', {
                'mu_prior': 0.0,
                'sigma_prior': 1.0,
                'optimize_phi': True,
            }),
            # ('spatial_poisson', {
            #     'alpha_prior': 0.5,
            #     'beta_prior': 0.5,
            #     'optimize_phi': True,
            #}),
        ],
        'n_tests': 2,
        'save_results': False,
        'run_options': {"rt":False, "it": True, "lt": True},
        'lengthscale_list': [#800,900, 1000,1100,1200,1300,1400,1500, 2000 ,
            #8000,10000,11000,12000,
            13000,15000,20000
            ]
    }
