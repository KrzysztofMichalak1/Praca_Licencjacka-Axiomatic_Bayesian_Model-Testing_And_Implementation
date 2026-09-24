"""Default configuration for the Bayesian Model System."""

def get_configuration():
    """Returns the default configuration dictionary."""
    return {
        'csv_path': r"Global_2020_MarineSpeciesRichness_AquaMaps (4).csv",
        'base_params': {
            'cutoff_km': 1000,
            'co_ktory': 50,
            'n_observations': 400,
            'n_points': 0
        },
        'models_to_test': [
            ('Model_Aksjomatyczny', {
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
                'mcmc_seed': 42
            }),
             ('Model_Dwumianowy_sprzężony', {
                'alpha_prior': 0.5,
                'beta_prior': 0.5,
                'smoothing_strength': 1,
                'optimize_phi': False,
                "phi": 4
             }),
             ('Model_Poissona_sprzężony', {
                'alpha_prior': 0.5,
                'beta_prior': 0.5,
                'optimize_phi': False,
                "phi": 4,
             }),
             ('Model_Lenka', {
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
            #  ('Model_Lenka-preparamed', {
            #     'lengthscale': 1250,
            #     'variance': 1.0,
            #     'distance_unit': "km",
            #     'mcmc_samples': 5000,
            #     'mcmc_burn': 3000,
            #     'mcmc_scale': 0.05,
            #     'mcmc_seed': 42
            # }),
            # ('Model_Aksjomatyczny-preparamed', {
            #     'lengthscale': 12000,
            #     'variance': 1.0,
            #     'distance_unit': "km",
            #     'mcmc_samples': 5000,
            #     'mcmc_burn': 3000,
            #     'mcmc_scale': 0.05,
            #     'mcmc_seed': 42
            #}),
            
            ('Model_Dirichleta', {}),
            ('Model_wygładzania_przestrzennego', {'smoothing_factor': 0.1})
        ],
        'impact_models_to_test': [
             ('Model_Dwumianowy_sprzężony', {
                'alpha_prior': 0.5,
                'beta_prior': 0.5,
                'smoothing_strength': 1,
                'optimize_phi': False,
                "phi": 4
             }),
             ('Model_Poissona_sprzężony', {
                'alpha_prior': 0.5,
                'beta_prior': 0.5,
                'optimize_phi': False,
                "phi": 4,
             }),
             ('Model_Lenka', {
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
             # ('Model_Aksjomatyczny-preparamed', {
             #    'lengthscale': 12000,
             #    'variance': 1.0,
             #    'distance_unit': "km",
             #    'mcmc_samples': 5000,
             #    'mcmc_burn': 3000,
             #    'mcmc_scale': 0.05,
             #    'mcmc_seed': 42
            #}),
             ('Model_Aksjomatyczny', {
                'variance': 1.0,
                'distance_unit': "km",
                'mcmc_samples': 5000,
                'mcmc_burn': 3000,
                'mcmc_scale': 0.05,
                'mcmc_seed': 42
            }),
             ('Model_Dirichleta', {}),
            ('Model_wygładzania_przestrzennego', {'smoothing_factor': 0.1})
        ],
        'db_path': 'wyniki_testow.csv',
        'visualise': True,
        'save_results': True,
        'run_options': {"rt": True, "it":True, "lt": False, "nt": True},
        'lengthscale_list': [100,250,500,1000,2500,5000,10000,25000,50000,100000],
        'nemenyi_params': {
            'n_observations': 400,
            'k': 100
        }
    }
