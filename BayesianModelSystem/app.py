"""Main entry point for the Bayesian Model System."""
import sys
import os

# This is a workaround to allow running the script directly
# It adds the root of the project to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BayesianModelSystem.program import program
from BayesianModelSystem.config.default_params import get_configuration

if __name__ == "__main__":
    config = get_configuration()
    program(**config)