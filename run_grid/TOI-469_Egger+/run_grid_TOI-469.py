# *****************************************************************************
# --- plaNETic ---
# INFERRING THE INTERNAL STRUCTURE OF AN OBSERVED EXOPLANET
# EXECUTABLE FOR TOI-469
# Written by Jo Ann Egger, Universität Bern
# Copyright (c) 2024, Jo Ann Egger, Universität Bern
# *****************************************************************************
import os
import sys
sys.path.append('..')

import plaNETic

comp_option_mass = 0                        # 0 - relative mass, 1 - planet mass
comp_option_radius = 1                      # 0 - transit depth, 1 - radius ratio, 2 - planet radius
with_gas = True
with_water = True
use_log_prior_for_gas_mass = True
csv_file = 'stellar_planetary_parameters.csv'


# ------------------------------------- uniformly mixed water and H/He envelope ------------------------------------

SiMgFe_ratio_option = 0                     # Stellar (Thiabaud et al. 2015)
water_mixing_option = 'uniform'
comment = 'A1'
luminosity_option = 'from_stellar_age'

plaNETic.compute_posterior(comp_option_mass, comp_option_radius, SiMgFe_ratio_option, water_mixing_option, luminosity_option, use_log_prior_for_gas_mass, with_gas, with_water, csv_file, comment, verbose=False, save_prior=True)

SiMgFe_ratio_option = 1                     # Iron-enriched (Adibekyan et al. 2021)
water_mixing_option = 'uniform'
comment = 'A2'
luminosity_option = 'from_stellar_age'

plaNETic.compute_posterior(comp_option_mass, comp_option_radius, SiMgFe_ratio_option, water_mixing_option, luminosity_option, use_log_prior_for_gas_mass, with_gas, with_water, csv_file, comment, verbose=False, save_prior=True)

SiMgFe_ratio_option = 2                     # Free, with max. 75% iron with respect to refractories
water_mixing_option = 'uniform'
comment = 'A3'
luminosity_option = 'from_stellar_age'

plaNETic.compute_posterior(comp_option_mass, comp_option_radius, SiMgFe_ratio_option, water_mixing_option, luminosity_option, use_log_prior_for_gas_mass, with_gas, with_water, csv_file, comment, verbose=False, save_prior=True)


# ---------------- uniformly mixed water and H/He envelope, assuming water is only accreted as gas ------------------

SiMgFe_ratio_option = 0                     # Stellar (Thiabaud et al. 2015)
water_mixing_option = 'water_from_gas'
comment = 'B1'
luminosity_option = 'from_stellar_age'

plaNETic.compute_posterior(comp_option_mass, comp_option_radius, SiMgFe_ratio_option, water_mixing_option, luminosity_option, use_log_prior_for_gas_mass, with_gas, with_water, csv_file, comment, verbose=False, save_prior=True)

SiMgFe_ratio_option = 1                     # Iron-enriched (Adibekyan et al. 2021)
water_mixing_option = 'water_from_gas'
comment = 'B2'
luminosity_option = 'from_stellar_age'

plaNETic.compute_posterior(comp_option_mass, comp_option_radius, SiMgFe_ratio_option, water_mixing_option, luminosity_option, use_log_prior_for_gas_mass, with_gas, with_water, csv_file, comment, verbose=False, save_prior=True)

SiMgFe_ratio_option = 2                     # Free, with max. 75% iron with respect to refractories
water_mixing_option = 'water_from_gas'
comment = 'B3'
luminosity_option = 'from_stellar_age'

plaNETic.compute_posterior(comp_option_mass, comp_option_radius, SiMgFe_ratio_option, water_mixing_option, luminosity_option, use_log_prior_for_gas_mass, with_gas, with_water, csv_file, comment, verbose=False, save_prior=True)