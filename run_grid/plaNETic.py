# *****************************************************************************
# --- plaNETic ---
# CODE FOR INFERING THE INTERNAL STRUCTURE OF AN OBSERVED EXOPLANET
# Written by Jo Ann Egger and Yann Alibert, Universität Bern
# Copyright (c) 2024, Jo Ann Egger, Universität Bern
# *****************************************************************************

# IMPORTS
# *****************************************************
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import pylab as p
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from datetime import date
from datetime import datetime

from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error

import matplotlib
import matplotlib.ticker as plticker
matplotlib.rcParams['xtick.labelsize'] = 'x-large'
matplotlib.rcParams['ytick.labelsize'] = 'x-large'

import pickle
import corner

import h5py
import scipy.optimize as opt
import ternary

from scipy.ndimage.filters import gaussian_filter
import tqdm
from scipy import interpolate



# ----------------------------------------------------------------------------------------------------------------------------------------------------
# NUMERICAL CONSTANTS
# *****************************************************
Ptransit = 0.1  # SI
RGP = 8.314 # SI
MH2O = 1.8e-2 # SI
MHHe = 2.23e-3 # SI
constPI = 3.141592654
constG = 6.67e-11 # SI
Mearth = 5.97e24  # SI
AU = 149597870700 # SI
Rearth = 6.3782e6 # SI
const_kb = 1.380649e-23  # SI
amu = 1.6605390666e-27 # SI
mu_HHe = 2.23 * amu
mu_H2O = 18.01528 * amu
Patm_max = 100e5 # SI
Ptransit = 20e2 # SI
const_sigma =  5.67037441e-8 # SI
Msun = 1.988413833e30 # SI
Rsun = 6.96e+8 # SI
day = 86400. # SI
Mjup_Mearth = 317.82 # in Earth masses
Msun_Mjup = Msun / (Mjup_Mearth * Mearth)
fe_si_mass_sun = 1.69
mu_Si = 28.085
mu_Fe = 55.845
fe_si_sun = mu_Si/mu_Fe*fe_si_mass_sun
SiFe_sun = 1/fe_si_sun
sigma_SiFe_sun = 0.18 * mu_Si/mu_Fe
mu_Mg = 24.305
MgFe_sun = 0.89/1.69*mu_Fe/mu_Mg
sigma_MgFe_sun = sigma_SiFe_sun+0.08*mu_Fe/mu_Mg
#meteoritic abundances in the sun, Asplund et al. AR&A
SiH_sun = 7.51
MgH_sun = 7.53
FeH_sun = 7.45


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Functions
# *****************************************************

# Read input files
# ----------------------------------------------------------
def read_csv_info(comp_option_mass, comp_option_radius, with_gas, with_water, csv_file='stellar_planetary_parameters.csv', comment='', date_string='auto', verbose=True):
    """ Read CSV input file with stellar and planetary parameters
    
    Args:
        comp_option_mass:       Provided observables for mass (0 - relative mass, 1 - planet mass)
        comp_option_radius:     Provided observables for radius (0 - transit depth, 1 - radius ratio, 2 - planet radius)
        csv_file:               Name of input CSV file
        comment:                Comment string to be added to name
        date_string:            
        verbose:                Verbose setting (print read input data for each system)
    """

    obs = pd.read_csv(csv_file)
    obs['Name'] = obs['Name'].str.replace(' ','_')
    if date_string == 'auto':
        obs['Name'] = obs['Name'] + '_' + date.today().strftime("%Y-%m-%d")
    else:
        obs['Name'] = obs['Name'] + '_' + date_string
    if comment != '':
        obs['Name'] = obs['Name'] + '_' + comment
    
    if not with_gas:
        obs['Name'] = obs['Name'] + '_noGas'
    if not with_water:
        obs['Name'] = obs['Name'] + '_dryModel'

    nplanets_total = int(len(obs))
    
    count = 0
    nsystems_total = 0
    while count < nplanets_total:
        count += obs['Nplanets_system'].iloc[count]
        nsystems_total += 1

    if verbose:
        if nplanets_total > 1:
            print(str(nplanets_total) + ' planets total')
        else:
            print(str(nplanets_total) + ' planet total')
        if nsystems_total > 1:
            print(str(nsystems_total) + ' systems total')
        else:
            print(str(nsystems_total) + ' system total')

        count = 0
        for j in range(nsystems_total):
            print('OVERVIEW OVER INPUT DATA')
            print('')
            print('System ' + str(j+1) + ': ')
            print('STELLAR PARAMETERS')
            print('Name:                    ' + str(obs['Name'].iloc[count]))
            print('M*:                      (' + str(obs['Mstar[Msun]'].iloc[count]) + ' +' + str(obs['Mstar_E[Msun]'].iloc[count]) + ' -' + str(obs['Mstar_e[Msun]'].iloc[count]) + ') M_Sun')
            print('R*:                      (' + str(obs['Rstar[Rsun]'].iloc[count]) + ' +' + str(obs['Rstar_E[Rsun]'].iloc[count]) + ' -' + str(obs['Rstar_e[Rsun]'].iloc[count]) + ') R_Sun')
            print('Teff:                    (' + str(obs['Teff[K]'].iloc[count]) + ' +' + str(obs['Teff_E[K]'].iloc[count]) + ' -' + str(obs['Teff_e[K]'].iloc[count]) + ') K')
            print('[Si/H]:                  (' + str(obs['SiH'].iloc[count]) + ' +' + str(obs['SiH_E'].iloc[count]) + ' -' + str(obs['SiH_e'].iloc[count]) + ')')
            print('[Mg/H]:                  (' + str(obs['MgH'].iloc[count]) + ' +' + str(obs['MgH_E'].iloc[count]) + ' -' + str(obs['MgH_e'].iloc[count]) + ')')
            print('[Fe/H]:                  (' + str(obs['FeH'].iloc[count]) + ' +' + str(obs['FeH_E'].iloc[count]) + ' -' + str(obs['FeH_e'].iloc[count]) + ')')
            print('Age:                     (' + str(obs['Age[Gyr]'].iloc[count]) + ' +' + str(obs['Age_E[Gyr]'].iloc[count]) + ' -' + str(obs['Age_e[Gyr]'].iloc[count]) + ')')
            print('')
            print('')

            print('PLANETARY PARAMETERS')
            print('Planets in the System:   ' + str(obs['Nplanets_system'].iloc[count]))
            for i in range(int(obs['Nplanets_system'].iloc[count])):
                print('')
                print('PLANET ' + chr(ord('`')+i+2))
                print('Period:                  (' + str(obs['P[day]'].iloc[i+count]) + ' +' + str(obs['P_E[day]'].iloc[i+count]) + ' -' + str(obs['P_e[day]'].iloc[i+count]) + ') days')
                if comp_option_mass == 0:
                    print('Mplanet:                 (' + str(obs['m[M_Earth]'].iloc[i+count]) + ' +' + str(obs['m_E[M_Earth]'].iloc[i+count]) + ' -' + str(obs['m_e[M_Earth]'].iloc[i+count]) + ') M_Earth')
                    print('K_RV:                    (' + str(obs['KRV[ms-1]'].iloc[i+count]) + ' +' + str(obs['KRV_E[ms-1]'].iloc[i+count]) + ' -' + str(obs['KRV_e[ms-1]'].iloc[i+count]) + ') m s-1')
                elif comp_option_mass == 1:
                    print('Mplanet:                 (' + str(obs['m[M_Earth]'].iloc[i+count]) + ' +' + str(obs['m_E[M_Earth]'].iloc[i+count]) + ' -' + str(obs['m_e[M_Earth]'].iloc[i+count]) + ') M_Earth')
                if comp_option_radius == 0:
                    print('Transit depth:           (' + str(obs['td[ppm]'].iloc[i+count]) + ' +' + str(obs['td_E[ppm]'].iloc[i+count]) + ' -' + str(obs['td_e[ppm]'].iloc[i+count]) + ') ppm')
                elif comp_option_radius == 1:
                    print('Radius ratio:            (' + str(obs['rr'].iloc[i+count]) + ' +' + str(obs['rr_E'].iloc[i+count]) + ' -' + str(obs['rr_e'].iloc[i+count]) + ')')
                elif comp_option_radius == 2:
                    print('Rplanet:                 (' + str(obs['R[R_Earth]'].iloc[i+count]) + ' +' + str(obs['R_E[R_Earth]'].iloc[i+count]) + ' -' + str(obs['R_e[R_Earth]'].iloc[i+count]) + ') R_Earth')
                print('')

            print('')
            print('SETTINGS')
            if comp_option_mass == 0:
                print('Mass comparisons:        Sampling relative masses')
            elif comp_option_mass == 1:
                print('Mass comparisons:        Sampling absolute masses')
            if comp_option_radius == 0:
                print('Radius comparisons:      Comparing transit depths (from transit depth input)')
            elif comp_option_radius == 1:
                print('Radius comparisons:      Comparing transit depths (from radius ratio input)')
            elif comp_option_radius == 2:
                print('Radius comparisons:      Comparing absolute radii')

            # update count: index of first planet in next system
            count += obs['Nplanets_system'].iloc[count]
        
    return obs, nplanets_total, nsystems_total

# Load DNNs and scalers
# ----------------------------------------------------------
def load_dnns(option, verbose=False):
    """ Load trained DNN
    
    Args:
        option:       Model version that should be loaded (X_mY)
    """

    if option == 'A_m1' or option == 'A_m2' or option == 'B_m1' or option == 'B_m2':
        dnn_prediction = keras.models.load_model('../../dnns/dnn_' + option + '_prediction/dnn_' + option + '_prediction.h5')
        if verbose:
            print('Loaded DNN at: ../../dnns/dnn_' + option + '_prediction/dnn_' + option + '_prediction.h5')
    else:
        print('Error: Undefined model version - ', option)
        exit()

    return dnn_prediction

def scaler_from_files(params_filename, attributes_filename):
    """ Open parameter and attribute files and set scaler to right values
    
    Args:
        params_filename:        Filename of parameter file
        attributes_filename:    Filename of attributes file
    """

    with open(params_filename, 'rb') as params_file:
        params = pickle.load(params_file)

    with open(attributes_filename, 'rb') as attributes_file:
        attributes = pickle.load(attributes_file)
    
    scaler = StandardScaler()
    scaler.set_params(**params)

    setattr(scaler,'n_samples_seen_',attributes[3])
    setattr(scaler,'mean_',attributes[1])
    setattr(scaler,'var_',attributes[2])
    setattr(scaler,'scale_',attributes[0])
    
    return scaler

def load_scalers(option):
    """ Load scalers used for training DNN
    
    Args:
        option:       Model version that should be loaded (X_mY)
    """

    if option == 'A_m1' or option == 'A_m2' or option == 'B_m1' or option == 'B_m2':
        foldername_prediction = '../../dnns/dnn_' + option + '_prediction/'
    else:
        print('Error: Undefined model version - ', option)
        exit()

    scaler_x_prediction = scaler_from_files(foldername_prediction + 'params_scaler_x.pkl', foldername_prediction + 'attributes_scaler_x.pkl')
    scaler_y_prediction = scaler_from_files(foldername_prediction + 'params_scaler_y.pkl', foldername_prediction + 'attributes_scaler_y.pkl')
    return scaler_x_prediction, scaler_y_prediction

# Define stellar/planetary parameters and priors
# ----------------------------------------------------------
def stellar_params(obs, i):
    """ Define stellar parameters from read-in CSV file 
    
    Output:     2 1D arrays with parameters and sigmas (0 - M_star, 1 - R_star, 2 - Teff_star, 3 - SiFe_star, 4 - MgFe_star, 5 - age_star)
    
    Args:
        obs:    read-in CSV file (pandas object)
        i:      line number of first planet in current system
    """

    # *********************************
    # Mass
    # *********************************
    M_star = obs['Mstar[Msun]'].iloc[i] * Msun_Mjup
    sigma_M_star = (obs['Mstar_e[Msun]'].iloc[i] + obs['Mstar_E[Msun]'].iloc[i])/2 * Msun_Mjup
    
    # *********************************
    # Radius
    # *********************************
    R_star = obs['Rstar[Rsun]'].iloc[i]
    sigma_R_star = (obs['Rstar_e[Rsun]'].iloc[i] + obs['Rstar_E[Rsun]'].iloc[i])/2
    
    # *********************************
    # Teff
    # *********************************
    Teff_star = obs['Teff[K]'].iloc[i]
    sigma_Teff_star = (obs['Teff_e[K]'].iloc[i] + obs['Teff_E[K]'].iloc[i])/2
    
    # *********************************
    # Composition
    # *********************************
    SiH = obs['SiH'].iloc[i] 
    SiH_e = obs['SiH_e'].iloc[i] 
    SiH_E = obs['SiH_E'].iloc[i] 
    MgH = obs['MgH'].iloc[i] 
    MgH_e = obs['MgH_e'].iloc[i] 
    MgH_E = obs['MgH_E'].iloc[i] 
    FeH = obs['FeH'].iloc[i] 
    FeH_e = obs['FeH_e'].iloc[i] 
    FeH_E = obs['FeH_E'].iloc[i]
        
    SiH_absolute = SiH + SiH_sun
    MgH_absolute = MgH + MgH_sun
    FeH_absolute = FeH + FeH_sun
        
    SiFe_star = 10.**(SiH_absolute-FeH_absolute)
    MgFe_star =  10.**(MgH_absolute-FeH_absolute)
        
    sigma_SiFe_star = SiFe_star*((((SiH_e + SiH_E)/2)**2+((FeH_e + FeH_E)/2)**2)**0.5)*np.log(10)
    sigma_MgFe_star = MgFe_star*((((MgH_e + MgH_E)/2)**2+((FeH_e + FeH_E)/2)**2)**0.5)*np.log(10)

    # *********************************
    # Age
    # *********************************
    age_star = obs['Age[Gyr]'].iloc[i]
    sigma_age_star = (obs['Age_e[Gyr]'].iloc[i] + obs['Age_E[Gyr]'].iloc[i])/2
        
    # *********************************
    # Concatenate
    # *********************************
    params = np.array([M_star,R_star,Teff_star,SiFe_star,MgFe_star,age_star])
    sigmas = np.array([sigma_M_star,sigma_R_star,sigma_Teff_star,sigma_SiFe_star,sigma_MgFe_star,sigma_age_star])
    
    return params, sigmas

def compute_relative_mass(Mstar, Mplanet):
    """ Calculate relative mass of planet (to star), proportional to RV semi-amplitude
    
    Args:
        Mstar:          Stellar mass [Mjup]
        Mplanet:        Planet mass [Mearth]
    """

    return (Mplanet / Mjup_Mearth) / (Mstar)**(2/3)

def planetary_params(obs, i, comp_option_mass, comp_option_radius):
    """ Define planetary parameters from read-in CSV file
    
    Args:
        obs:                    read-in CSV file (pandas object)
        i:                      line number of current planet
        comp_option_mass:       provided observables for mass (0 - relative mass, 1 - planet mass)
        comp_option_radius:     provided observables for radius (0 - transit depth, 1 - radius ratio, 2 - planet radius)
    """

    # *********************************
    # Mass
    # *********************************
    if comp_option_mass == 0:      # compare relative mass
        mass = compute_relative_mass(obs['Mstar[Msun]'].iloc[0] * Msun_Mjup, obs['m[M_Earth]'].iloc[i])
        sigma_mass = mass * (obs['KRV_e[ms-1]'].iloc[i]+obs['KRV_E[ms-1]'].iloc[i])/(2*obs['KRV[ms-1]'].iloc[i])
    elif comp_option_mass == 1:    # compare planet mass
        mass = obs['m[M_Earth]'].iloc[i]
        sigma_mass = (obs['m_e[M_Earth]'].iloc[i]+obs['m_E[M_Earth]'].iloc[i])/2
    else:    # invalid option
        print('Invalid option: Mass comparison (' + comp_option_mass + ')')
        exit()
    
    # *********************************
    # Radius
    # *********************************
    if comp_option_radius == 0:      # compare transit depth
        radius = obs['td[ppm]'].iloc[i]
        sigma_radius = (obs['td_e[ppm]'].iloc[i]+obs['td_E[ppm]'].iloc[i])/2
    elif comp_option_radius == 1:    # compare transit depth, input is radius ratio
        radius = 1.e6*(obs['rr'].iloc[i])**2
        sigma_radius = radius * 2 * (obs['rr_e'].iloc[i]+obs['rr_E'].iloc[i])/(2*obs['rr'].iloc[i])
    elif comp_option_radius == 2:    # compare radius
        radius = obs['R[R_Earth]'].iloc[i]
        sigma_radius = (obs['R_e[R_Earth]'].iloc[i]+obs['R_E[R_Earth]'].iloc[i])/2
    else:    # invalid option
        print('Invalid option: Radius comparison (' + comp_option_radius + ')')
        exit()
    
    # *********************************
    # Period
    # *********************************
    period = obs['P[day]'].iloc[i]
    sigma_period = (obs['P_e[day]'].iloc[i] + obs['P_E[day]'].iloc[i])/2

    # *********************************
    # Concatenate
    # *********************************
    params = np.array([mass,radius,period]).reshape(1,-1)
    sigmas = np.array([sigma_mass,sigma_radius,sigma_period]).reshape(1,-1)
    
    return params, sigmas

def define_priors(params_star, sigmas_star, water_mixing_option, with_gas=False, with_water=True):
    """ Define priors for sampling
    
    Args:
        params_star:            Stellar parameters (0 - M_star, 1 - R_star, 2 - Teff_star, 3 - SiFe_star, 4 - MgFe_star, 5 - age_star)
        sigmas_star:            Sigmas of stellar parameters
        water_mixing_option:    
        with_gas:               Model with or without gas layer (True/False)
        with_water:             Model with or without water layer (True/False)
    """

    global planet_mass_max, planet_mass_min
    global M_star_max, M_star_min
    global R_star_max, R_star_min
    global Teff_star_max, Teff_star_min
    global SiFe_star_max, SiFe_star_min
    global MgFe_star_max, MgFe_star_min
    global age_star_max, age_star_min
    global luminosity_max, luminosity_min
    global w_gas_max, w_gas_min, ws_water_max
    
    # STAR
    M_star_max = min(1.5*Msun_Mjup, params_star[0] + 4*sigmas_star[0])
    M_star_min = max(0.1*Msun_Mjup, params_star[0] - 4*sigmas_star[0])
    R_star_max =  min(5,params_star[1] + 4*sigmas_star[1])
    R_star_min =  max(0.1,params_star[1] - 4*sigmas_star[1])
    Teff_star_max = min(10000, params_star[2] + 4*sigmas_star[2])
    Teff_star_min = max(100, params_star[2] - 4*sigmas_star[2])
    SiFe_star_max = min(10, params_star[3] + 4*sigmas_star[3])
    SiFe_star_min = max(0, params_star[3] - 4*sigmas_star[3])
    MgFe_star_max = min(10, params_star[4] + 4*sigmas_star[4])
    MgFe_star_min = max(0, params_star[4] - 4*sigmas_star[4])
    age_star_max = params_star[5] + 4*sigmas_star[5]
    age_star_min = max(0.1, params_star[5] - 4*sigmas_star[5])

    # PLANETS
    planet_mass_max = 30.0
    planet_mass_min = 0.5
    luminosity_max = 3.35E23
    luminosity_min = 1.0E18

    if with_gas:
        w_gas_max = 0.5
        w_gas_min = 1.0E-6
    else:
        w_gas_max = 0.0
        w_gas_min = 0.0

    if with_water:
        ws_water_max = 0.5
    else:
        ws_water_max = 0.0

    if water_mixing_option == 'water_from_gas':
        ws_water_max = 0.0

# Generate table of stars
# ----------------------------------------------------------
def generate_stars(npts, params_star, sigmas_star, debug_star_generation=False):
    """ Generate random array with stars
    
    Output:     3d array with axes:
                    axis 0: empty (for consistency)  
                    axis 1: accepted generated stars  
                    axis 2: stellar properties (M_star, R_star, Teff_star, SiFe_star, MgFe_star)  

    Args:
        npts:                       Number of stars to be generated initially
        params_star:                Stellar parameters (0 - M_star, 1 - R_star, 2 - Teff_star, 3 - SiFe_star, 4 - MgFe_star, 5 - age_star)
        sigmas_star:                Sigmas of stellar parameters
        debug_star_generation:      Debug option (print number of stars and array size)
    """

    if debug_star_generation:
        print('')
        print('GENERATING STARS')
        print('')
        print('Original number of stars:   ' + str(npts))

    # Sample stellar mass
    M_star = np.random.normal(params_star[0], sigmas_star[0], npts).reshape(-1,1)        # 2d array with dimensions (Nstars, 1)
    index = np.where((M_star > M_star_min) & (M_star < M_star_max))                      # 2d array with indices where condition is true ([x1,x2,..], [0,0,..])
    M_star = M_star[index].reshape(-1,1)
    if debug_star_generation:
        print('After sampling M_star:      ' + str(M_star.shape[0]))

    # Sample stellar radius
    R_star = np.random.normal(params_star[1], sigmas_star[1], np.shape(M_star))
    index = np.where((R_star > R_star_min) & (R_star < R_star_max))
    R_star = R_star[index].reshape(-1,1)
    M_star = M_star[index].reshape(-1,1)
    if debug_star_generation:
        print('After sampling R_star:      ' + str(M_star.shape[0]))

    # Sample effective temperature of the star
    Teff_star = np.random.normal(params_star[2], sigmas_star[2], np.shape(R_star))
    index = np.where((Teff_star > Teff_star_min) & (Teff_star < Teff_star_max))
    Teff_star = Teff_star[index].reshape(-1,1)
    R_star = R_star[index].reshape(-1,1)
    M_star = M_star[index].reshape(-1,1)
    if debug_star_generation:
        print('After sampling Teff_star:   ' + str(Teff_star.shape[0]))

    # Sample Si/Fe ratio of the star
    SiFe_star = np.random.normal(params_star[3], sigmas_star[3], np.shape(Teff_star))
    index = np.where((SiFe_star > SiFe_star_min) & (SiFe_star < SiFe_star_max))
    SiFe_star = SiFe_star[index].reshape(-1,1)
    Teff_star = Teff_star[index].reshape(-1,1)
    R_star = R_star[index].reshape(-1,1)
    M_star = M_star[index].reshape(-1,1)
    if debug_star_generation:
        print('After sampling SiFe_star:  ' + str(M_star.shape[0]))

    # Sample Mg/Fe ratio of the star
    MgFe_star = np.random.normal(params_star[4], sigmas_star[4], np.shape(SiFe_star))
    index = np.where((MgFe_star > MgFe_star_min) & (MgFe_star < MgFe_star_max))
    MgFe_star = MgFe_star[index].reshape(-1,1)
    SiFe_star = SiFe_star[index].reshape(-1,1)
    Teff_star = Teff_star[index].reshape(-1,1)
    R_star = R_star[index].reshape(-1,1)
    M_star = M_star[index].reshape(-1,1)
    if debug_star_generation:
        print('After sampling MgFe_star:  ' + str(M_star.shape[0]))

    # Sample stellar age
    age_star = np.random.normal(params_star[5], sigmas_star[5], np.shape(M_star))
    index = np.where((age_star > age_star_min) & (age_star < age_star_max))
    age_star = age_star[index].reshape(-1,1)
    MgFe_star = MgFe_star[index].reshape(-1,1)
    SiFe_star = SiFe_star[index].reshape(-1,1)
    Teff_star = Teff_star[index].reshape(-1,1)
    R_star = R_star[index].reshape(-1,1)
    M_star = M_star[index].reshape(-1,1)
    if debug_star_generation:
        print('After sampling R_star:      ' + str(M_star.shape[0]))
    
    # Concatenate different arrays and add empty new axis
    stars = np.concatenate([M_star,R_star,Teff_star,SiFe_star,MgFe_star,age_star],axis=1)
    stars = stars[np.newaxis, :, :]
    if debug_star_generation:
        print('Final array shape:          ' + str(stars.shape))
        print('')
    
    return stars

# Sample planet mass
# ----------------------------------------------------------
def select(array, index):
    """ Helper function to only keep the rows (axis=1) of a 3d array where a condition holds true  
        
        Args:
            array:  Array that operation should be perfomed on
            index:  Boolean array with same length as array, tells us where in array condition holds true
    """
    
    nkept = np.sum(index)                                                # count number of True values in array (True=1, False=0)
    length1 = np.shape(array)[0]
    length2 = np.shape(array)[-1]
    new_array = np.zeros((length1,nkept,length2))
    
    for i1 in range(length1):
        for i2 in range(length2):
            # select only rows along axis=1 where condition is true
            temp_array = np.squeeze(array[i1,:,i2].reshape(1,-1,1),axis=(0,2))[index]    # np.squeeze: get rid of 1d axes (here 0 and 2)
            temp_array = temp_array[np.newaxis,:]
            temp_array = temp_array[:,:,np.newaxis]
            new_array[i1:i1+1,:,i2:i2+1] = temp_array
            
    array = np.copy(new_array)
    return array

def sample_planet_mass(parameters_target, sigmas_target, stars, nplanets, comp_option_mass):
    """ Generate array with planets (as many as there are stars in stars) and sample planet masses
    
    Output:     3d array with axes:
                    axis 0 - different planets in the system (total: nplanets_system)
                    axis 1 - accepted generated planets
                    axis 2 - planetary properties (only mass after this function)

    Args:
        params_target:              Planetary parameters (for each planet: 0 - mass, 1 - radius, 2 - period)
        sigmas_target:              Sigmas of planetary parameters
        stars:                      3d array (axis 0: empty, axis 1: accepted generated stars, axis 2: stellar properties (M_star, R_star, Teff_star, SiFe_star, MgFe_star))
        nplanets:                   Number of planets in the system
        comp_option_mass:           provided observables for mass (0 - relative mass, 1 - planet mass)
    """

    planet_list = []

    for i in range(nplanets): # different planets in the system
        if comp_option_mass == 0:    # compare relative planet masses
            relativeM_target = parameters_target[i,0]
            sigma_relativeM = sigmas_target[i,0]
            relative_mass = np.random.normal(relativeM_target, sigma_relativeM, np.shape(stars)[1])
            M_star = stars[0,:,0]
            planet_mass = relative_mass * M_star**(2/3) * Mjup_Mearth
        
        elif comp_option_mass == 1:  # compare absolute planet masses
            planet_mass_target = parameters_target[i,0]
            sigma_planet_mass = sigmas_target[i,0]
            planet_mass = np.random.normal(planet_mass_target, sigma_planet_mass, np.shape(stars)[1])
        
        else:
            print('Invalid option: Mass comparison (' + comp_option_mass + ')')
            exit()

        planet_mass = planet_mass.reshape(-1,1)
        planet_list.append(planet_mass)
    
    planets = np.array(np.stack([planet_list[i] for i in range(nplanets)]))
    
    for i in range(nplanets):
        planet_mass = planets[i,:,0]
        index = (planet_mass > planet_mass_min) & (planet_mass < planet_mass_max)
        stars = select(stars,index)
        planets = select(planets,index)

    return planets, stars

# Sample mass fractions of different layers
# ----------------------------------------------------------
def sample_simplex_coordinates(planets, stars, nplanets):
    """ Sample simplex coordinates for composition for planets in planets array
    
    Output:     3d array with axes:
                    axis 0 - different planets in the system (total: nplanets_system)
                    axis 1 - accepted generated planets
                    axis 2 - planetary properties (mass, X, Y)

    Args:
        planets:        Previously generated 3d array with planet masses
        stars:          3d array (axis 0: empty, axis 1: accepted generated stars, axis 2: stellar properties (M_star, R_star, Teff_star, SiFe_star, MgFe_star))
        nplanets:       Number of planets in the system
    """
    
    planet_list = []
    for i in range(nplanets): # different planets in the system
        planet = planets[i,:,:]
        X = np.random.random(np.shape(planet)).reshape(-1,1) * 2**0.5
        Y = np.random.random(np.shape(planet)).reshape(-1,1) * ws_water_max * (3/2)**0.5
        planet_list.append(np.concatenate([planet,X,Y],axis=1))
    
    planets = np.stack([planet_list[i] for i in range(nplanets)])
    
    for i in range(nplanets): # different planets in the system
        X = planets[i,:,1]
        Y = planets[i,:,2]
        index = (Y <= X * 3**0.5) & (Y <= -X * 3**0.5 + 6**0.5)
        stars = select(stars,index)
        planets = select(planets,index)
        
    return planets, stars

def convert(X, Y):
    """ convert from cartesian coordinates (X, Y) to ternary coordinates on simplex (x, y, z)"""
    x = (2 - X * 2**0.5 - Y * (2/3)**0.5) / 2
    if Y.all() == 0.0:
        y = 1 -x
    else:
        y = (X * 2**0.5 - Y * (2/3)**0.5) / 2
    if Y.all() == 0.0:
        z = np.zeros(X.shape)
    else:
        z = 1 -x -y

    return x, y, z

def invert(x, y, z):
    """ convert back from ternary coordinates on simplex (x, y, z) to cartesian coordinates (X, Y)""" 
    X = (1/2**0.5) * (1-x+y)
    Y = (3/2)**0.5 * (1-x-y)
    return X, Y

def sample_mass_fractions(planets, stars, nplanets):
    """ compute layer mass fractions for the solid part of the planet

    Output:     3d array with axes:
                    axis 0 - different planets in the system (total: nplanets_system)
                    axis 1 - accepted generated planets
                    axis 2 - planetary properties (mass, X, Y, ws_core, ws_mantle, ws_water, x_Fe_core, x_S_core)

    Args:
        planets:        Previously generated 3d array with planet properties
        stars:          3d array (axis 0: empty, axis 1: accepted generated stars, axis 2: stellar properties (M_star, R_star, Teff_star, SiFe_star, MgFe_star))
        nplanets:       Number of planets in the system
    """
    
    planet_list = []
    for i in range(nplanets): # different planets in the system
        planet = planets[i,:,:]
        X1 = planet[:,1]
        Y1 = planet[:,2]
        
        # sampling core composition (Fe and S molar fractions)
        x_Fe_core = np.random.random(np.shape(X1)) * 0.19 + 0.81
        x_S_core = 1 - x_Fe_core
        
        # convert sampled X and Y values to layer mass fractions (with respect to solid planet)
        ws_core, ws_mantle, ws_water = convert(X1,Y1)

        planet = np.concatenate([planet, ws_core.reshape(-1,1), ws_mantle.reshape(-1,1), ws_water.reshape(-1,1), x_Fe_core.reshape(-1,1), x_S_core.reshape(-1,1)], axis=1)
        planet_list.append(planet)
        
    planets = np.stack([planet_list[i] for i in range(nplanets)])
    
    for i in range(nplanets): # different planets in the system
        ws_water = planets[i,:,5]
        index = ws_water <= ws_water_max
        stars = select(stars,index)
        planets = select(planets,index)

    return planets, stars

def sample_gas_fraction(planets, nplanets, use_log_prior_for_gas_mass, with_gas):
    """ sample layer mass fraction for H/He layer

    Output:     3d array with axes:
                    axis 0 - different planets in the system (total: nplanets_system)
                    axis 1 - accepted generated planets
                    axis 2 - planetary properties (mass, X, Y, ws_core, ws_mantle, ws_water, x_Fe_core, x_S_core, w_gas)

    Args:
        planets:                        Previously generated 3d array with planet properties
        nplanets:                       Number of planets in the system
        use_log_prior_for_gas_mass:     Prior option for gas mass (log-uniform or uniform)
    """
    
    planet_list = []
    for i in range(nplanets): # different planets in the system
        planet = planets[i,:,:]
        planet_mass = planet[:,0]
        
        if with_gas:
            if use_log_prior_for_gas_mass:
                lower_boundary_w_gas = np.ones(np.shape(planet_mass)) * np.log10(w_gas_min)
                upper_boundary_w_gas = np.ones(np.shape(planet_mass)) * np.log10(w_gas_max)
            else:
                lower_boundary_w_gas = np.ones(np.shape(planet_mass)) * w_gas_min
                upper_boundary_w_gas = np.ones(np.shape(planet_mass)) * w_gas_max

            w_gas = np.random.random(np.shape(planet_mass))
            range_w_gas = upper_boundary_w_gas - lower_boundary_w_gas
            w_gas = lower_boundary_w_gas + w_gas * range_w_gas

            if use_log_prior_for_gas_mass == True:
                w_gas = 10.**(w_gas)

        else:
            w_gas = np.zeros(np.shape(planet_mass))

        planet = np.concatenate([planet, w_gas.reshape(-1,1)], axis=1)
        planet_list.append(planet)
    
    planets = np.stack([planet_list[i] for i in range(nplanets)])    

    return planets

# Sample luminosity
# ----------------------------------------------------------
def luminosity_from_stellar_age(M_core, M_envelope, age_star):
    """ calculate luminosity of low mass planets from stellar age using fit from Mordasini 2020

    Output:     estimated planet luminosity

    Args:
        M_core:             Core mass in Earth masses
        M_envelope:         Envelope mass in Earth masses
        age_star:           Sampled age of the star
    """

    # params: [time[Gyr],a0,b1,b2,c1,c2]
    params = np.array([[0.1,0.002529060,-0.002002380,0.001044080,0.05864850, 0.000967878],
                       [0.3,0.001213240,-0.000533601,0.000360703,0.02141140, 0.000368533],
                       [0.5,0.000707416,-0.000394131,0.000212475,0.01381380, 0.000189456],
                       [0.8,0.000423376,-0.000187283,0.000125872,0.00887292, 0.000117141],
                       [1.0,0.000352187,-0.000141480,9.94382E-5 ,0.00718831, 9.20563E-5],
                       [2.0,0.000175775,-4.07832E-5, 4.58530E-5, 0.00357941, 5.52851E-5],
                       [3.0,0.000114120,-2.09944E-5, 2.91169E-5, 0.00232693, 4.00546E-5],
                       [4.0,8.81462E-5, -2.07673E-5, 2.12932E-5, 0.00171412, 2.90984E-5],
                       [5.0,6.91819E-5, -1.90159E-5, 1.62128E-5, 0.00134355, 2.30387E-5],
                       [6.0,5.49615E-5, -1.68620E-5, 1.29045E-5, 0.00109019, 1.96163E-5],
                       [7.0,4.50320E-5, -1.51951E-5, 1.05948E-5, 0.00091005, 1.70934E-5],
                       [8.0,3.80363E-5, -1.40113E-5, 8.93639E-6, 0.00077687, 1.50107E-5],
                       [9.0,3.30102E-5, -1.31146E-5, 7.69121E-6, 0.000675243,1.32482E-5],
                       [10.0,2.92937E-5,-1.24023E-5, 6.73922E-6, 0.000595191,1.17809E-5]])

    # do interpolation of table
    interpol_a0 = interpolate.interp1d(params[:,0], params[:,1])
    interpol_b1 = interpolate.interp1d(params[:,0], params[:,2])
    interpol_b2 = interpolate.interp1d(params[:,0], params[:,3])
    interpol_c1 = interpolate.interp1d(params[:,0], params[:,4])
    interpol_c2 = interpolate.interp1d(params[:,0], params[:,5])

    age_truncated = age_star

    age_truncated[age_truncated >= 10] = 10
    age_truncated[age_truncated <= 0.1] = 0.1

    a0 = interpol_a0(age_truncated)
    b1 = interpol_b1(age_truncated)
    b2 = interpol_b2(age_truncated)
    c1 = interpol_c1(age_truncated)
    c2 = interpol_c2(age_truncated)

    # do luminosity fit
    luminosity = (a0 + b1 * M_core + b2 * M_core**2 + c1 * M_envelope + c2 * M_envelope**2) * 3.35e24 #erg/s

    return luminosity

def sample_luminosity(planets, stars, nplanets, luminosity_option):
    """ sample planet luminosity

    Output:     3d array with axes:
                    axis 0 - different planets in the system (total: nplanets_system)
                    axis 1 - accepted generated planets
                    axis 2 - planetary properties (mass, X, Y, ws_core, ws_mantle, ws_water, x_Fe_core, x_S_core, w_gas, luminosity)

    Args:
        planets:                        Previously generated 3d array with planet properties
        stars:                          3d array (axis 0: empty, axis 1: accepted generated stars, axis 2: stellar properties (M_star, R_star, Teff_star, SiFe_star, MgFe_star))
        nplanets:                       Number of planets in the system
        luminosity_option:              Prior option for luminosity (log-uniform or uniform)
    """
    
    planet_list = []
    for i in range(nplanets): # different planets in the system
        planet = planets[i,:,:]
        planet_mass = planet[:,0]
        
        if luminosity_option == 'log_sampling':
            lower_boundary_luminosity = np.ones(np.shape(planet_mass)) * np.log10(luminosity_min)
            upper_boundary_luminosity = np.ones(np.shape(planet_mass)) * np.log10(luminosity_max)

            luminosity = np.random.random(np.shape(planet_mass))
            range_luminosity = upper_boundary_luminosity - lower_boundary_luminosity
            luminosity = lower_boundary_luminosity + luminosity * range_luminosity
            luminosity = 10.**(luminosity)

        elif luminosity_option == 'uniform_sampling':
            lower_boundary_luminosity = np.ones(np.shape(planet_mass)) * luminosity_min
            upper_boundary_luminosity = np.ones(np.shape(planet_mass)) * luminosity_max

            luminosity = np.random.random(np.shape(planet_mass))
            range_luminosity = upper_boundary_luminosity - lower_boundary_luminosity
            luminosity = lower_boundary_luminosity + luminosity * range_luminosity

        elif luminosity_option == 'from_stellar_age':
            w_gas = planet[:,8]
            age_star = stars[0,:,5]

            M_core = planet_mass * (1-w_gas)
            M_envelope = planet_mass * w_gas

            luminosity = luminosity_from_stellar_age(M_core, M_envelope, age_star)

        else:
            print('Invalid luminosity option: (' + luminosity_option + ')')
            exit()
            
        planet = np.concatenate([planet, luminosity.reshape(-1,1)], axis=1)
        planet_list.append(planet)
    
    planets = np.stack([planet_list[i] for i in range(nplanets)])    

    return planets

# Sample period
# ----------------------------------------------------------
def sample_period(planets, parameters_target, sigmas_target, stars, nplanets):
    """ sample period of the planet

    Output:     3d array with axes:
                    axis 0 - different planets in the system (total: nplanets_system)
                    axis 1 - accepted generated planets
                    axis 2 - planetary properties (mass, X, Y, ws_core, ws_mantle, ws_water, x_Fe_core, x_S_core, w_gas, luminosity, period)

    Args:
        planets:            Previously generated 3d array with planet properties
        params_target:      Planetary parameters (for each planet: 0 - mass, 1 - radius, 2 - period)
        sigmas_target:      Sigmas of planetary parameters
        stars:              3d array (axis 0: empty, axis 1: accepted generated stars, axis 2: stellar properties (M_star, R_star, Teff_star, SiFe_star, MgFe_star))
        nplanets:           Number of planets in the system
    """
    
    planet_list = []
    for i in range(nplanets): # different planets in the system
        planet = planets[i,:,:]
        period_target = parameters_target[i,2]
        sigma_period = sigmas_target[i,2]
        
        period = np.random.normal(period_target, sigma_period, np.shape(stars)[1]).reshape(-1,1)
        planet = np.concatenate([planet, period.reshape(-1,1)], axis=1)
        planet_list.append(planet)
    
    planets = np.array(np.stack([planet_list[i] for i in range(nplanets)]))

    return planets

# Sample Si/Mg/Fe ratios and mantle composition
# ----------------------------------------------------------
def sample_SiMgFe_ratios(planets, stars, nplanets, SiMgFe_ratio_option):
    """ sample Si/Mg/Fe ratios of the planet

    Output:     3d array with axes:
                    axis 0 - different planets in the system (total: nplanets_system)
                    axis 1 - accepted generated planets
                    axis 2 - planetary properties (mass, X, Y, ws_core, ws_mantle, ws_water, x_Fe_core, x_S_core, w_gas, luminosity, period, SiFe_planet, MgFe_planet)

    Args:
        planets:                Previously generated 3d array with planet properties
        stars:                  3d array (axis 0: empty, axis 1: accepted generated stars, axis 2: stellar properties (M_star, R_star, Teff_star, SiFe_star, MgFe_star))
        nplanets:               Number of planets in the system
        SiMgFe_ratio_option:    
    """
    
    planet_list = []
    for i in range(nplanets): # different planets in the system
        planet = planets[i,:,:]
        SiFe = stars[0,:,3]
        MgFe = stars[0,:,4]
        period = planet[:,9]
        
        if SiMgFe_ratio_option == 0:        # Thiabaud et al. 2015 (stellar)
            SiFe_planet = SiFe
            MgFe_planet = MgFe

        elif SiMgFe_ratio_option == 1:      # Adibekyan et al. 2021 (steeper slope)
            a = np.random.normal(4.84, 0.92, np.shape(period))
            b = np.random.normal(-1.35, 0.36, np.shape(period))
            SiplusMg_Fe_planet = 1./(b + a * (1./(SiFe + MgFe)))

            SiMg_planet = SiFe/MgFe
            MgFe_planet = SiplusMg_Fe_planet/(1+SiMg_planet)
            SiFe_planet = SiplusMg_Fe_planet - MgFe_planet

        elif SiMgFe_ratio_option == 2:      # Unconstrained (sample mantle composition uniformly from simplex)
            X = np.random.random(np.shape(period)) * 2**0.5
            Y = np.random.random(np.shape(period)) * (3/2)**0.5 * 0.75

        else:                               # invalid option
            print('Invalid option: Si/Mg/Fe ratios (' + SiMgFe_ratio_option + ')')
            exit()

        if SiMgFe_ratio_option == 0 or SiMgFe_ratio_option == 1:
            planet = np.concatenate([planet, SiFe_planet.reshape(-1,1), MgFe_planet.reshape(-1,1)], axis=1)
        else: # SiMgFe_ratio_option == 2
            planet = np.concatenate([planet, X.reshape(-1,1), Y.reshape(-1,1)], axis=1)
        planet_list.append(planet)
        
    planets = np.stack([planet_list[i] for i in range(nplanets)])
    
    if SiMgFe_ratio_option == 0 or SiMgFe_ratio_option == 1:
        for i in range(nplanets):
            SiFe_planet = planets[i,:,11]
            MgFe_planet = planets[i,:,12]
            index = (SiFe_planet > 0) & (MgFe_planet > 0)
            stars = select(stars,index)
            planets = select(planets,index)

    else: # SiMgFe_ratio_option == 2
        for i in range(nplanets): # different planets in the system
            X = planets[i,:,11]
            Y = planets[i,:,12]

            index = (Y <= X * 3**0.5) & (Y <= -X * 3**0.5 + 6**0.5)
            stars = select(stars,index)
            planets = select(planets,index)

    return planets, stars

def mantle_composition_analytical(w_core, w_mantle, x_Fe_core, SiFe_bulk, MgFe_bulk):
#     """ calculate mantle composition from bulk ratios 
#         Note: everything is per unit mass
#     """

    # molar masses
    mu_SiO2 = 28.085 + 2 * 15.9994
    mu_MgO = 24.305 + 15.9994
    mu_Fe = 55.845
    mu_FeO = 55.845 + 15.9994
    mu_S = 32.0650

    mu_core = mu_Fe*x_Fe_core + mu_S*(1-x_Fe_core)

#     mu_core = mu_Fe*x_Fe_core + mu_S*(1-x_Fe_core)
#     N_Fe_core = w_core * x_Fe_core / mu_core

#     # System of equations: 
#     # (1) Expressing (Si/Fe)bulk through x_Fe_mantle, x_Si_mantle, x_Mg_mantle:     a1 * x_Fe_mantle + a2 * x_Si_mantle + a3 * x_Mg_mantle = 0
#     # (2) Expressing (Mg/Fe)bulk through x_Fe_mantle, x_Si_mantle, x_Mg_mantle:     b1 * x_Fe_mantle + b2 * x_Si_mantle + b3 * x_Mg_mantle = 0
#     # (3) The mass fractions need to add up to 1:                                        x_Fe_mantle +      x_Si_mantle +      x_Mg_mantle = 1

#     a1 = SiFe_bulk * (w_mantle + N_Fe_core * mu_FeO)
#     a2 = MgFe_bulk * (w_mantle + N_Fe_core * mu_FeO)
#     a3 = 1

#     b1 = SiFe_bulk * N_Fe_core * mu_SiO2 - w_mantle
#     b2 = MgFe_bulk * N_Fe_core * mu_SiO2
#     b3 = 1

#     c1 = SiFe_bulk * N_Fe_core * mu_MgO
#     c2 = MgFe_bulk * N_Fe_core * mu_MgO - w_mantle
#     c3 = 1

#     # Solve system of equations
#     A = np.array(np.stack([np.stack([a1, b1, c1]), np.stack([a2, b2, c2]), np.stack([a3, b3, c3])]))
#     b = np.array([0, 0, 1])

#     x = np.linalg.solve(A,b)

#     x_Si_mantle = x[1]
#     x_Mg_mantle = x[2]
#     # x_Fe_mantle = x[0]

#     return x_Si_mantle, x_Mg_mantle

    alpha = 1
    lam = MgFe_bulk/SiFe_bulk
    beta = -(1+lam)
    a1 = beta + (w_core/w_mantle)*(x_Fe_core/mu_core)*(mu_SiO2+beta*mu_FeO+lam*mu_MgO)
    a2 = alpha*(1+(w_core/w_mantle)*(x_Fe_core/mu_core)*mu_FeO)
    si_mantle = a2*SiFe_bulk/(1-a1*SiFe_bulk)
    mg_mantle = lam*si_mantle

    return si_mantle,mg_mantle

def compute_molar_ratio(ws_core, ws_mantle, x_Si_mantle, x_Mg_mantle, x_Fe_core):
    """ compute Si/Mg/Fe bulk ratios from sampled layer mass and molar fractions """
    
    x_S_core = 1 - x_Fe_core
    x_Fe_mantle = 1 - x_Si_mantle - x_Mg_mantle
    ws_water = 1 - ws_core - ws_mantle
    
    mu_Si = 28.085
    mu_SiO2 = 28.085+2*15.9994
    mu_Mg = 24.305
    mu_MgO = 24.305+15.9994
    mu_Fe = 55.845
    mu_FeO = 55.845+15.9994
    mu_S = 32.0650

    # calculate number of relative number of atoms in layers
    mu_mantle = mu_SiO2 * x_Si_mantle + mu_FeO * x_Fe_mantle + mu_MgO * x_Mg_mantle
    N_mantle = ws_mantle / mu_mantle   # per mass unit

    mu_core = mu_Fe * x_Fe_core + mu_S * x_S_core
    N_core = ws_core / mu_core         # per mass unit

    # for single species
    N_Si_mantle = N_mantle * x_Si_mantle
    N_Mg_mantle = N_mantle * x_Mg_mantle
    N_Fe_mantle = N_mantle * x_Fe_mantle
    N_Fe_core = N_core * x_Fe_core

    # calculate ratios
    MgFe_bulk = N_Mg_mantle / (N_Fe_mantle + N_Fe_core)
    SiFe_bulk = N_Si_mantle / (N_Fe_mantle + N_Fe_core)
    
    return SiFe_bulk, MgFe_bulk

def compute_mantle_composition(planets, stars, nplanets, SiMgFe_ratio_option):
    """ compute mantle composition

    Output:     3d array with axes:
                    axis 0 - different planets in the system (total: nplanets_system)
                    axis 1 - accepted generated planets
                    axis 2 - planetary properties (mass, X, Y, ws_core, ws_mantle, ws_water, x_Fe_core, x_S_core, w_gas, luminosity, period, shift_SiFe_planet, shift_MgFe_planet, SiFe_planet, MgFe_planet, x_Si_mantle, x_Mg_mantle, x_Fe_mantle)

    Args:
        planets:        Previously generated 3d array with planet properties
        stars:          3d array (axis 0: empty, axis 1: accepted generated stars, axis 2: stellar properties (M_star, R_star, Teff_star, SiFe_star, MgFe_star))
        nplanets:       Number of planets in the system
    """
    
    planet_list = []
    for i in range(nplanets): # different planets in the system
        planet = planets[i,:,:]
        ws_core = planet[:,3]
        ws_mantle = planet[:,4]
        x_Fe_core = planet[:,6]

        if SiMgFe_ratio_option == 2: # first calculate SiFe_planet and MgFe_planet from X and Y sampled in last step
            X = planet[:,11]
            Y = planet[:,12]

            x_Si_planet, x_Mg_planet, x_Fe_planet = convert(X,Y)
            planets[i,:,11] = x_Si_planet/x_Fe_planet # SiFe_planet
            planets[i,:,12] = x_Mg_planet/x_Fe_planet # MgFe_planet
            planet = planets[i,:,:]

        SiFe_planet = planet[:,11]
        MgFe_planet = planet[:,12]
        
        x_Si_mantle, x_Mg_mantle = mantle_composition_analytical(ws_core, ws_mantle, x_Fe_core, SiFe_planet, MgFe_planet)
        x_Fe_mantle = 1 - x_Si_mantle - x_Mg_mantle

    planet = np.concatenate([planet, x_Si_mantle.reshape(-1,1), x_Mg_mantle.reshape(-1,1), x_Fe_mantle.reshape(-1,1)], axis=1)
    planet_list.append(planet)
        
    planets = np.stack([planet_list[i] for i in range(nplanets)])
    
    if SiMgFe_ratio_option == 2:
        for i in range(nplanets):
            SiFe_planet = planets[i,:,11]
            MgFe_planet = planets[i,:,12]
            index = (SiFe_planet > 0) & (MgFe_planet > 0)
            stars = select(stars,index)
            planets = select(planets,index)

    for i in range(nplanets): # different planets in the system
        x_Si_mantle = planets[i,:,13]
        index =  (x_Si_mantle >= 0) & (x_Si_mantle <= 1)
        stars = select(stars, index)
        planets = select(planets, index)

    for i in range(nplanets): # different planets in the system
        x_Mg_mantle = planets[i,:,14]
        index =  (x_Mg_mantle >= 0) & (x_Mg_mantle <= 1)
        stars = select(stars, index)
        planets = select(planets, index)  

    for i in range(nplanets): # different planets in the system
        x_Fe_mantle = planets[i,:,15]
        index =  (x_Fe_mantle >= 0) & (x_Fe_mantle <= 1)
        stars = select(stars, index)
        planets = select(planets, index)

    return planets, stars

# Calculate total radius and compare to observations
# ----------------------------------------------------------
def compute_a(M_star, P_planet):
    """ compute semimajor axis from mass of star and period of the planet

    Output:
        a [m]

    Args:
        M_star [Mjup]
        P_planet [day]
    """
    
    return (constG*M_star*(Msun/Msun_Mjup)/(2*constPI/(P_planet*day))**2)**(1/3)

def compute_Teq(Teff_star, R_star, M_star, P_planet):
    """ compute equilibrium temperature of the planet 
    
    Output:
        Teq [K]

    Args:
        Teff_star [K]
        R_star [Rsun]
        M_star [Mjup]
        P_planet [day]
    """
    
    a = compute_a(M_star, P_planet)
    
    return Teff_star * (R_star*Rsun/(2*a))**0.5

def compute_Hill_radius(M_star, m_planet, P_planet):
    """ compute Hill radius of the planet 
    
    Output:
        R_H [Rearth]

    Args:
        M_star [Mjup]
        m_planet [Mearth]
        P_planet [day]
    """

    a = compute_a(M_star, P_planet)

    return (m_planet/(3.*M_star*Mjup_Mearth))**(1./3.) * a/Rearth

def compute_Bondi_radius(m_planet, Teq, Z_atmo):
    """ compute Bondi radius of the planet (assumes ideal gas law)
    
    Output:
        R_B [Rearth]

    Args:
        m_planet [Mearth]
        Teq [K]
        Z_atmo [weight mass fraction]
    """

    Z_atmo_mol = (Z_atmo/mu_H2O) / (Z_atmo/mu_H2O + (1-Z_atmo)/mu_HHe)
    gamma = Z_atmo_mol * 8./6. + (1-Z_atmo_mol) * 5./3.
    mu = Z_atmo_mol * mu_H2O + (1-Z_atmo_mol) * mu_HHe

    return (gamma-1)/gamma * (constG*m_planet*Mearth*mu)/(const_kb*Teq*Rearth)

def sample_Teq_and_Zatmo(planets, stars, nplanets, water_mixing_option):
    planet_list = []
    for i in range(nplanets): # different planets in the system
        planet = planets[i,:,:]

        M_star = stars[0,:,0]
        R_star = stars[0,:,1]
        Teff_star = stars[0,:,2]
        #SiFe_star = stars[0,:,3]
        #MgFe_star = stars[0,:,4]

        mass_planet = planet[:,0]
        X = planet[:,1]
        Y = planet[:,2]
        ws_core = planet[:,3]
        ws_mantle = planet[:,4]
        ws_water = planet[:,5]
        #x_Fe_core = planet[:,6]
        x_S_core = planet[:,7]
        w_gas = planet[:,8]
        luminosity = planet[:,9]
        period = planet[:,10]
        #SiFe_planet = planet[:,11]
        #MgFe_planet = planet[:,12]
        x_Si_mantle = planet[:,13]
        x_Mg_mantle = planet[:,14]
        #x_Fe_mantle = planet[:,15]

        w_core = ws_core * (1-w_gas)
        w_mantle = ws_mantle * (1-w_gas)
        w_water = ws_water * (1-w_gas)
        Teq = compute_Teq(Teff_star, R_star, M_star, period)

        if water_mixing_option == 'uniform':
            Z_atmo = w_water/(w_water+w_gas)
        else: #water_mixing_option == 'water_from_gas':
            Z_atmo = np.random.normal(0.005, 0.0025, np.shape(Teq))
            w_water = w_gas * Z_atmo
            planet[:,8] = w_gas * (1-Z_atmo)
    
    planet = np.concatenate([planet, w_core.reshape(-1,1), w_mantle.reshape(-1,1), w_water.reshape(-1,1), Teq.reshape(-1,1), Z_atmo.reshape(-1,1)], axis=1)
    planet_list.append(planet)
    
    planets = np.stack([planet_list[i] for i in range(nplanets)])

    for i in range(nplanets): # different planets in the system
        Z_atmo = planets[i,:,19]
        index =  (Z_atmo > 0)
        stars = select(stars, index)
        planets = select(planets, index)

    return planets, stars

def compute_radius(planets, stars, parameters_target, sigmas_target, nplanets, mass_range, dnn_prediction, scaler_x_prediction, scaler_y_prediction, comp_option_radius, water_mixing_option):
    """ compute total radius of structure using DNN and compare to observations

    Output:     3d array with axes:
                    axis 0 - different planets in the system (total: nplanets_system)
                    axis 1 - accepted generated planets
                    axis 2 - planetary properties (mass, X, Y, ws_core, ws_mantle, ws_water, x_Fe_core, x_S_core, w_gas, luminosity, period, SiFe_planet, MgFe_planet, x_Si_mantle, x_Mg_mantle, x_Fe_mantle)

    Args:
        planets:                Previously generated 3d array with planet properties
        stars:                  3d array (axis 0: empty, axis 1: accepted generated stars, axis 2: stellar properties (M_star, R_star, Teff_star, SiFe_star, MgFe_star))
        params_target:          Planetary parameters (for each planet: 0 - mass, 1 - radius, 2 - period)
        sigmas_target:          Sigmas of planetary parameters
        nplanets:               Number of planets in the system
        dnn_prediction:         
        scaler_x:               
        scaler_y:               
        comp_option_radius:     provided observables for radius (0 - transit depth, 1 - radius ratio, 2 - planet radius)
        water_mixing_option:
    """

    nmodels = np.shape(planets)[1]
    planet_list = []
    for j in range(nplanets):
        planet = planets[j,:,:]
        additional_data = np.zeros((nmodels,5))
        planet = np.concatenate([planet,additional_data],axis=1)
        planet_list.append(planet)
        
    planets_all = np.stack([planet_list[k] for k in range(nplanets)])

    for i in range(nplanets):
        planet = planets_all[i,:,:]

        M_star = stars[0,:,0]
        R_star = stars[0,:,1]
        #Teff_star = stars[0,:,2]
        #SiFe_star = stars[0,:,3]
        #MgFe_star = stars[0,:,4]

        mass_planet = planet[:,0]
        X = planet[:,1]
        Y = planet[:,2]
        #ws_core = planet[:,3]
        #ws_mantle = planet[:,4]
        #ws_water = planet[:,5]
        #x_Fe_core = planet[:,6]
        x_S_core = planet[:,7]
        w_gas = planet[:,8]
        luminosity = planet[:,9]
        period = planet[:,10]
        #SiFe_planet = planet[:,11]
        #MgFe_planet = planet[:,12]
        x_Si_mantle = planet[:,13]
        x_Mg_mantle = planet[:,14]
        #x_Fe_mantle = planet[:,15]
        w_core = planet[:,16]
        #w_mantle = planet[:,17]
        w_water = planet[:,18] 
        Teq = planet[:,19] 
        Z_atmo = planet[:,20]

        if water_mixing_option == 'separate':
            parameters_unscaled = np.concatenate([Teq.reshape(-1,1), np.log10(luminosity.reshape(-1,1)), mass_planet.reshape(-1,1), w_core.reshape(-1,1), w_water.reshape(-1,1), np.log10(w_gas.reshape(-1,1)), x_S_core.reshape(-1,1), x_Si_mantle.reshape(-1,1), x_Mg_mantle.reshape(-1,1)], axis=1)
        else: #water_mixing_option == 'uniform' or water_mixing_option == 'water_from_gas':
            parameters_unscaled = np.concatenate([Teq.reshape(-1,1), Z_atmo.reshape(-1,1), np.log10(luminosity.reshape(-1,1)), mass_planet.reshape(-1,1), w_core.reshape(-1,1), np.log10((w_gas+w_water).reshape(-1,1)), x_S_core.reshape(-1,1), x_Si_mantle.reshape(-1,1), x_Mg_mantle.reshape(-1,1)], axis=1)

        parameters_scaled_prediction = scaler_x_prediction[int(mass_range)].transform(parameters_unscaled)
        Rtot = dnn_prediction[int(mass_range)].predict(parameters_scaled_prediction)
        Rtot = scaler_y_prediction[int(mass_range)].inverse_transform(Rtot)
        transit_depth = (Rtot*Rearth/(R_star.reshape(-1,1)*Rsun))**2*1.e6

        if comp_option_radius == 0 or comp_option_radius == 1:
            transitdepth_target = parameters_target[i,1]
            sigma_transitdepth = sigmas_target[i,1]
            likelihood = np.exp(-0.5*(transit_depth-transitdepth_target)**2/sigma_transitdepth**2)
        elif comp_option_radius == 2:
            radius_target = parameters_target[i,1]
            sigma_radius = sigmas_target[i,1]
            likelihood = np.exp(-0.5*(Rtot-radius_target)**2/sigma_radius**2)
        else:
            print('Invalid option: Radius comparison (' + comp_option_radius + ')')
            exit()

        #print(np.min(Rtot), np.max(Rtot), np.median(Rtot))

        R_H = compute_Hill_radius(M_star, mass_planet, period)
        R_B = compute_Bondi_radius(mass_planet, Teq, Z_atmo)

        planets_all[i,:,-5:] = np.concatenate([transit_depth.reshape(-1,1), Rtot.reshape(-1,1), likelihood.reshape(-1,1), R_H.reshape(-1,1), R_B.reshape(-1,1)],axis=1)
        
        threshold = np.random.random(np.shape(transit_depth))
        index = np.squeeze(likelihood > threshold,axis=1)
        stars = select(stars,index)
        planets_all = select(planets_all,index)

    for i in range(nplanets): # different planets in the system
        Rtot = planets_all[i,:,22]
        R_H = planets_all[i,:,24]
        index = (Rtot < R_H)
        stars = select(stars,index)
        planets_all = select(planets_all,index)

    for i in range(nplanets): # different planets in the system
        Rtot = planets_all[i,:,22]
        R_B = planets_all[i,:,25]
        index = (Rtot < R_B)
        stars = select(stars,index)
        planets_all = select(planets_all,index)

    return planets_all,stars

# Run grid
# ----------------------------------------------------------
def generate_single_planet(parameters_target, sigmas_target, stars, iplanet, nplanets, comp_option_mass, luminosity_option, SiMgFe_ratio_option, water_mixing_option, mass_range, dnn_prediction, scaler_x_prediction, scaler_y_prediction, comp_option_radius, use_log_prior_for_gas_mass,with_gas,debug_planet_generation=False,prior=False):
    # sampling from priors/observations for 1 planet (with index iplanet) in the system (nplanets=1)
    # number of samples is given by stars.shape[1]

    # Planet parameters:
    # 0:   planet_mass
    # 1:   X (triangle coordinates)
    # 2:   Y (triangle coordinates)
    # 3:   ws_core  
    # 4:   ws_mantle  
    # 5:   ws_water  
    # 6:   x_Fe_core  
    # 7:   x_S_core  
    # 8:   w_gas  
    # 9:   luminosity  
    # 10:  period  
    # 11:  SiFe_planet  
    # 12:  MgFe_planet  
    # 13:  x_Si_mantle  
    # 14:  x_Mg_mantle  
    # 15:  x_Fe_mantle  
    # 16:  w_core
    # 17:  w_mantle
    # 18:  w_water
    # 19:  Teq
    # 20:  Z_atmo
    # 21:  transit_depth
    # 22:  Rtot
    # 23:  likelihood
    
    param_iplanet = parameters_target[iplanet,:].reshape(1,-1)
    sig_iplanet = sigmas_target[iplanet,:].reshape(1,-1)
    
    planets, stars = sample_planet_mass(param_iplanet,sig_iplanet,stars,nplanets,comp_option_mass)
    if debug_planet_generation:
        print('After sampling planetary mass:          ', np.shape(planets),np.shape(stars))
        
    planets, stars = sample_simplex_coordinates(planets,stars,nplanets)
    if debug_planet_generation:
        print('After sampling simplex coordinates:     ', np.shape(planets),np.shape(stars))
        
    planets, stars = sample_mass_fractions(planets,stars,nplanets)
    if debug_planet_generation:
        print('After sampling mass fractions:          ', np.shape(planets),np.shape(stars))
    
    planets = sample_gas_fraction(planets,nplanets,use_log_prior_for_gas_mass,with_gas)
    if debug_planet_generation:
        print('After sampling gas fraction:            ', np.shape(planets),np.shape(stars))
    
    planets = sample_luminosity(planets,stars,nplanets,luminosity_option)
    if debug_planet_generation:
        print('After sampling luminosity:              ', np.shape(planets),np.shape(stars))

    planets = sample_period(planets,param_iplanet,sig_iplanet,stars,nplanets)
    if debug_planet_generation:
        print('After sampling period:                  ', np.shape(planets),np.shape(stars))
    
    planets, stars = sample_SiMgFe_ratios(planets,stars,nplanets,SiMgFe_ratio_option)
    if debug_planet_generation:
        print('After sampling Si/Mg/Fe ratios:         ', np.shape(planets),np.shape(stars))
    
    planets, stars = compute_mantle_composition(planets,stars,nplanets,SiMgFe_ratio_option)
    if debug_planet_generation:
        print('After computing mantle composition:     ', np.shape(planets),np.shape(stars))

    planets, stars = sample_Teq_and_Zatmo(planets,stars,nplanets,water_mixing_option)
    if debug_planet_generation:
        print('After sampling Teq and Zatmo:           ', np.shape(planets),np.shape(stars))
    
    if not prior:
        planets, stars = compute_radius(planets,stars,param_iplanet,sig_iplanet,nplanets,mass_range[iplanet],dnn_prediction,scaler_x_prediction,scaler_y_prediction,comp_option_radius,water_mixing_option)
        if debug_planet_generation:
            print('After computing transit radius:         ', np.shape(planets),np.shape(stars))

    return planets, stars

def make_posteriors(planets,stars,nplanets):
    posterior_list= []
    for i in range(nplanets):
        planet = planets[i,:,:]

        M_star = stars[0,:,0].reshape(-1,1)
        R_star = stars[0,:,1].reshape(-1,1)
        Teff_star = stars[0,:,2].reshape(-1,1)
        SiFe_star = stars[0,:,3].reshape(-1,1)
        MgFe_star = stars[0,:,4].reshape(-1,1)
        age_star = stars[0,:,5].reshape(-1,1)

        mass_planet = planet[:,0].reshape(-1,1)
        X = planet[:,1].reshape(-1,1)
        Y = planet[:,2].reshape(-1,1)
        ws_core = planet[:,3].reshape(-1,1)
        ws_mantle = planet[:,4].reshape(-1,1)
        ws_water = planet[:,5].reshape(-1,1)
        x_Fe_core = planet[:,6].reshape(-1,1)
        x_S_core = planet[:,7].reshape(-1,1)
        w_gas = planet[:,8].reshape(-1,1)
        luminosity = planet[:,9].reshape(-1,1)
        period = planet[:,10].reshape(-1,1)
        SiFe_planet = planet[:,11].reshape(-1,1)
        MgFe_planet = planet[:,12].reshape(-1,1)
        x_Si_mantle = planet[:,13].reshape(-1,1)
        x_Mg_mantle = planet[:,14].reshape(-1,1)
        x_Fe_mantle = planet[:,15].reshape(-1,1)
        w_core = planet[:,16].reshape(-1,1)
        w_mantle = planet[:,17].reshape(-1,1)
        w_water = planet[:,18].reshape(-1,1)
        Teq = planet[:,19].reshape(-1,1)
        Z_atmo = planet[:,20].reshape(-1,1)
        transit_depth = planet[:,21].reshape(-1,1)
        Rtot = planet[:,22].reshape(-1,1)
        likelihood = planet[:,23].reshape(-1,1)
    
        posterior_list.append(np.concatenate([mass_planet,          # 0
                                              ws_core,              # 1
                                              ws_mantle,            # 2
                                              ws_water,             # 3
                                              w_core,               # 4
                                              w_mantle,             # 5
                                              w_water,              # 6
                                              w_gas,                # 7
                                              x_Fe_core,            # 8
                                              x_S_core,             # 9
                                              x_Si_mantle,          # 10
                                              x_Mg_mantle,          # 11
                                              x_Fe_mantle,          # 12
                                              SiFe_planet,          # 13
                                              MgFe_planet,          # 14
                                              luminosity,           # 15
                                              period,               # 16
                                              Teq,                  # 17
                                              Z_atmo,               # 18
                                              transit_depth,        # 19
                                              Rtot,                 # 20
                                              likelihood,           # 21
                                              M_star,               # 22
                                              R_star,               # 23
                                              Teff_star,            # 24
                                              SiFe_star,            # 25
                                              MgFe_star,            # 26
                                              age_star],axis=1))    # 27

    posteriors = np.stack([posterior_list[i] for i in range(nplanets)])

    return posteriors

def make_priors(planets,stars,nplanets):
    prior_list= []
    for i in range(nplanets):
        planet = planets[i,:,:]

        M_star = stars[0,:,0].reshape(-1,1)
        R_star = stars[0,:,1].reshape(-1,1)
        Teff_star = stars[0,:,2].reshape(-1,1)
        SiFe_star = stars[0,:,3].reshape(-1,1)
        MgFe_star = stars[0,:,4].reshape(-1,1)
        age_star = stars[0,:,5].reshape(-1,1)

        mass_planet = planet[:,0].reshape(-1,1)
        X = planet[:,1].reshape(-1,1)
        Y = planet[:,2].reshape(-1,1)
        ws_core = planet[:,3].reshape(-1,1)
        ws_mantle = planet[:,4].reshape(-1,1)
        ws_water = planet[:,5].reshape(-1,1)
        x_Fe_core = planet[:,6].reshape(-1,1)
        x_S_core = planet[:,7].reshape(-1,1)
        w_gas = planet[:,8].reshape(-1,1)
        luminosity = planet[:,9].reshape(-1,1)
        period = planet[:,10].reshape(-1,1)
        SiFe_planet = planet[:,11].reshape(-1,1)
        MgFe_planet = planet[:,12].reshape(-1,1)
        x_Si_mantle = planet[:,13].reshape(-1,1)
        x_Mg_mantle = planet[:,14].reshape(-1,1)
        x_Fe_mantle = planet[:,15].reshape(-1,1)
        w_core = planet[:,16].reshape(-1,1)
        w_mantle = planet[:,17].reshape(-1,1)
        w_water = planet[:,18].reshape(-1,1)
        Teq = planet[:,19].reshape(-1,1)
        Z_atmo = planet[:,20].reshape(-1,1)
        transit_depth = np.ones(mass_planet.shape)*(-1)
        Rtot = np.ones(mass_planet.shape)*(-1)
        likelihood = np.ones(mass_planet.shape)*(-1)
    
        prior_list.append(np.concatenate([mass_planet,          # 0
                                          ws_core,              # 1
                                          ws_mantle,            # 2
                                          ws_water,             # 3
                                          w_core,               # 4
                                          w_mantle,             # 5
                                          w_water,              # 6
                                          w_gas,                # 7
                                          x_Fe_core,            # 8
                                          x_S_core,             # 9
                                          x_Si_mantle,          # 10
                                          x_Mg_mantle,          # 11
                                          x_Fe_mantle,          # 12
                                          SiFe_planet,          # 13
                                          MgFe_planet,          # 14
                                          luminosity,           # 15
                                          period,               # 16
                                          Teq,                  # 17
                                          Z_atmo,               # 18
                                          transit_depth,        # 19
                                          Rtot,                 # 20
                                          likelihood,           # 21
                                          M_star,               # 22
                                          R_star,               # 23
                                          Teff_star,            # 24
                                          SiFe_star,            # 25
                                          MgFe_star,            # 26
                                          age_star],axis=1))    # 27

    priors = np.stack([prior_list[i] for i in range(nplanets)])

    return priors    

def run_grid(nplanets_system,parameters_star,sigmas_star,parameters_target,sigmas_target,comp_option_mass,luminosity_option,SiMgFe_ratio_option,water_mixing_option,mass_range,dnn_prediction,scaler_x_prediction,scaler_y_prediction,comp_option_radius,use_log_prior_for_gas_mass,with_gas,file,verbose,save_prior=False):
    # format of planets_all array:
    # axis 0 - different planets in the system (total: nplanets_system)
    # axis 1 - accepted generated planets during grid search (total: sum of n_kept for each star)
    # axis 2 - planetary properties
    
    # format of stars_all array:
    # axis 0 - empty axis (for consistency)
    # axis 1 - accepted generated stars belonging to accepted systems (N_stars_acc different ones, total: same number as accepted system, some are duplicate)
    # axis 2 - stellar properties

    np.random.seed(42)
    planet_list = []
    star_list = []
    
    # size of grid: 100M planets total
    N_stars = 10000
    N_planets = 10000
    
    # generate stars
    star = generate_stars(N_stars, parameters_star, sigmas_star, debug_star_generation=False)
    N_stars_acc = np.shape(star)[1]
    if verbose:
        print('Generated ' + str(N_stars_acc) + ' stars')
    
    # generate planets for each star
    for i in tqdm.tqdm(np.arange(N_stars_acc), mininterval=60, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
        if verbose:
            print('')
            print('')
            print('*********************************************')
            print('Adding planets for star with index ' + str(i))

        # make array with current star, repeated N_planets times
        curr_star_repeated = np.tile(star[:,i,:],(1,N_planets,1))

        single_planets_list = []
        n_kept = N_planets
        
        # for each planet in the system
        for iplanet in np.arange(nplanets_system):
            if verbose:
                print('')
                print('---------------------------------------------')
                print('Computing planet ' + chr(ord('`')+iplanet+2))
            
            n_planets_computed_simultaneously = 1            # only one planet of the system (iplanet) simultaneously sampled
            single_stars = np.copy(curr_star_repeated)
            single_planet, single_stars = generate_single_planet(parameters_target,sigmas_target,single_stars,iplanet,
                                                                 n_planets_computed_simultaneously,comp_option_mass,
                                                                 luminosity_option,SiMgFe_ratio_option,water_mixing_option,
                                                                 mass_range,dnn_prediction,scaler_x_prediction,scaler_y_prediction,
                                                                 comp_option_radius,use_log_prior_for_gas_mass,with_gas,
                                                                 debug_planet_generation=False,prior=False)
            single_planets_list.append(single_planet)
            n_kept = min(n_kept,np.shape(single_planet)[1])  # need one of each planet for each accepted system
            
            if verbose:
                print('Accepted number of planets: ' + str(n_kept))
        
        # need one of each planet for each accepted system
        planets = np.stack([single_planets_list[k][0,:n_kept,:] for k in range(nplanets_system)],axis=0)

        planet_list.append(planets)
        star_list.append(single_stars[:,:n_kept,:])
        
        if verbose:
            print('---------------------------------------------')
            print('')
            print('Accepted number of systems for current star: ' + str(np.shape(planets)[1]))
        
    # convert list to numpy array    
    planets_all = np.concatenate([planet_list[k] for k in range(len(planet_list))],axis=1)
    stars_all = np.concatenate([star_list[k] for k in range(len(planet_list))],axis=1)

    if verbose:
        print('*********************************************')
        print('')
        
    posteriors = make_posteriors(planets_all, stars_all, nplanets_system)
    np.save('posteriors/' + file + '_posterior', posteriors)

    #save_gas_posterior(file, nplanets_system, posteriors)
    
    print('Shape of posteriors:')
    print(np.shape(posteriors))

    # ----------------------------------------------------------------------------------
    if save_prior:
        np.random.seed(42)
        planet_list = []
        star_list = []
        
        # size of grid: 100M planets total
        N_stars = 1000 #10000
        N_planets = 500 #10000
        
        # generate stars
        star = generate_stars(N_stars, parameters_star, sigmas_star, debug_star_generation=False)
        N_stars_acc = np.shape(star)[1]
        if verbose:
            print('Generated ' + str(N_stars_acc) + ' stars')
        
        # generate planets for each star
        for i in tqdm.tqdm(np.arange(N_stars_acc), mininterval=60, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
            if verbose:
                print('')
                print('')
                print('*********************************************')
                print('Adding planets for star with index ' + str(i))

            # make array with current star, repeated N_planets times
            curr_star_repeated = np.tile(star[:,i,:],(1,N_planets,1))

            single_planets_list = []
            n_kept = N_planets
            
            # for each planet in the system
            for iplanet in np.arange(nplanets_system):
                if verbose:
                    print('')
                    print('---------------------------------------------')
                    print('Computing planet ' + chr(ord('`')+iplanet+2))
                
                n_planets_computed_simultaneously = 1            # only one planet of the system (iplanet) simultaneously sampled
                single_stars = np.copy(curr_star_repeated)
                single_planet, single_stars = generate_single_planet(parameters_target,sigmas_target,single_stars,iplanet,
                                                                    n_planets_computed_simultaneously,comp_option_mass,
                                                                    luminosity_option,SiMgFe_ratio_option,water_mixing_option,
                                                                    mass_range,dnn_prediction,scaler_x_prediction,scaler_y_prediction,
                                                                    comp_option_radius,use_log_prior_for_gas_mass,with_gas,
                                                                    debug_planet_generation=False,prior=True)
                single_planets_list.append(single_planet)
                n_kept = min(n_kept,np.shape(single_planet)[1])  # need one of each planet for each accepted system
                
                if verbose:
                    print('Accepted number of planets: ' + str(n_kept))
            
            # need one of each planet for each accepted system
            planets = np.stack([single_planets_list[k][0,:n_kept,:] for k in range(nplanets_system)],axis=0)

            planet_list.append(planets)
            star_list.append(single_stars[:,:n_kept,:])
            
            if verbose:
                print('---------------------------------------------')
                print('')
                print('Accepted number of systems for current star: ' + str(np.shape(planets)[1]))
            
        # convert list to numpy array    
        planets_all = np.concatenate([planet_list[k] for k in range(len(planet_list))],axis=1)
        stars_all = np.concatenate([star_list[k] for k in range(len(planet_list))],axis=1)

        if verbose:
            print('*********************************************')
            print('')
            
        priors = make_priors(planets_all, stars_all, nplanets_system)
        np.save('posteriors/' + file + '_prior', priors)
        
        print('Shape of priors:')
        print(np.shape(priors))

# Post-processing
# ----------------------------------------------------------
def save_gas_posterior(file,nplanets_system,posteriors):
    w_gas = posteriors[0,:,7].reshape(-1,1)

    for i in range(nplanets_system-1):
        w_gas_i = posteriors[i+1,:,7]
        w_gas = np.concatenate([w_gas, w_gas_i.reshape(-1,1)],axis=1)

    np.savetxt('gas_posteriors/' + file + '.txt', w_gas)

def plot_corner_medium(nplanets_system, posteriors, file):
    for i in range(nplanets_system):
        plt.clf()
        iplanet = i
        titles = ["","","","","","","","",""]
        labels = ["fm$_\mathrm{core}$", "fm$_\mathrm{mantle}$", "fm$_\mathrm{water}$", "log10(fm$_\mathrm{gas}$) ", "Si$_\mathrm{mantle}$","Mg$_\mathrm{mantle}$","Fe$_\mathrm{mantle}$","Fe$_\mathrm{core}$"]
        posteriors_short = np.concatenate([posteriors[iplanet,:,1:4],posteriors[iplanet,:,7:8],posteriors[iplanet,:,10:13],posteriors[iplanet,:,8:9]],axis=1)

        matplotlib.rc('xtick', labelsize=22) 
        matplotlib.rc('ytick', labelsize=22) 
        fig = corner.corner(posteriors_short, labels=labels, titles=titles, show_titles=True, quantiles=[0.05,0.95], plot_contours=True,log=True, label_kwargs=dict(fontsize=30), title_kwargs=dict(fontsize=25), labelpad=0.2);
        
        plt.savefig('plots/' + file + '_' + chr(ord('`')+i+2) + '_corner_medium.pdf',bbox_inches='tight')
        plt.close()

    return

def plot_corner_large(comp_option_mass, comp_option_radius, nplanets_system, parameters_star, parameters_target, posteriors, file):
    for i in range(nplanets_system):
        plt.clf()
        labels = ['mass_planet',          # 0
                  'ws_core',              # 1
                  'ws_mantle',            # 2
                  'ws_water',             # 3
                  'w_core',               # 4
                  'w_mantle',             # 5
                  'w_water',              # 6
                  'w_gas',                # 7
                  'x_Fe_core',            # 8
                  'x_S_core',             # 9
                  'x_Si_mantle',          # 10
                  'x_Mg_mantle',          # 11
                  'x_Fe_mantle',          # 12
                  'SiFe_planet',          # 13
                  'MgFe_planet',          # 14
                  'luminosity',           # 15
                  'period',               # 16
                  'Teq',                  # 17
                  'Z_atmo',               # 18
                  'classification',       # 19
                  'transit_depth',        # 20
                  'Rtot',                 # 21
                  'likelihood',           # 22
                  'M_star',               # 23
                  'R_star',               # 24
                  'Teff_star',            # 25
                  'SiFe_star',            # 26
                  'MgFe_star',            # 27
                  'age_star']             # 28
                  
        ranges = 0.9*np.ones(np.size(labels))
        truths = np.full(np.shape(labels),None)

        if comp_option_mass == 0:
            truths[0] = parameters_target[i,0]*parameters_star[0]**(2/3)*Mjup_Mearth               # planet mass (from relative mass)
        else: #comp_option_mass == 1:
            truths[0] = parameters_target[i,0]                                                          # planet mass
        
        if comp_option_radius == 0 or comp_option_radius == 1:
            truths[20] = parameters_target[i,1]                                                         # transit depth
            truths[21] = (parameters_target[i,1]*1e-6)**0.5*parameters_star[1]*Rsun/Rearth    # planet radius (from transit depth)
        else: #comp_option_radius == 2
            truths[20] = parameters_target[i,1]/(parameters_star[1]*Rsun/Rearth)              # transit depth (from planet radius)
            truths[21] = parameters_target[i,1]                                                         # planet radius

        truths[16] = parameters_target[i,2]                                                             # period

        truths[23] = parameters_star[0]/Msun_Mjup
        truths[24] = parameters_star[1]
        truths[25] = parameters_star[2]
        truths[26] = parameters_star[3]
        truths[27] = parameters_star[4]

        fig = corner.corner(posteriors[i,:,:], labels=labels, show_titles=True, range = ranges, truths=truths, quantiles=[0.11,0.89], plot_contours=True,log=True)
        plt.savefig('plots/' + file + '_' + chr(ord('`')+i+2) + '_corner_large.png',bbox_inches='tight')
        plt.close()

    return

def plot_ternary(nplanets_system, posteriors, file, scale=10):
    for iplanet in range(nplanets_system):
        w_core = posteriors[iplanet,:,4:5]
        w_mantle = posteriors[iplanet,:,5:6]
        w_water = posteriors[iplanet,:,6:7]
        w_gas = 10**posteriors[iplanet,:,7:8]

        xyz = np.concatenate([w_core, w_mantle, w_water+w_gas],axis=1)
        nbins = scale+1
        H, b = np.histogramdd((xyz[:, 0], xyz[:, 1], xyz[:, 2]),
                        bins=(nbins, nbins, nbins), range=((0, 1), (0, 1), (0, 1)))
        H = H / np.sum(H)

        # 3D smoothing and interpolation
        kde = gaussian_filter(H, sigma=2)
        interp_dict = dict()
        binx = np.linspace(0, 1, nbins)
        for i, x in enumerate(binx):
            for j, y in enumerate(binx):
                for k, z in enumerate(binx):
                    interp_dict[(i, j, k)] = kde[i, j, k]

        fig, tax = ternary.figure(scale=scale)
        fig.set_size_inches(10,7)
        tax.boundary(linewidth=2.0)
        #tax.gridlines(multiple=0.1, color="grey")
        tax.ticks(ticks=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],axis='lbr',linewidth=2, multiple=scale, tick_formats="%.1f", offset=0.025, fontsize=16)
        tax.heatmap(interp_dict, cmap='rocket_r', colorbar=True)

        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')
        tax.left_axis_label("w$_\mathrm{envelope}$", fontsize=20, offset=0.18)  # third value
        tax.right_axis_label("w$_\mathrm{mantle}$", fontsize=20, offset=0.18)  # second value
        tax.bottom_axis_label("w$_\mathrm{core}$", fontsize=20, offset=0.12)  # first value

        plt.savefig('plots/' + file + '_' + chr(ord('`')+iplanet+2) + '_ternary.pdf', bbox_inches='tight')
        plt.savefig('plots/' + file + '_' + chr(ord('`')+iplanet+2) + '_ternary.png', bbox_inches='tight')
        plt.close()
        
    return

def plot_water_vs_HHe(nplanets_system, posteriors, file):
    for iplanet in range(nplanets_system):
        w_water = posteriors[iplanet,:,6]
        w_gas = posteriors[iplanet,:,7]

        H, xedges, yedges = np.histogram2d(w_gas, w_water, bins=100)
        H = H / np.sum(H)

        plt.clf()
        fig,ax = plt.subplots(1,1, figsize=(6,4))
        myextent  =[xedges[0],xedges[-1],yedges[0],yedges[-1]]
        CS = ax.imshow(H.T,extent=myextent,interpolation='nearest', origin='lower',aspect='auto', cmap='rocket_r')
        fig.colorbar(CS)

        ax.set_xlabel(r'log10(w$_\mathrm{H/He}$)', fontsize=16)
        ax.set_ylabel(r'w$_\mathrm{water}$', fontsize=16)

        #loc_x = plticker.MultipleLocator(base=1.0)
        loc_y = plticker.MultipleLocator(base=0.1)
        #ax.xaxis.set_major_locator(loc_x)
        ax.yaxis.set_major_locator(loc_y)

        plt.tight_layout()
        plt.savefig('plots/' + file + '_' + chr(ord('`')+iplanet+2) + '_water_vs_HHe.pdf', bbox_inches='tight')
        plt.close()
        
    return

def make_ini_file(comp_option_mass, comp_option_radius, with_gas, with_water, water_mixing_option, csv_file, comment='', date_string='', verbose=False):
    obs, nplanets_total, nsystems_total = read_csv_info(comp_option_mass, comp_option_radius, with_gas, with_water, csv_file, comment=comment, date_string=date_string, verbose=verbose)

    count = 0
    for j in range(nsystems_total):
        file = obs['Name'].iloc[count]
        sys_name = file.partition('_')[0]
        nplanets_system = int(obs['Nplanets_system'].iloc[count])

        for i in range(nplanets_system):
            filename = 'ini/' + file + '_' + chr(ord('`')+i+2) + '_100samples.ini'
            f = open(filename, 'w')

            input_file = '\'Inverse_modelling/' + sys_name + '/sample_lists/' + file + '_' + chr(ord('`')+i+2) + '_100samples.dat\''
            output_file = '\'Inverse_modelling/' + sys_name + '/100samples/' + file + '_' + chr(ord('`')+i+2) + '_100samples\''

            f.write('[output file]\n')
            f.write('input_file = ' + input_file + '\n')
            f.write('output_file = ' + output_file + '\n')

            f.write('\n')
            f.write('[boundary conditions]\n')
            f.write('P_out = 1E0\n')
            f.write('T_eq  = -1.0\n')
            f.write('T_eff = -1.0\n')
            f.write('Z_atmo= -1.0\n')
            f.write('L_int = -1.0\n')
            f.write('R_hill= 2E11\n')
            f.write('\n')
            f.write('[paths]\n')
            f.write('# List of all directories which contain EoS Tables etc.\n')
            f.write('path_iron     = \'../IronEoS/EoSTables\'\n')
            f.write('path_mantle   = \'../BICEPS_MANTLE/TABLE\'\n')
            f.write('path_perplex  = \'../PerpleX\'\n')
            f.write('path_atmo     = \'../BICEPS_ATMOSPHERE/TABLES\'\n')
            f.write('path_water    = \'../WaterEoS/EoSTables\'\n')
            f.write('path_sotin    = \'../SotinEOS/Tables\'\n')
            f.write('path_debye    = \'../DebyeInt\'\n')
            f.write('path_freedman = \'../Freedman_2014/Tables\'\n')
            f.write('path_pg       = \'../Parmentier_Guillot_2014/Tables\'\n')
            f.write('path_sc       = \'../SaumonChabrierEoS/TABLES\'\n')
            f.write('path_nn       = \'../NN_CORE/NN/Weights\'\n')
            f.write('path_mgo      = \'../MgOEoS/Tables\'\n')
            f.write('path_tfd      = \'../TFD_EOS/TABLES\'\n')
            f.write('path_opac     = \'../BICEPS_OPACITIES/TABLES\'\n')
            f.write('\n')
            f.write('[layer models]\n')
            f.write('# Core model:\n')
            f.write('# -1: do not initialize core model, \n')
            f.write('#  0: load Bouchet et al. (2013) hcp-Fe EoS\n')
            f.write('#  1: load Bouchet et al. (2013) bcc-Fe EoS\n')
            f.write('#  2: load Hakim et al. (2018) hcp-Fe EoS and Fei et al. (2016) hcp-Fe EoS\n')
            f.write('#  3: load Sotin et al. (2007) hcp-Fe EoS\n')
            f.write('#  4: load Hakim et al. (2018) hcp-Fe EoS incl. FeX alloys and Fei et al. (2016) hcp-Fe EoS\n')
            f.write('#  5: load Collection of EoS used in BICEPS_CORE (see Haldemann et al. 2021)\n')
            f.write('core_model       = 6\n')
            f.write('\n')
            f.write('# Mantle model:\n')
            f.write('# -1: do not initialize mantle model\n')
            f.write('#  0: Sotin et al. (2007) mantle model\n')
            f.write('#  1: Perple_X and Stixrude & Lithgow-Bertelloni (2014) EoS\n')
            f.write('#  2: BICEPS mantle model\n')
            f.write('mantle_model     = 2\n')
            f.write('\n')
            f.write('# Water model:\n')
            f.write('# -1: do not initialize water model\n')
            f.write('#  0: Sotin et al. (2007)\n')
            f.write('#  1: AQUA EoS by Haldemann et al. (2020)\n')
            f.write('#  2: ANEOS \n')
            f.write('#  3: QEOS by Vazan & Helled (2013)\n')
            f.write('#  4: Mazevet et al. (2019)\n')
            f.write('water_model      = 1\n')
            f.write('\n')
            f.write('# Atmosphere model\n')
            f.write('# -1: do not initialize core model\n')
            f.write('#  0: no irradiation\n')
            f.write('#  1: irradiation model by Guillot (2010)\n')
            f.write('#  2:\n')
            f.write('#  3:\n')
            f.write('#  4: irradiation model by Parmentier & Guillot (2014)\n')
            f.write('atmosphere_model = 4\n')
            f.write('\n')
            f.write('# Opacity model\n')
            f.write('# 0: only Freedman+2014\n')
            f.write('# 1: Freedman+2014 and pure h2o opacities\n')
            f.write('opacity_model = 0\n')
            f.write('\n')
            f.write('[options]\n')
            f.write('#save_profiles = False\n')
            f.write('use_condensation_model = False\n')
            f.write('ratios_unit = mol\n')
            f.write('isothermal_core = False\n')
            f.write('isothermal_mantle = False\n')
            f.write('isothermal_water = False\n')
            f.write('isothermal_atmo = False\n')

            if water_mixing_option == 'separate':
                f.write('mixedAtmo = False\n')
            elif water_mixing_option == 'uniform' or water_mixing_option == 'water_from_gas':
                f.write('mixedAtmo = True\n')
            else:
                    print('Error: Invalid water mixing option')
                    exit()
            
            f.close()

        # update count: index of first planet in next system
        count += obs['Nplanets_system'].iloc[count]

    return

def make_sample_list(n_samples, comp_option_mass, comp_option_radius, with_gas, with_water, water_mixing_option, csv_file, comment='', date_string='', verbose=False):
    obs, nplanets_total, nsystems_total = read_csv_info(comp_option_mass, comp_option_radius, with_gas, with_water, csv_file, comment=comment, date_string=date_string, verbose=verbose)

    count = 0
    for j in range(nsystems_total):
        file = obs['Name'].iloc[count]
        nplanets_system = int(obs['Nplanets_system'].iloc[count])
        
        posteriors = np.load('posteriors/' + file + '_posterior.npy')
        parameters_star = np.load('posteriors/' + file + '_params_star.npy')
        parameters_target = np.load('posteriors/' + file + '_params_target.npy')

        indices = np.arange(posteriors.shape[1])
        chosen_indices = np.random.choice(indices, size=n_samples)

        for i in range(nplanets_system):
            filename = 'sample_lists/' + file + '_' + chr(ord('`')+i+2) + '_100samples.dat'
            f = open(filename, 'w')
            f.write(str(n_samples) + ' 15\n')

            for k in range(n_samples):
                if water_mixing_option == 'separate':
                    line = (str(k+1) + ' ' +                                    # ID
                            str(posteriors[i,k,0]) + ' ' +                      # M_tot
                            str(posteriors[i,k,4]) + ' ' +                      # w_core
                            str(posteriors[i,k,5]) + ' ' +                      # w_mantle
                            str(posteriors[i,k,6]) + ' ' +                      # w_water
                            str(posteriors[i,k,7]) + ' ' +                      # w_gas
                            str(posteriors[i,k,8]) + ' ' +                      # x_Fe_core
                            str(posteriors[i,k,9]) + ' ' +                      # x_S_core
                            str(posteriors[i,k,10]) + ' ' +                     # x_Si_mantle
                            str(posteriors[i,k,11]) + ' ' +                     # x_Mg_mantle
                            str(posteriors[i,k,12]) + ' ' +                     # x_Fe_mantle
                            str(posteriors[i,k,17]) + ' ' +                     # T_eq
                            str(posteriors[i,k,25]) + ' ' +                     # T_eff
                            str(posteriors[i,k,18]) + ' ' +                     # Z_atmo
                            str(posteriors[i,k,15]) + ' ' +                     # L
                            str(posteriors[i,k,21]) + '\n')                     # R_tot
                elif water_mixing_option == 'uniform' or water_mixing_option == 'water_from_gas':
                    line = (str(k+1) + ' ' +                                    # ID
                            str(posteriors[i,k,0]) + ' ' +                      # M_tot
                            str(posteriors[i,k,4]) + ' ' +                      # w_core
                            str(posteriors[i,k,5]) + ' ' +                      # w_mantle
                            str(0.0) + ' ' +                                    # w_water
                            str(posteriors[i,k,7]+posteriors[i,k,6]) + ' ' +    # w_gas
                            str(posteriors[i,k,8]) + ' ' +                      # x_Fe_core
                            str(posteriors[i,k,9]) + ' ' +                      # x_S_core
                            str(posteriors[i,k,10]) + ' ' +                     # x_Si_mantle
                            str(posteriors[i,k,11]) + ' ' +                     # x_Mg_mantle
                            str(posteriors[i,k,12]) + ' ' +                     # x_Fe_mantle
                            str(posteriors[i,k,17]) + ' ' +                     # T_eq
                            str(posteriors[i,k,25]) + ' ' +                     # T_eff
                            str(posteriors[i,k,18]) + ' ' +                     # Z_atmo
                            str(posteriors[i,k,15]) + ' ' +                     # L
                            str(posteriors[i,k,21]) + '\n')                     # R_tot
                else:
                    print('Error: Invalid water mixing option')
                    exit()
                
                f.write(line)
            
            f.close()

        # update count: index of first planet in next system
        count += obs['Nplanets_system'].iloc[count]

    return

def make_slurm_submit_file(n_samples, comp_option_mass, comp_option_radius, with_gas, with_water, water_mixing_option, csv_file, comment='', date_string='', verbose=False):
    obs, nplanets_total, nsystems_total = read_csv_info(comp_option_mass, comp_option_radius, with_gas, with_water, csv_file, comment=comment, date_string=date_string, verbose=verbose)

    count = 0
    for j in range(nsystems_total):
        file = obs['Name'].iloc[count]
        sys_name = file.partition('_')[0]
        version_name = file.split('_')[2]
        compo_option_name = file.split('_')[3]
        nplanets_system = int(obs['Nplanets_system'].iloc[count])

        filename = 'slurm_submit/' + file + '.sh'
        f = open(filename, 'w')

        f.write('#!/bin/bash\n')
        f.write('#SBATCH -a 1\n')
        f.write('#SBATCH -n 1\n')
        f.write('#SBATCH -p all\n')
        f.write('#SBATCH --cpus-per-task 1\n')
        f.write('#SBATCH -o slurm.%A.%a.out # STDOUT\n')
        f.write('#SBATCH -e slurm.%A.%a.err # STDERR\n')
        f.write('#SBATCH -t 0\n')
        f.write('#SBATCH -J ' + version_name + '_' + compo_option_name + '\n')
        f.write('#SBATCH --exclude=opteron[15-18,20-21,24]\n')
        f.write('\n')

        for i in range(nplanets_system):
            f.write('./structure_from_list \'Inverse_modelling/' + sys_name + '/ini/' + file + '_' + chr(ord('`')+i+2) + '_100samples.ini\' 1 ' + str(n_samples) + '\n')

        f.close()

        # update count: index of first planet in next system
        count += obs['Nplanets_system'].iloc[count]

    return

# Executables
# --------------------------------------------------------------------------------------------------------------------------------------------------------
def compute_posterior(comp_option_mass, comp_option_radius, SiMgFe_ratio_option, water_mixing_option, luminosity_option, use_log_prior_for_gas_mass=True, with_gas=True, with_water=True, csv_file='stellar_planetary_parameters.csv', comment='', verbose=True, save_prior=True):
    obs, nplanets_total, nsystems_total = read_csv_info(comp_option_mass, comp_option_radius, with_gas, with_water, csv_file, comment=comment, verbose=verbose)

    count_m1 = 0
    count_m2 = 0
    count_m3 = 0
    mass_range = np.zeros((nplanets_total))
    for i in range(nplanets_total):
        mass = obs['m[M_Earth]'].iloc[i]
        if mass >= 0.5 and mass < 6:
            count_m1+=1
            mass_range[i]=0
        elif mass >= 6 and mass < 15:
            count_m2+=1
            mass_range[i]=1
        elif mass >= 15 and mass < 30:
            count_m3+=1
            mass_range[i]=2
        else:
            print('Error: Planet outside of allowed mass range', i)
            exit()

    dnn_prediction = [None]*3
    scaler_x_prediction = [None]*3
    scaler_y_prediction = [None]*3

    if verbose:
        print('')
        print('Planets in mass range m1: ', count_m1)
        print('Planets in mass range m2: ', count_m2)
        print('Planets in mass range m3: ', count_m3)
        print('')

    if water_mixing_option == 'uniform':
        if count_m1 > 0:
            dnn_prediction[0] = load_dnns('A_m1', verbose)
            scaler_x_prediction[0], scaler_y_prediction[0] = load_scalers('A_m1')

        if count_m2 > 0:
            dnn_prediction[1] = load_dnns('A_m2', verbose)
            scaler_x_prediction[1], scaler_y_prediction[1] = load_scalers('A_m2')

        if count_m3 > 0:
            print('Error: DNN for m3 currently not available yet!')
            exit()
            # dnn_prediction[2] = load_dnns('A_m3', verbose)
            # scaler_x_prediction[2], scaler_y_prediction[2] = load_scalers('A_m3')

    elif water_mixing_option == 'water_from_gas':
        if count_m1 > 0:
            dnn_prediction[0] = load_dnns('B_m1', verbose)
            scaler_x_prediction[0], scaler_y_prediction[0] = load_scalers('B_m1')

        if count_m2 > 0:
            dnn_prediction[1] = load_dnns('B_m2', verbose)
            scaler_x_prediction[1], scaler_y_prediction[1] = load_scalers('B_m2')

        if count_m3 > 0:
            print('Error: DNN for m3 currently not available yet!')
            exit()
            # dnn_prediction[2] = load_dnns('B_m3', verbose)
            # scaler_x_prediction[2], scaler_y_prediction[2] = load_scalers('B_m3')

    else:
        print('Error: Invalid water mixing option')
        exit()

    if count_m1 == 0 and count_m2 == 0 and count_m3 == 0:
        print('Error: No DNNs loaded')
        exit()

    count = 0
    for j in range(nsystems_total):
        file = obs['Name'].iloc[count]
        nplanets_system = int(obs['Nplanets_system'].iloc[count])
        parameters_star, sigmas_star = stellar_params(obs, count)
        parameters_target, sigmas_target = planetary_params(obs, count, comp_option_mass, comp_option_radius)

        define_priors(parameters_star, sigmas_star, water_mixing_option, with_gas, with_water)

        for i in range(1,nplanets_system):
            params, sigmas = planetary_params(obs, count+i, comp_option_mass, comp_option_radius)
            parameters_target = np.concatenate([parameters_target, params],axis=0)
            sigmas_target = np.concatenate([sigmas_target, sigmas],axis=0)
            
        np.save('posteriors/' + file + '_params_star', parameters_star)
        np.save('posteriors/' + file + '_params_target', parameters_target)

        run_grid(nplanets_system, parameters_star, sigmas_star, parameters_target, sigmas_target,
                 comp_option_mass, luminosity_option, SiMgFe_ratio_option, water_mixing_option, mass_range,
                 dnn_prediction, scaler_x_prediction, scaler_y_prediction,
                 comp_option_radius, use_log_prior_for_gas_mass, with_gas, file, verbose, save_prior)

        # update count: index of first planet in next system
        count += obs['Nplanets_system'].iloc[count]

def make_plots(comp_option_mass, comp_option_radius, with_gas, with_water, csv_file, comment='', date_string='', verbose=True):
    obs, nplanets_total, nsystems_total = read_csv_info(comp_option_mass, comp_option_radius, with_gas, with_water, csv_file, comment=comment, date_string=date_string, verbose=verbose)

    count = 0
    for j in range(nsystems_total):
        file = obs['Name'].iloc[count]
        nplanets_system = int(obs['Nplanets_system'].iloc[count])
        
        posteriors = np.load('posteriors/' + file + '_posterior.npy')
        parameters_star = np.load('posteriors/' + file + '_params_star.npy')
        parameters_target = np.load('posteriors/' + file + '_params_target.npy')

        posteriors[:,:,23] = posteriors[:,:,23]/Msun_Mjup
        posteriors[:,:,7] = np.log10(posteriors[:,:,7])
        posteriors[:,:,15] = np.log10(posteriors[:,:,15])

        #plot_ternary(nplanets_system, posteriors, file, scale=10)
        #print('')
        #print('Done with ternary plots')
        #plot_water_vs_HHe(nplanets_system, posteriors, file)
        #print('')
        #print('Done with water vs. H/He plots')
        plot_corner_medium(nplanets_system, posteriors, file)
        print('')
        print('Done with medium corner plots')
        plot_corner_large(comp_option_mass, comp_option_radius, nplanets_system, parameters_star, parameters_target, posteriors, file)
        print('')
        print('Done with large corner plots')

        # update count: index of first planet in next system
        count += obs['Nplanets_system'].iloc[count]

    return

def get_posteriors_to_print(planet, comp_option_mass, comp_option_radius, with_gas, with_water, csv_file, comment='', date_string='', verbose=True):
    obs, nplanets_total, nsystems_total = read_csv_info(comp_option_mass, comp_option_radius, with_gas, with_water, csv_file, comment=comment, date_string=date_string, verbose=verbose)

    count = 0
    for j in range(nsystems_total):
        file = obs['Name'].iloc[count]
        nplanets_system = int(obs['Nplanets_system'].iloc[count])
        
        posteriors = np.load('posteriors/' + file + '_posterior.npy')
        parameters_star = np.load('posteriors/' + file + '_params_star.npy')
        parameters_target = np.load('posteriors/' + file + '_params_target.npy')

        i = planet

        median = np.zeros(11)
        lower = np.zeros(11)
        upper = np.zeros(11)
        lower2 = np.zeros(11)
        upper2 = np.zeros(11)
        lower3 = np.zeros(11)
        upper3 = np.zeros(11)

        median[0] = np.median(posteriors[i,:,4])
        median[1] = np.median(posteriors[i,:,5])
        median[2] = np.median(posteriors[i,:,6]+posteriors[i,:,7])
        median[3] = np.median(posteriors[i,:,6])
        median[4] = np.median(posteriors[i,:,7])
        median[5] = np.median(posteriors[i,:,18])
        median[6] = np.median(posteriors[i,:,8])
        median[7] = np.median(posteriors[i,:,9])
        median[8] = np.median(posteriors[i,:,10])
        median[9] = np.median(posteriors[i,:,11])
        median[10] = np.median(posteriors[i,:,12])

        lower[0] = np.median(posteriors[i,:,4]) - np.quantile(posteriors[i,:,4],0.16)
        lower[1] = np.median(posteriors[i,:,5]) - np.quantile(posteriors[i,:,5],0.16)
        lower[2] = np.median(posteriors[i,:,6]+posteriors[i,:,7]) - np.quantile(posteriors[i,:,6]+posteriors[i,:,7],0.16)
        lower[3] = np.median(posteriors[i,:,6]) - np.quantile(posteriors[i,:,6],0.16)
        lower[4] = np.median(posteriors[i,:,7]) - np.quantile(posteriors[i,:,7],0.16)
        lower[5] = np.median(posteriors[i,:,18]) - np.quantile(posteriors[i,:,18],0.16)
        lower[6] = np.median(posteriors[i,:,8]) - np.quantile(posteriors[i,:,8],0.16)
        lower[7] = np.median(posteriors[i,:,9]) - np.quantile(posteriors[i,:,9],0.16)
        lower[8] = np.median(posteriors[i,:,10]) - np.quantile(posteriors[i,:,10],0.16)
        lower[9] = np.median(posteriors[i,:,11]) - np.quantile(posteriors[i,:,11],0.16)
        lower[10] = np.median(posteriors[i,:,12]) - np.quantile(posteriors[i,:,12],0.16)

        upper[0] = np.quantile(posteriors[i,:,4],0.84) - np.median(posteriors[i,:,4])
        upper[1] = np.quantile(posteriors[i,:,5],0.84) - np.median(posteriors[i,:,5])
        upper[2] = np.quantile(posteriors[i,:,6]+posteriors[i,:,7],0.84) - np.median(posteriors[i,:,6]+posteriors[i,:,7])
        upper[3] = np.quantile(posteriors[i,:,6],0.84) - np.median(posteriors[i,:,6])
        upper[4] = np.quantile(posteriors[i,:,7],0.84) - np.median(posteriors[i,:,7])
        upper[5] = np.quantile(posteriors[i,:,18],0.84) - np.median(posteriors[i,:,18])
        upper[6] = np.quantile(posteriors[i,:,8],0.84) - np.median(posteriors[i,:,8])
        upper[7] = np.quantile(posteriors[i,:,9],0.84) - np.median(posteriors[i,:,9])
        upper[8] = np.quantile(posteriors[i,:,10],0.84) - np.median(posteriors[i,:,10])
        upper[9] = np.quantile(posteriors[i,:,11],0.84) - np.median(posteriors[i,:,11])
        upper[10] = np.quantile(posteriors[i,:,12],0.84) - np.median(posteriors[i,:,12])

        lower2[0] = np.median(posteriors[i,:,4]) - np.quantile(posteriors[i,:,4],0.025)
        lower2[1] = np.median(posteriors[i,:,5]) - np.quantile(posteriors[i,:,5],0.025)
        lower2[2] = np.median(posteriors[i,:,6]+posteriors[i,:,7]) - np.quantile(posteriors[i,:,6]+posteriors[i,:,7],0.025)
        lower2[3] = np.median(posteriors[i,:,6]) - np.quantile(posteriors[i,:,6],0.025)
        lower2[4] = np.median(posteriors[i,:,7]) - np.quantile(posteriors[i,:,7],0.025)
        lower2[5] = np.median(posteriors[i,:,18]) - np.quantile(posteriors[i,:,18],0.025)
        lower2[6] = np.median(posteriors[i,:,8]) - np.quantile(posteriors[i,:,8],0.025)
        lower2[7] = np.median(posteriors[i,:,9]) - np.quantile(posteriors[i,:,9],0.025)
        lower2[8] = np.median(posteriors[i,:,10]) - np.quantile(posteriors[i,:,10],0.025)
        lower2[9] = np.median(posteriors[i,:,11]) - np.quantile(posteriors[i,:,11],0.025)
        lower2[10] = np.median(posteriors[i,:,12]) - np.quantile(posteriors[i,:,12],0.025)

        upper2[0] = np.quantile(posteriors[i,:,4],0.975) - np.median(posteriors[i,:,4])
        upper2[1] = np.quantile(posteriors[i,:,5],0.975) - np.median(posteriors[i,:,5])
        upper2[2] = np.quantile(posteriors[i,:,6]+posteriors[i,:,7],0.975) - np.median(posteriors[i,:,6]+posteriors[i,:,7])
        upper2[3] = np.quantile(posteriors[i,:,6],0.975) - np.median(posteriors[i,:,6])
        upper2[4] = np.quantile(posteriors[i,:,7],0.975) - np.median(posteriors[i,:,7])
        upper2[5] = np.quantile(posteriors[i,:,18],0.975) - np.median(posteriors[i,:,18])
        upper2[6] = np.quantile(posteriors[i,:,8],0.975) - np.median(posteriors[i,:,8])
        upper2[7] = np.quantile(posteriors[i,:,9],0.975) - np.median(posteriors[i,:,9])
        upper2[8] = np.quantile(posteriors[i,:,10],0.975) - np.median(posteriors[i,:,10])
        upper2[9] = np.quantile(posteriors[i,:,11],0.975) - np.median(posteriors[i,:,11])
        upper2[10] = np.quantile(posteriors[i,:,12],0.975) - np.median(posteriors[i,:,12])

        lower3[0] = np.median(posteriors[i,:,4]) - np.quantile(posteriors[i,:,4],0.0015)
        lower3[1] = np.median(posteriors[i,:,5]) - np.quantile(posteriors[i,:,5],0.0015)
        lower3[2] = np.median(posteriors[i,:,6]+posteriors[i,:,7]) - np.quantile(posteriors[i,:,6]+posteriors[i,:,7],0.0015)
        lower3[3] = np.median(posteriors[i,:,6]) - np.quantile(posteriors[i,:,6],0.0015)
        lower3[4] = np.median(posteriors[i,:,7]) - np.quantile(posteriors[i,:,7],0.0015)
        lower3[5] = np.median(posteriors[i,:,18]) - np.quantile(posteriors[i,:,18],0.0015)
        lower3[6] = np.median(posteriors[i,:,8]) - np.quantile(posteriors[i,:,8],0.0015)
        lower3[7] = np.median(posteriors[i,:,9]) - np.quantile(posteriors[i,:,9],0.0015)
        lower3[8] = np.median(posteriors[i,:,10]) - np.quantile(posteriors[i,:,10],0.0015)
        lower3[9] = np.median(posteriors[i,:,11]) - np.quantile(posteriors[i,:,11],0.0015)
        lower3[10] = np.median(posteriors[i,:,12]) - np.quantile(posteriors[i,:,12],0.0015)

        upper3[0] = np.quantile(posteriors[i,:,4],0.9985) - np.median(posteriors[i,:,4])
        upper3[1] = np.quantile(posteriors[i,:,5],0.9985) - np.median(posteriors[i,:,5])
        upper3[2] = np.quantile(posteriors[i,:,6]+posteriors[i,:,7],0.9985) - np.median(posteriors[i,:,6]+posteriors[i,:,7])
        upper3[3] = np.quantile(posteriors[i,:,6],0.9985) - np.median(posteriors[i,:,6])
        upper3[4] = np.quantile(posteriors[i,:,7],0.9985) - np.median(posteriors[i,:,7])
        upper3[5] = np.quantile(posteriors[i,:,18],0.9985) - np.median(posteriors[i,:,18])
        upper3[6] = np.quantile(posteriors[i,:,8],0.9985) - np.median(posteriors[i,:,8])
        upper3[7] = np.quantile(posteriors[i,:,9],0.9985) - np.median(posteriors[i,:,9])
        upper3[8] = np.quantile(posteriors[i,:,10],0.9985) - np.median(posteriors[i,:,10])
        upper3[9] = np.quantile(posteriors[i,:,11],0.9985) - np.median(posteriors[i,:,11])
        upper3[10] = np.quantile(posteriors[i,:,12],0.9985) - np.median(posteriors[i,:,12])

        return median, lower, upper, lower2, upper2, lower3, upper3