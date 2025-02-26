"""
This module contains wrappers for the "Photochem" photochemical
model (https://github.com/Nicholaswogan/photochem). These wrappers are 
called in "justdoit.py" during climate simulations if photochemistry
is turned on. The only function useful for general users is 
`generate_photochem_rx_and_thermo_file`, which can generate reaction and 
thermodynamic files used for initializing Photochem
"""

import numpy as np
import warnings
from photochem.extensions import EvoAtmosphereGasGiant
import pickle
import os

import yaml

# Turn off Panda's performance warnings
import pandas as pd
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    
###
### Extension of EvoAtmosphere class for gas giants
###

class EvoAtmosphereGasGiantPicaso(EvoAtmosphereGasGiant):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_file = None

###
### Some PICASO specific methods for the class
###

    def add_concentrations_to_picaso_df(self, df):
        """Adds photochem concentrations to a PICASO "profile" DataFrame

        Parameters
        ----------
        df : DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame containing pressure (bar), 
            temperature (K), and gas concentrations in volume mixing ratios.

        Returns
        -------
        DataFrame
            Pandas DataFrame with Photochem result added. Mixing ratios are normalized so that
            they sum to 1.
        """        

        # Get the photochem results
        sol = self.return_atmosphere_climate_grid()

        # Check to make sure P in df match what is in photochem
        if not np.all(np.isclose(df['pressure'].to_numpy()[::-1].copy()*1e6, self.gdat.P_clima_grid)):
            raise Exception('The pressures in `df` does not match the climate grid in photochem')

        # Add mixing ratios to df. Make sure to exclude particles.
        species_names = self.dat.species_names[self.dat.np:(-2-self.dat.nsl)]
        for key in species_names:
            if key not in ['pressure','temperature','Kzz']:
                df[key] = sol[key][::-1].copy()

        # Renormalized so that mixing ratios sum to 1
        mix_tot = np.zeros(len(df['pressure']))
        for key in df:
            if key not in ['pressure', 'temperature', 'kz']:
                mix_tot += df[key].to_numpy()
        for key in df:
            if key not in ['pressure', 'temperature', 'kz']:
                df[key] = df[key]/mix_tot

        return df

    def initialize_to_climate_equilibrium_PT_picaso(self, df, Kzz_in, metallicity, CtoO, rainout_condensed_atoms=True):
        """Wrapper to `initialize_to_climate_equilibrium_PT`, which accepts a Pandas DataFrame
        containing the input pressure (bar) and temperature (K). The order of all input arrays 
        flipped (i.e., first element is TOA) following the PICASO convention.

        Parameters
        ----------
        df : DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame containing pressure (bar), 
            temperature (K). 
        Kzz_in : ndarray[dim=1,float64]
            Eddy diffusion (cm^2/s) array corresponding to each pressure level in df['pressure']
        """
        P_in = df['pressure'].to_numpy()
        T_in = df['temperature'].to_numpy()

        self.initialize_to_climate_equilibrium_PT(P_in[::-1].copy()*1e6, T_in[::-1].copy(), Kzz_in[::-1].copy(), 
                                                  metallicity, CtoO, rainout_condensed_atoms)
        
    def reinitialize_to_new_climate_PT_picaso(self, df, Kzz_in):
        """Wrapper to `reinitialize_to_new_climate_PT`, which accepts a Pandas DataFrame which contains
        `pressure` in bar, `temperature` in K, and mixing ratios. The order of input arrays are flipped 
        (i.e., first element is TOA) following the PICASO convention. 

        Parameters
        ----------
        df : DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame containing pressure (bar), 
            temperature (K), and gas concentrations in volume mixing ratios. 
        Kzz_in : ndarray[dim=1,float64]
            Eddy diffusion (cm^2/s) array corresponding to each pressure level in df['pressure']
        """

        P_in = df['pressure'].to_numpy()[::-1].copy()*1e6
        T_in = df['temperature'].to_numpy()[::-1].copy()
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        mix = {}
        for key in df:
            if key in species_names:
                mix[key] = df[key].to_numpy()[::-1].copy()

        # normalize
        mix_tot = np.zeros(P_in.shape[0])
        for key in mix:
            mix_tot += mix[key]
        for key in mix:
            mix[key] = mix[key]/mix_tot

        self.reinitialize_to_new_climate_PT(P_in, T_in, Kzz_in[::-1].copy(), mix)

    def run_for_picaso(self, df, log10metallicity, CtoO, Kzz, first_run, rainout_condensed_atoms=True):
        """Runs the Photochemical model to steady-state using inputs from the PICASO climate model.

        Parameters
        ----------
        df : DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame containing pressure (bar), 
            temperature (K), and gas concentrations in volume mixing ratios. The first element
            of each array is the top of the atomsphere (i.e. order is flipped).
        log10metallicity : float
            log10 metallicity relative to solar.
        CtoO : float
            The C/O ratio relative to solar.
        Kzz : ndarray[dim=1,float64]
            Eddy diffusion (cm^2/s) corresponding to each pressure in df['pressure'].
        first_run : bool
            If this is the first photochem call, then this should be True
        rainout_condensed_atoms : bool, optional
            If True and `first_run` is True, then the code rains out condensed
            atoms when guesing the initial solution, by default True.

        Returns
        -------
        DataFrame
            A PICASO "inputs['atmosphere']['profile']" DataFrame similar to the input, except
            steady-state photochemistry gas concentrations are loaded in.
        """        

        # Initialize Photochem to `df`
        if first_run:
            self.initialize_to_climate_equilibrium_PT_picaso(df, Kzz, 10.0**log10metallicity, CtoO, rainout_condensed_atoms)
        else:
            self.reinitialize_to_new_climate_PT_picaso(df, Kzz)
            if not np.isclose(self.gdat.metallicity, 10.0**log10metallicity) or not np.isclose(self.gdat.CtoO, CtoO):
                raise Exception('`metallicity` or `CtoO` does not match.')

        # Compute steady state 
        success = self.find_steady_state()
        assert success

        if self.save_file is not None:
            sol = self.return_atmosphere_climate_grid()
            model = self.model_state_to_dict()
            if not os.path.isfile(self.save_file):
                with open(self.save_file, 'wb') as f:
                    pass
            with open(self.save_file,'ab') as f:
                pickle.dump((sol,model,),f)

        # Return a DataFrame with the Photochem chemistry
        return self.add_concentrations_to_picaso_df(df)

def set_equilibrium_composition_to_picaso_df(pc, mechanism_file, df):
    
    # Read the mechanism file
    with open(mechanism_file,'r') as f:
        data = yaml.load(f,Loader=yaml.Loader)
    species_composition = {}
    for i,sp in enumerate(data['species']):
        species_composition[sp['name']] = sp['composition']
    for i,sp in enumerate(data['particles']):
        species_composition[sp['name']] = sp['composition']
    
    # Build a composition dictionary
    comp = {}
    for i,atom in enumerate(data['atoms']):
        comp[atom['name']] = 0.0

    # Compute the composition of the deepest layer in PICASO df
    for i,sp in enumerate(df):
        if sp in species_composition:
            for atom in species_composition[sp]:
                comp[atom] += species_composition[sp][atom]*df[sp].to_numpy()[-1]
    
    # Renormalize the composition
    tot = 0.0
    for key in comp:
        tot += comp[key]
    for key in comp:
        comp[key] = comp[key]/tot
    
    # Convert composition to array
    molfracs_atoms = np.empty(len(pc.m.gas.atoms_names))
    for i,atom in enumerate(pc.m.gas.atoms_names):
        molfracs_atoms[i] = comp[atom]

    # Set composition in equilibrium solver
    pc.m.gas.molfracs_atoms_sun = molfracs_atoms