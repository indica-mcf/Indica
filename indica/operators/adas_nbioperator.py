from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import constants
import xarray as xr
from xarray import DataArray

from indica.operators import NbiOperator
from indica.readers import ADASReader
from indica.utilities import get_element_info


class NbiADAS(NbiOperator):
    """Calculate neutral beam attenuation from beam stopping coefficients, typically
    ADAS ADF21 (bms) data.

    """

    bms: dict[str, DataArray]

    def prepare(
        self,
        bms: dict[str, DataArray] | None = None,
        reader: ADASReader | None = None,
        year: str = "97",
        bms_quantity: str = "bms",
    ):
        """Calculate neutral attenuation factors

        Parameters
        ----------
        **kwargs : [type]
            [description]
        """
        if bms is None:
            if reader is None:
                reader = ADASReader()
            beam_species = str(self.nbi_element_info["symbol"]).lower()
            if beam_species == "d" or beam_species == "t":
                beam_species = "h"
            self.bms = {
                str(elem): reader.get_adf21(
                    element=str(elem),
                    charge=str(get_element_info(elem)[0]),
                    year=year,
                    beam=beam_species,
                    quantity=bms_quantity,
                )
                for elem in self.Ni.element.values
            }
        else:
            self.bms = bms
        

    
        # beam species mass in [amu]
        amu = float(self.nbi_element_info["A"]) 
       
        # beam energy in [eV/amu]
        e_amu = self.energy / amu #beam energy in [eV/amu]
    
        # beam velocity in [m/s] for each fraction
        v_beam =  np.sqrt(2 * self.power_fractions_identifiers * e_amu * constants.e / constants.m_u)

        # map kinetic profiles to NBI path
        te_mapped = self.transform.map_profile_to_los(self.Te, np.asarray(self.t)) # eV
        ne_mapped = self.transform.map_profile_to_los(self.Ne, np.asarray(self.t)) # m-3
        ni_mapped = self.transform.map_profile_to_los(self.Ni, np.asarray(self.t)) # for all impurities in one go?
        ci_mapped = ni_mapped/ne_mapped # ion concentrations (need to check that only ions are present for which we have bms data)
        zeff_mapped=xr.zeros_like(ne_mapped)
        for element in self.Ni.element:
            zeff_mapped += get_element_info(element)[0]^2 * ci_mapped.sel(element=element)
      
        # evaluate bms data on NBI path 
        bms_mapped= np.zeros(self.transform.x1.size,len(self.power_fractions)) #(n_beamlength,n_fractions) #[m^3/s]
        dl = self.transform.dl # [m] grid step 

        # do attenuation calculation 
        # the collision energy should be corrected with the toroidal rotation
        for k_e in range(len(self.power_fractions)): #energy fractions
            for element in self.Ni.element: #impurities including main ions
                # evaluate atomic data for each ion 
                z_element=get_element_info(element)[0]
                
                ne_equiv=ne_mapped * zeff_mapped/float(z_element) # [m^3]
                        
                # The reader returns bms data in m^3/s      
                bms_mapped[:,k_e] += bms[element].interp(target_temperature=te_mapped, target_density=ne_equiv, beam_energy=e_amu/float(k_e)) #[m^3/s]
              
                normalization += z_element * ci_mapped.sel(element=element)
              
             bms_mapped[:,k_e] = bms_mapped[:,k_e] / normalization
             self.zeta[:,k_e] = np.exp( -( bms_mapped[:,k_e] /v_beam[k_e] * ne_mapped).cumsum("los_position") *dl ) 
             # source rate is particles/s
             self.source_neutral_rate[k_e]= self.power * self.power_fractions[k_e] / (self.energy * constants.e * self.power_fractions_identfiers[k_e]) # particles/s



    def run(self, **kwargs):
        #self.neutral_density = self.source_neutral_rate * self.zeta #not a density

    def refactor_output(self):
        result = {"neutral_density": self.neutral_density}
        return result

    def __call__(
        self,
        Ti: DataArray,
        Te: DataArray,
        Ne: DataArray,
        Ni: DataArray, # contains arrays of impurity concentrations 
        Vtor: DataArray,
        t: float | ArrayLike,
        prepare_kwargs: dict = {},
        run_kwargs: dict = {},
    ) -> dict:
        """
        Run NBI code for specified time-point (one only!)

        target_element - plasma main ion element symbol (e.g. "d" for deuterium)
        file_name - first part of the file name to save the NBI model data to
        """
        if not hasattr(self, "transform"):
            raise ValueError("transform is required (set it before calling)")

        if not hasattr(self.transform, "equilibrium"):
            raise ValueError("transform is missing equilibrium data")

        _element_info = get_element_info(target_element)
        self.target_element_info = {
            "Z": _element_info[0],
            "A": _element_info[1],
            "name": _element_info[2],
            "symbol": _element_info[3],
        }

        self.t = t
        self.file_name = file_name
        self.pulse = pulse
        self.machine = machine

        # self.Ti = Ti.interp(t=t)
        self.Te = Te
        self.Ni = Ni
        self.Ne = Ne
        # self.Nn = Nn.interp(t=t)
        # self.Vtor = Vtor.interp(t=t)
        self.Zeff = Zeff.sum("element")
        self.MeanZ = MeanZ
        self.impurity_charge = ImpurityCharge

        """Prepare input data structure for NBI code"""
        self.prepare(**prepare_kwargs)

        """Run NBI code"""
        self.run(**run_kwargs)

        """Reorganise NBI code output to return Indica-native results"""
        result = self.refactor_output()

        return result
