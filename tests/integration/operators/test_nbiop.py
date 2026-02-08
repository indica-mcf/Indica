
import copy
from indica.operators import nbioperator
from indica.operators.nbi_configs import DEFAULT_NBI_SPECS
from indica.defaults.load_defaults import load_default_objects

class testNBI:

    def __init__(self):
        self.machine = "st40"
        self.machine = self.machine
        self.transforms = load_default_objects(self.machine, "geometry")
        nbi_transform=self.transforms["tws_c"] #This should actually be an nbi transfor
        self.equilibrium = load_default_objects(self.machine, "equilibrium")
        self.plasma = load_default_objects(self.machine, "plasma")
        self.plasma.set_equilibrium(self.equilibrium)

        pulse=13475


        
        
        
        #These are the default values used in JW's code, just adding them here for the sake of testing
        nbi_transform.focal_length = -0.03995269  # meter, this is not in default objects, but should be
        spot_width = 1.1 * 1e-3  # manual change for testing purposes
        spot_height = 1.1 * 1e-3  # manual change for testing purposes
        nbi_transform.spot_width = spot_width
        nbi_transform.spot_height = spot_height


        # Spectroscopy config (copied so tests can tweak locally if needed)
        nbispecs = copy.deepcopy(DEFAULT_NBI_SPECS)

        nbi_op = nbioperator.NBIOperator(nbi_transform,nbispecs)
        profiles = {
            "t": self.plasma.t,
            "ion_temperature": self.plasma.ion_temperature,
            "electron_temperature": self.plasma.electron_temperature,
            "electron_density": self.plasma.electron_density,
            "neutral_density": self.plasma.neutral_density,
            "toroidal_rotation": self.plasma.toroidal_rotation,
            "zeff": self.plasma.zeff,
        }
        eqdata = {
            "rhop": self.equilibrium.rhop,
            "convert_flux_coords": self.equilibrium.convert_flux_coords,
            "Br": self.equilibrium.Br,
            "Bz": self.equilibrium.Bz,
        }
        nbi_op(profiles,eqdata, pulse=pulse)


a=testNBI()
