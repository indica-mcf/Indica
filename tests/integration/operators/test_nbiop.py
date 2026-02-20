
import copy
from indica.operators import nbioperator
from indica.operators.nbi_configs import DEFAULT_NBI_SPECS
from indica.defaults.load_defaults import load_default_objects

class testNBI:

    def __init__(self):
        self.machine = "st40"
        self.machine = self.machine
        self.transforms = load_default_objects(self.machine, "geometry")
        nbi_transform=self.transforms["tws_c"] #This should actually be an nbi transform
        self.equilibrium = load_default_objects(self.machine, "equilibrium")
        self.plasma = load_default_objects(self.machine, "plasma")
        self.plasma.set_equilibrium(self.equilibrium)

        pulse=13475 #This is actually used in nbioperator to build output paths, and to locate the
        #output files later. #nbi_utils uses this to create the fidasim output dictionary.
        #Does not affect any of the computation though


        
        
        
        #These are the default values used in JW's code, just adding them here for the sake of testing
        #These should obviously come from the transform we use for testing
        nbi_transform.focal_length = -0.03995269  # meter, this is not in default objects, but should be. Known issue
        spot_width = 1.1 * 1e-3  
        spot_height = 1.1 * 1e-3  
        nbi_transform.spot_width = spot_width
        nbi_transform.spot_height = spot_height


        # NBI config (copied so tests can tweak locally if needed). This should probably just come from
        #configs and not be given to the operator.
        nbispecs = copy.deepcopy(DEFAULT_NBI_SPECS)



        #Operator initialisation
        nbi_op = nbioperator.NBIOperator(nbi_transform,nbispecs)


        #Profiles and eqdata organisation
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


        #Go time
        neutrals_by_time=nbi_op(profiles,eqdata, pulse=pulse)
        print(neutrals_by_time)


a=testNBI()
