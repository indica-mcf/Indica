
from indica.operators import nbioperator
from indica.defaults.load_defaults import load_default_objects

class testNBI:

    def __init__(self):
        self.machine = "st40"
        self.machine = self.machine
        self.transforms = load_default_objects(self.machine, "geometry")
        nbi_transform = self.transforms["tws_c"]  # This should actually be an nbi transform
        self.equilibrium = load_default_objects(self.machine, "equilibrium")
        self.plasma = load_default_objects(self.machine, "plasma")
        self.plasma.set_equilibrium(self.equilibrium)
        nbi_transform.set_equilibrium(self.equilibrium)

        pulse = 13475  # This is used to build output paths and locate output files later.
        # fidasim_utils uses this to create the fidasim output dictionary.
        # It does not affect the computation.


        
        
        
        #These are the default values used in JW's code, just adding them here for the sake of testing
        #These should obviously come from the transform we use for testing
        nbi_transform.focal_length = -0.03995269  # meter, this is not in default objects, but should be. Known issue
        spot_width = 1.1 * 1e-3
        spot_height = 1.1 * 1e-3
        nbi_transform.spot_width = spot_width
        nbi_transform.spot_height = spot_height


        # Operator initialization (verbose parameters).
        nbi_op = nbioperator.NBIOperator(
            name="hnbi",
            einj=52.0,  # keV
            pinj=0.5,  # MW
            current_fractions=[0.5, 0.35, 0.15],
            ab=2.014,
            pulse=pulse,
        )
        nbi_op.set_transform(nbi_transform)
        nbi_op.set_plasma(self.plasma)


        #Go time
        #just plasma, if. Look thomson. 

        nbi_model="FIDASIM"

        #This call can be zero params. If it already has a plasma.
        neutrals_by_time = nbi_op(
            nbi_model="FIDASIM",
            ion_temperature=self.plasma.ion_temperature,
            electron_temperature=self.plasma.electron_temperature,
            electron_density=self.plasma.electron_density,
            neutral_density=self.plasma.neutral_density,
            toroidal_rotation=self.plasma.toroidal_rotation,
            zeff=self.plasma.zeff,
            t=self.plasma.t,
            pulse=pulse,
        )
        
        print(neutrals_by_time)


a=testNBI()
