
from indica.operators import nbioperator
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



        #These are the default values used in JW's code, just adding them here for the sake of testing
        nbi_transform.focal_length = -0.03995269  # meter, this is not in default objects, but should be
        spot_width = 1.1 * 1e-3  # manual change for testing purposes
        spot_height = 1.1 * 1e-3  # manual change for testing purposes
        nbi_transform.spot_width = spot_width
        nbi_transform.spot_height = spot_height



        #How do we wanna call?
        #Specifically beam related stuff in init
        nbispecs = {
            "name": "hnbi",
            "einj": 52.0,  # keV
            "pinj": 0.5,   # MW
            "current_fractions": [
                0.5,
                0.35,
                0.15
            ],
            "ab": 2.014
        }

        print(dir(nbi_transform))
        nbi_op = nbioperator.NBIOperator(nbi_transform,nbispecs)
        ata
        nbi_op(profiles,eqdata)


a=testNBI()