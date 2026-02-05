
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

        print(self.transforms.keys())


        #How do we wanna call?
        #Specifically beam related stuff in init
        nbiconfig = {
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
        nbi_op = nbioperator.NBIOperator(nbi_transform,nbiconfig)
        nbi_op(profiles,eqdata,nbiconfig)


a=testNBI()