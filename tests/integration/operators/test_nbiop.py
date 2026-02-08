
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

        print(dir(self.equilibrium))
        ata

        
        
        #plasma members: ['Electron_pressure', 'Fz', 'Ion_density', 'Lz_tot', 'Meanz', 'R', 'R_midplane', 'Thermal_pressure', 'Total_radiation', 'Wth', 'Zeff', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_electron_pressure', '_fast_ion_pressure', '_fz', '_ion_density', '_lz_tot', '_meanz', '_prad_tot', '_pressure', '_thermal_pressure', '_time_to_calculate', '_total_radiation', '_wfast', '_wp', '_wth', '_zeff', 'area', 'build_atomic_data', 'calc_electron_pressure', 'calc_fz', 'calc_ion_density', 'calc_lz_tot', 'calc_meanz', 'calc_thermal_pressure', 'calc_total_radiation', 'calc_wth', 'calc_zeff', 'dt', 'electron_density', 'electron_pressure', 'electron_temperature', 'element_a', 'element_name', 'element_symbol', 'element_z', 'elements', 'equilibrium', 'fast_ion_density', 'fast_ion_pressure', 'fract_abu', 'full_run', 'fz', 'impurities', 'impurity_concentration', 'impurity_density', 'ion_density', 'ion_temperature', 'lz_tot', 'machine_conf', 'main_ion', 'map_to_2d', 'meanz', 'neutral_density', 'parallel_fast_ion_pressure', 'perpendicular_fast_ion_pressure', 'power_loss_tot', 'prad_tot', 'pressure', 'rho_type', 'rhop', 'rmag', 'rmin', 'rmji', 'rmjo', 'set_adf11', 'set_equilibrium', 'set_impurity_concentration', 't', 'tau', 'tend', 'thermal_pressure', 'time_to_calculate', 'toroidal_rotation', 'total_radiation', 'tstart', 'verbose', 'volume', 'wfast', 'wp', 'write_to_pickle', 'wth', 'z', 'z_midplane', 'zeff', 'zmag']
        #Equilibrium members: ['Bfield', 'Bp', 'Br', 'Bt', 'Btot', 'Bz', 'R', 'R_hfs', 'R_lfs', 'R_offset', 'Rmax', 'Rmin', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_data', 'ajac', 'area', 'convert_flux_coords', 'corner_angles', 'cross_sectional_area', 'enclosed_volume', 'f', 'flux_coords', 'ftor', 'index', 'ipla', 'minor_radius', 'psi', 'psi_axis', 'psi_boundary', 'psin', 'rbnd', 'rgeo', 'rhop', 'rhot', 'rmag', 'rmji', 'rmjo', 'spatial_coords', 't', 'vjac', 'volume', 'wp', 'write_to_geqdsk', 'z', 'z_offset', 'zbnd', 'zmag', 'zmax', 'zmin', 'zx_low', 'zx_up']
        
        
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
        nbi_op(profiles,eqdata)


a=testNBI()
