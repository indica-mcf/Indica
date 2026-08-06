from indica.configs.readers.machineconf import MachineConf


class MASTUConf(MachineConf):
    def __init__(self):
        self.MACHINE_DIMS = ((0.1, 2.0), (-2.2, 2.2))
        self.INSTRUMENT_METHODS = {}
        self.QUANTITIES_PATH = {}
