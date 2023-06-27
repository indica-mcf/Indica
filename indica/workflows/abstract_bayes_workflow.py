from abc import ABC, abstractmethod


class BayesWorkflow(ABC):
    @abstractmethod
    def setup_plasma(self):
        """
        Contains all methods and settings for plasma object to be used in optimisation
        """
        self.plasma = None

    @abstractmethod
    def read_data(self, diagnostics: list):
        """
        Reads data from server

        Returns

        nested dictionary of data
        """
        self.data = {}

    def setup_opt_data(self, phantom: bool = False):
        """
        Prepare the data in necessary format for optimiser i.e. flat dictionary
        """
        self.opt_data = {}

    @abstractmethod
    def setup_models(self, diagnostics: list):
        """
        Initialising models normally requires data to be read so transforms can be set

        """
        self.models = {}

    @abstractmethod
    def setup_optimiser(self):
        """
        Initialise and provide settings for optimiser
        """
        self.bayesopt = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        return {}

