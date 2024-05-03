from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable
import flatdict

from indica.defaults.read_write_defaults import load_default_objects
from indica.models.charge_exchange import pi_transform_example
from indica.models.helike_spectroscopy import helike_transform_example
from indica.models.interferometry import smmh1_transform_example
from indica.models.thomson_scattering import ts_transform_example
from indica.readers.read_st40 import ReadST40


@dataclass
class ReaderSettings:
    revisions: dict = field(default_factory=lambda: {})
    filters: dict = field(default_factory=lambda: {})


@dataclass  # type: ignore[misc]
class DataContext(ABC):
    reader_settings: ReaderSettings
    pulse: Optional[int]
    tstart: float
    tend: float
    dt: float
    diagnostics: list
    phantoms = False

    @abstractmethod
    def read_data(
        self,
    ):
        self.equilbrium = None
        self.transforms = None
        self.raw_data = None
        self.binned_data = None

    @abstractmethod
    def data_strategy(self):
        return None

    def _check_if_data_present(self, data_strategy: Callable = lambda: None):
        if not self.binned_data:
            print("Data not given: using data strategy")
            self.binned_data = data_strategy()

    def pre_process_data(self, model_callable: Callable):
        self.model_callable = model_callable
        # TODO: handle this dependency (phantom data) some other way?
        self._check_if_data_present(self.data_strategy)

    @abstractmethod
    def process_data(self, model_callable: Callable):
        self.pre_process_data(model_callable)
        self.opt_data = flatdict.FlatDict(self.binned_data)


@dataclass
class ExpData(DataContext):
    phantoms = False

    """
    Considering: either rewriting this class to take over from ReadST40 or vice versa

    """

    def read_data(
        self,
    ):
        self.reader = ReadST40(
            self.pulse,
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
        )
        self.reader(
            self.diagnostics,
            revisions=self.reader_settings.revisions,
            filters=self.reader_settings.filters,
        )
        missing_keys = set(self.diagnostics) - set(self.reader.binned_data.keys())
        if len(missing_keys) > 0:
            raise ValueError(f"missing data: {missing_keys}")
        self.equilibrium = self.reader.equilibrium
        self.transforms = self.reader.transforms
        # TODO raw data included
        self.raw_data = self.reader.raw_data
        self.binned_data = self.reader.binned_data

    def data_strategy(self):
        raise ValueError("Data strategy: Fail")

    def process_data(self, model_callable: Callable):
        self.pre_process_data(model_callable)
        opt_data = flatdict.FlatDict(self.binned_data, ".")
        if "xrcs.spectra" in opt_data.keys():
            background = opt_data["xrcs.spectra"].where(
                (opt_data["xrcs.spectra"].wavelength < 0.392)
                & (opt_data["xrcs.spectra"].wavelength > 0.388),
                drop=True,
            )
            opt_data["xrcs.spectra"]["error"] = np.sqrt(
                opt_data["xrcs.spectra"] + background.std(dim="wavelength") ** 2
            )
        # TODO move the channel filtering to the read_data method in filtering = {}
        if "ts.ne" in opt_data.keys():
            opt_data["ts.ne"]["error"] = opt_data["ts.ne"].max(dim="channel") * 0.05
            # opt_data["ts.ne"] = opt_data["ts.ne"].where(
            #     opt_data["ts.ne"].channel < 21)

        if "ts.te" in opt_data.keys():
            opt_data["ts.ne"]["error"] = opt_data["ts.ne"].max(dim="channel") * 0.05
            # opt_data["ts.te"] = opt_data["ts.te"].where(
            # opt_data["ts.te"].channel < 21)

        if "cxff_tws_c.ti" in opt_data.keys():
            opt_data["cxff_tws_c.ti"] = opt_data["cxff_tws_c.ti"].where(
                opt_data["cxff_tws_c.ti"].channel == 0
            )

        if "cxff_pi.ti" in opt_data.keys():
            opt_data["cxff_pi.ti"] = opt_data["cxff_pi.ti"].where(
                (opt_data["cxff_pi.ti"].channel > 2)
                & (opt_data["cxff_pi.ti"].channel < 5)
            )

        self.opt_data = opt_data


@dataclass
class PhantomData(DataContext):
    phantoms = True

    def read_data(
        self,
    ):
        self.reader = ReadST40(
            self.pulse,
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
        )
        self.reader(
            self.diagnostics,
            revisions=self.reader_settings.revisions,
            filters=self.reader_settings.filters,
        )
        missing_keys = set(self.diagnostics) - set(self.reader.binned_data.keys())
        if len(missing_keys) > 0:
            raise ValueError(f"missing data: {missing_keys}")
        self.equilibrium = self.reader.equilibrium
        self.transforms = self.reader.transforms
        self.raw_data = {}
        self.binned_data = {}

    def data_strategy(self):
        print("Data strategy: Phantom data")
        return self.model_callable()

    def process_data(self, model_callable: Callable):
        self.pre_process_data(model_callable)
        self.opt_data = flatdict.FlatDict(self.binned_data, ".")


@dataclass
class MockData(PhantomData):
    diagnostic_transforms: dict = field(
        default_factory=lambda: {
            "xrcs": helike_transform_example(1),
            "smmh1": smmh1_transform_example(1),
            "cxff_pi": pi_transform_example(5),
            "cxff_tws_c": pi_transform_example(3),
            "ts": ts_transform_example(11),
            "efit": lambda: None,
            # placeholder to stop missing_transforms error
        }
    )

    def read_data(self):
        print("Reading mock equilibrium / transforms")
        self.equilibrium = load_default_objects("st40", "equilibrium")

        missing_transforms = list(
            set(self.diagnostics).difference(self.diagnostic_transforms.keys())
        )
        if missing_transforms:
            raise ValueError(f"Missing transforms: {missing_transforms}")

        # self.transforms = load_default_objects("st40", "geometry")
        self.transforms = self.diagnostic_transforms
        self.binned_data: dict = {}
        self.raw_data: dict = {}
