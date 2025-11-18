from copy import deepcopy
from getpass import getuser
from os import cpu_count
from pathlib import Path
import pickle as pkl
from socket import getfqdn
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray
import pyswarms as ps
from sal.core.exception import NodeNotFound
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
import xarray as xr
from xarray import DataArray
from xarray import Dataset
import yaml

from indica.configs.readers.adasconf import ADF11
from indica.equilibrium import Equilibrium
from indica.models.bremsstrahlung import BremsstrahlungSpectrometer
from indica.models.charge_exchange_spectrometer import ChargeExchangeSpectrometer
from indica.models.effective_charge import EffectiveCharge
from indica.models.pinhole_camera import PinholeCamera
from indica.numpy_typing import RevisionLike
from indica.operators.atomic_data import default_profiles
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.plasma import Plasma
from indica.plasma import PlasmaProfiler
from indica.plasma import ProfilerBase
from indica.profilers.profiler_spline import ProfilerCubicSpline
from indica.readers.adas import ADASReader
from indica.readers.jetreader import JETReader
from indica.utilities import get_element_info
from indica.workflows.pywsxp.optimise import _profile_parameters
from indica.workflows.pywsxp.optimise import costfn
from indica.workflows.pywsxp.optimise import DEFAULT_OPTIONS
from indica.workflows.pywsxp.optimise import recover_profiles
from indica.workflows.pywsxp.results import convert_to_dataset
from indica.workflows.pywsxp.utilities import Config
from indica.workflows.pywsxp.utilities import Diagnostic
from indica.workflows.pywsxp.utilities import Diagnostics
from indica.workflows.pywsxp.utilities import History
from indica.workflows.pywsxp.utilities import Inputs
from indica.workflows.pywsxp.utilities import Results

DEFAULTS: Dict[str, Any] = dict(
    n_iters=250,
    nt=5,
    avrg=0.25,
    xknots=(0.0, 0.3, 0.6, 0.8, 0.9, 1.05),
    highz="w",
    midz="ni",
    lowz="be",
    equilibrium=("jetppf", "eftp", 0),
    electron_density=("jetppf", "hrts", 0),
    electron_temperature=("jetppf", "hrts", 0),
    ion_temperature=("jetppf", "cxg6", 0),
    toroidal_rotation=("jetppf", "cxg6", 0),
    diagnostics=[
        ("jetppf", "sxrv", 0, 1.0),
        ("jetppf", "sxrh", 0, 1.0),
        ("jetppf", "kb5v", 0, 1.0),
        ("jetppf", "kb5h", 0, 1.0),
        ("jetppf", "cxg6", 0, 1.0),
        ("jetppf", "ks3v", 0, 1.5),
    ],
    diag_to_scale=["sxrv", "sxrh", "kb5h"],
    impurity_densities=None,
    zeff_rho_max=0.8,
    channels=dict(
        sxrv=list(range(1, 26)),
        sxrh=[7, 8, 10],
        kb5v=[3, 4, 16, 18, 20],
        kb5h=[11, 14, 16, 18],
    ),
)

DIAGNOSTICS: dict[str, Callable] = dict(
    sxrv=lambda **kwargs: ("brightness", PinholeCamera("sxrv", **kwargs)),
    sxrh=lambda **kwargs: ("brightness", PinholeCamera("sxrh", **kwargs)),
    kb5v=lambda **kwargs: ("brightness", PinholeCamera("kb5v", **kwargs)),
    kb5h=lambda **kwargs: ("brightness", PinholeCamera("kb5h", **kwargs)),
    cxg6=lambda **kwargs: ("conc", ChargeExchangeSpectrometer("cxg6", element="ne")),
    cxd6=lambda **kwargs: ("conc", ChargeExchangeSpectrometer("cxd6", element="ne")),
    cxs4=lambda **kwargs: ("conc", ChargeExchangeSpectrometer("cxs4", element="be")),
    cxh4=lambda **kwargs: ("conc", ChargeExchangeSpectrometer("cxh4", element="be")),
    cxg6_vtor=lambda **kwargs: (
        "vtor",
        ChargeExchangeSpectrometer("cxg6", element="ne"),
    ),
    cxd6_vtor=lambda **kwargs: (
        "vtor",
        ChargeExchangeSpectrometer("cxd6", element="ne"),
    ),
    cxs4_vtor=lambda **kwargs: (
        "vtor",
        ChargeExchangeSpectrometer("cxs4", element="be"),
    ),
    cxh4_vtor=lambda **kwargs: (
        "vtor",
        ChargeExchangeSpectrometer("cxh4", element="be"),
    ),
    cxg6_zeff=lambda **kwargs: ("zeff_avrg", EffectiveCharge("cxg6")),
    cxd6_zeff=lambda **kwargs: ("zeff_avrg", EffectiveCharge("cxd6")),
    cxs4_zeff=lambda **kwargs: ("zeff_avrg", EffectiveCharge("cxs4")),
    cxh4_zeff=lambda **kwargs: ("zeff_avrg", EffectiveCharge("cxh4")),
    cxg6_base=lambda **kwargs: (
        "spectra",
        BremsstrahlungSpectrometer("cxg6", central_wavelength=527.0),
    ),
    cxd6_base=lambda **kwargs: (
        "spectra",
        BremsstrahlungSpectrometer("cxd6", central_wavelength=527.0),
    ),
    cxs4_base=lambda **kwargs: (
        "spectra",
        BremsstrahlungSpectrometer("cxs4", central_wavelength=468.5),
    ),
    cxh4_base=lambda **kwargs: (
        "spectra",
        BremsstrahlungSpectrometer("cxh4", central_wavelength=468.5),
    ),
    ks3v=lambda **kwargs: ("zeff_avrg", EffectiveCharge("ks3v")),
    ks3h=lambda **kwargs: ("zeff_avrg", EffectiveCharge("ks3h")),
    ks3v_base=lambda **kwargs: (
        "spectra",
        BremsstrahlungSpectrometer("ks3v", central_wavelength=523.0),
    ),
    ks3h_base=lambda **kwargs: (
        "spectra",
        BremsstrahlungSpectrometer("ks3h", central_wavelength=523.0),
    ),
)

ADF11 = deepcopy(ADF11)
ADF11.update(
    {
        "n": deepcopy(ADF11["ne"]),
        "ni": {
            "scd": "85",
            "acd": "85",
            "ccd": "89",
            "plt": "89",
            "prb": "89",
            "prc": "89",
            "pls": "15",
            "prs": "15",
        },
        "w": {
            "scd": "50",
            "acd": "50",
            "ccd": "89",
            "plt": "41",
            "prb": "50",
            "prc": "89",
            "pls": "15",
            "prs": "15",
        },
    }
)


def load(
    configfile: Union[str, Path] = "config.yaml",
    resultsfile: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, Any], Plasma, Diagnostics, Inputs,]:
    """
    Load config and :py:`pickle` of input data, building inputs if they don't exist
    """
    configfile = Path(configfile)
    assert configfile.exists()
    config = deepcopy(DEFAULTS)
    with open(configfile, "r") as f:
        config.update(yaml.safe_load(f))

    for key in ("pulse", "tstart", "tend"):
        if key not in config:
            raise UserWarning(f"{key} required in {configfile}")

    with open(configfile, "w+") as f:
        yaml.safe_dump(config, f, sort_keys=True, default_flow_style=None)

    inputfile = configfile.parent / "inputs.pkl"
    if inputfile.exists():
        print(f"Loading from {inputfile}")
        plasma, diagnostics, inputs = load_state(file=inputfile)
    else:
        print(f"Building plasma for pulse {config['pulse']}")
        plasma, diagnostics, inputs = make_inputs(**config)
    save_state(file=inputfile, plasma=plasma, diagnostics=diagnostics, inputs=inputs)

    if resultsfile is None:
        resultsfile = "results.nc"
    resultsfile = Path(resultsfile)
    if not resultsfile.is_absolute():
        resultsfile = configfile.parent / resultsfile
    if resultsfile.exists():
        results = xr.load_dataset(resultsfile)
        for t in np.array(results.t.data, ndmin=1):
            impurities: tuple[str, ...] = tuple(
                val
                for val in (str(results.highz), str(results.midz), str(results.lowz))
                if val != "None"
            )
            xknots = results.xknots
            profiler = make_plasma_profiler(
                plasma,
                impurities,
                xknots,
                t,
            )[0]
            recover_profiles(
                profiler,
                (
                    results.results.sel(t=t)
                    if "t" in results.results.dims
                    else results.results
                ),
                impurities,
                xknots,
                time=t,
            )
            plasma = profiler.plasma

    return config, plasma, diagnostics, inputs


def save_state(
    file: Path, plasma: Plasma, diagnostics: Diagnostics, inputs: Inputs
) -> None:
    file = Path(file).expanduser().resolve()
    with open(file, "wb+") as f:
        pkl.dump(
            dict(plasma=plasma, diagnostics=diagnostics, inputs=inputs),
            f,
            protocol=-1,
        )


def load_state(file: Path) -> Tuple[Plasma, Diagnostics, Inputs]:
    import shutil
    import tempfile
    import time

    file = Path(file).expanduser().resolve()
    with tempfile.TemporaryDirectory() as tempdir:
        tempfile = Path(shutil.copy(str(file), str(Path(tempdir) / file.name)))
        time.sleep(5)
        with open(tempfile, "rb") as f:
            data = pkl.load(f)
    plasma: Plasma = data["plasma"]
    diagnostics: Diagnostics = data["diagnostics"]
    inputs: Inputs = data["inputs"]
    for diag in diagnostics.values():
        diag.model.set_plasma(plasma)
    return plasma, diagnostics, inputs


def fetch_bolt(
    plasma: Plasma,
    reader: JETReader,
    uid: str = "jetppf",
    instrument: str = "bolt",
    revision: int = 0,
) -> xr.DataArray:
    bolt_times = (
        reader.reader_utils._get_signal(uid, instrument, "dt", revision)[0]
        .dimensions[0]
        .data
    )
    target_times = sorted(
        set([np.argmin(np.abs(bolt_times - float(t))) + 1 for t in plasma.t])
    )
    data: List[xr.DataArray] = []
    for target in target_times:
        signal, _ = reader.reader_utils._get_signal(
            uid, instrument, "pr{:02d}".format(target), revision
        )
        data.append(
            xr.DataArray(
                data=signal.data,
                coords={
                    "R": signal.dimensions[0].data,
                    "z": signal.dimensions[1].data,
                },
                dims=("R", "z"),
            )
            .interp(
                {
                    "R": plasma.equilibrium.rhop.R,  # type: ignore
                    "z": plasma.equilibrium.rhop.z,  # type: ignore
                }
            )
            .expand_dims(dim={"t": [bolt_times[target - 1]]})
        )
    bolt = xr.concat(data, dim="t")
    rhop, theta, _ = plasma.equilibrium.flux_coords(bolt.R, bolt.z, bolt.t)
    return bolt.assign_coords(
        {
            "rhop": (("R", "z"), rhop.mean("t").data),
            "theta": (("R", "z"), theta.mean("t").data),
        }
    )


def bolt_back_calculated(
    bolt: xr.DataArray, bolo: xr.DataArray, rho_max: float = 1.0
) -> xr.DataArray:
    """
    Return data from BOLT line-integrated on BOLO LOS.

    :param bolt: Tomographic reconstruction data
    :param bolo: Bolometry diagnostic data
    :param rho_max: PLACEHOLDER
    """
    assert "rhop" in bolt.coords.keys()
    bolt = (
        bolo.transform.integrate_on_los(
            bolt.where(bolt.rhop < rho_max, other=0.0),
            bolt.t,
        )
        .assign_attrs({"transform": bolo.transform})
        .transpose(*bolo.dims)
    )
    bolt.name = bolo.name
    return bolt


def fit_to_profiler(
    data: xr.DataArray,
    xspl: xr.DataArray,
    t: Union[int, float],
    avrg: float = 0.01,
    datatype: Optional[str] = None,
) -> ProfilerBase:
    if datatype is None:
        datatype = str(data.name)
    assert datatype is not None
    transform = data.transform
    data = data.where((data.t >= (t - avrg)) & (data.t <= (t + avrg)), drop=True).mean(
        "t"
    )
    if "rhop" in data.dims:
        data = data.interp(rhop=xspl.to_numpy())
    else:
        rhop, _ = transform.convert_to_rho_theta(t=t)
        data = (
            data.assign_coords({"rhop": ("channel", rhop.data)})
            .swap_dims({"channel": "rhop"})
            .drop_vars("channel")
            .interp(rhop=xspl.to_numpy())
        )

    data = data.where(data > 0).ffill("rhop").bfill("rhop")
    filtered = xr.ones_like(data) * savgol_filter(
        data, max((data.rhop.size // 4), 2), 1
    )
    xknots = np.asarray([*np.linspace(0, 1.0, 5).tolist(), 1.05])
    y0 = np.log10(float(filtered.interp(rhop=0).data))
    init_params = [1.0] * (len(xknots) - 2)
    profiler = ProfilerCubicSpline(
        datatype=datatype,
        xspl=xspl.to_numpy(),
        parameters={"xknots": xknots, **_profile_parameters([y0, *init_params, 0.0])},
    )

    def resid(x) -> NDArray:
        profiler.set_parameters(**_profile_parameters([y0, *list(x), 0.0]))
        _res = np.sum((data - profiler()).to_numpy() ** 2)
        if isinstance((spline := getattr(profiler, "spline")), CubicSpline):
            _res += 0.01 * np.trapz(spline(profiler.xspl) ** 2)
        return np.sqrt(_res)

    opt = least_squares(resid, init_params, bounds=(0.1, 10.0))
    profiler.set_parameters(**_profile_parameters([y0, *list(opt.x), 0.0]))
    return profiler


def reformat_inputs(
    data: xr.DataArray,
    xspl: xr.DataArray,
    t: Union[int, float],
    avrg: float = 0.01,
) -> xr.DataArray:
    transform = getattr(data, "transform", None)
    data = data.where((data.t >= (t - avrg)) & (data.t <= (t + avrg)), drop=True).mean(
        "t"
    )
    if "rhop" in data.dims:
        data = data.interp(rhop=xspl.to_numpy())
    else:
        assert transform is not None
        rhop, _ = transform.convert_to_rho_theta(t=t)
        data = (
            data.assign_coords({"rhop": ("channel", rhop.data)})
            .where(~rhop.isnull(), drop=True)
            .swap_dims({"channel": "rhop"})
            .drop_vars("channel")
            .interp(rhop=xspl.to_numpy())
        )

    return data.where(data > 0).ffill("rhop").bfill("rhop")
    # data = data.where(data > 0).bfill("rhop").fillna(0.0)
    # return xr.ones_like(data) * savgol_filter(data, max((data.rhop.size // 4), 2), 1)


def make_atomic_data(
    pulse: int,
    elements: Iterable[str],
    adf11: dict[str, Any],
    reader: Optional[ADASReader] = None,
) -> Tuple[Dict[str, FractionalAbundance], Dict[str, Dict[str, PowerLoss]]]:
    """Fetch ADAS atomic data for fractional abundance and power loss."""
    if reader is None:
        local_adas = Path("/home/adas/adas/").resolve()
        if local_adas.exists():
            reader = ADASReader(local_adas)
        else:
            reader = ADASReader()

    if pulse >= 92504:
        # All changed to 250μm windows
        adf11_sxrv = {
            "w": {"plt": "88_250", "prb": "88_250"},
            "ni": {"plt": "88_250", "prb": "88_250"},
            "mo": {"plt": "88_250", "prb": "88_250"},
            "be": {"plt": "88_250", "prb": "88_250"},
            "ne": {"plt": "88_250", "prb": "88_250"},
            "n": {"plt": "88_250", "prb": "88_250"},
            "h": {"plt": "88_250", "prb": "88_250"},
        }
        adf11_sxrh = adf11_sxrv
    else:
        # Different windows for vertical and horizontal systems
        adf11_sxrv = {
            "w": {"plt": "88_250", "prb": "88_250"},
            "ni": {"plt": "88_250", "prb": "88_250"},
            "mo": {"plt": "88_250", "prb": "88_250"},
            "be": {"plt": "88_250", "prb": "88_250"},
            "ne": {"plt": "88_250", "prb": "88_250"},
            "n": {"plt": "88_250", "prb": "88_250"},
            "h": {"plt": "88_250", "prb": "88_250"},
        }
        adf11_sxrh = {
            "w": {"plt": "88_350", "prb": "88_350"},
            "ni": {"plt": "88_350", "prb": "88_350"},
            "mo": {"plt": "88_350", "prb": "88_350"},
            "be": {"plt": "88_350", "prb": "88_350"},
            "ne": {"plt": "88_350", "prb": "88_350"},
            "n": {"plt": "88_350", "prb": "88_350"},
            "h": {"plt": "88_250", "prb": "88_250"},
        }

    Te, Ne, *_ = default_profiles()
    fract_abu: Dict[str, FractionalAbundance] = {}
    power_loss_tot: Dict[str, PowerLoss] = {}
    power_loss_sxrv: Dict[str, PowerLoss] = {}
    power_loss_sxrh: Dict[str, PowerLoss] = {}
    for elem in elements:
        scd = reader.get_adf11("scd", elem, adf11[elem]["scd"])
        acd = reader.get_adf11("acd", elem, adf11[elem]["acd"])
        ccd = reader.get_adf11("ccd", elem, adf11[elem]["ccd"])
        fract_abu[elem] = FractionalAbundance(scd, acd, ccd=ccd)
        F_z_t = fract_abu[elem](Te=Te, Ne=Ne)

        plt = reader.get_adf11("plt", elem, adf11[elem]["plt"])
        prb = reader.get_adf11("prb", elem, adf11[elem]["prb"])
        prc = reader.get_adf11("prc", elem, adf11[elem]["prc"])
        power_loss_tot[elem] = PowerLoss(plt, prb, prc=prc)
        power_loss_tot[elem](Te=Te, F_z_t=F_z_t, Ne=Ne)

        try:
            filtered_sxrv = ADASReader(Path("~/Documents/adas/").expanduser().resolve())
            plt = filtered_sxrv.get_adf11("plt", elem, adf11_sxrv[elem]["plt"])
            prb = filtered_sxrv.get_adf11("prb", elem, adf11_sxrv[elem]["prb"])
        except (KeyError, FileNotFoundError, AssertionError):
            print(f"No SXRV-filtered data available for element {elem}")
            plt = xr.zeros_like(plt)
            prb = xr.zeros_like(prb)
        power_loss_sxrv[elem] = PowerLoss(plt, prb, prc=prc)
        power_loss_sxrv[elem](Te=Te, F_z_t=F_z_t, Ne=Ne)

        try:
            filtered_sxrh = ADASReader(Path("~/Documents/adas/").expanduser().resolve())
            plt = filtered_sxrh.get_adf11("plt", elem, adf11_sxrh[elem]["plt"])
            prb = filtered_sxrh.get_adf11("prb", elem, adf11_sxrh[elem]["prb"])
        except (KeyError, FileNotFoundError, AssertionError):
            print(f"No SXRH-filtered data available for element {elem}")
            plt = xr.zeros_like(plt)
            prb = xr.zeros_like(prb)
        power_loss_sxrh[elem] = PowerLoss(plt, prb, prc=prc)
        power_loss_sxrh[elem](Te=Te, F_z_t=F_z_t, Ne=Ne)

    return (
        fract_abu,
        dict(
            power_loss_tot=power_loss_tot,
            power_loss_sxrv=power_loss_sxrv,
            power_loss_sxrh=power_loss_sxrh,
        ),
    )


def make_inputs(
    pulse: int,
    tstart: float,
    tend: float,
    nt: int,
    highz: str,
    midz: str,
    lowz: str,
    equilibrium: Tuple[str, str, RevisionLike],
    electron_density: Tuple[str, str, RevisionLike],
    electron_temperature: Tuple[str, str, RevisionLike],
    ion_temperature: Tuple[str, str, RevisionLike],
    toroidal_rotation: Tuple[str, str, RevisionLike],
    diagnostics: List[Tuple[str, str, RevisionLike, float]],
    concentrations: Optional[dict[str, tuple[float, int, int]]] = None,
    impurity_densities: Optional[Dict[str, Tuple[str, str, RevisionLike]]] = None,
    n_rad: int = 41,
    avrg: float = 0.01,
    zeff_rho_max=1.0,
    channels: Optional[Dict[str, list[int]]] = None,
    **_,
) -> Tuple[Plasma, Dict[str, Diagnostic], Dict[str, DataArray]]:
    concentrations = concentrations if concentrations is not None else {}
    impurity_densities = impurity_densities if impurity_densities is not None else {}
    channels = channels if channels is not None else deepcopy(DEFAULTS["channels"])

    tbuf = max([((tend - tstart) / nt), (2 * avrg)])
    server = f"https://{sal if (sal := getfqdn('sal')) != 'sal' else 'sal.jet.uk'}"
    reader = JETReader(  # type: ignore
        pulse,
        tstart - tbuf - 0.1,
        tend + tbuf + 0.1,
        server=server,
    )

    _te, _ne, *_ = default_profiles()
    te_min = float(_te.min().data)
    ne_min = float(_ne.min().data)

    def _get_abst(quantity: str, uid: str, instrument: str, revision: RevisionLike = 0):
        from indica.readers.jetreader import assign_trivial_transform

        factor = {"nefe": 1e19, "tefe": 1e3}.get(quantity, 1.0)
        data, (t, psin), *_ = reader.reader_utils.get_data(
            uid=uid, instrument=instrument, revision=revision, quantity=quantity
        )
        transform = assign_trivial_transform()
        da = xr.DataArray(data * factor, coords={"t": t, "rhop": np.sqrt(psin)})
        return da.where(~da.rhop.isnull(), drop=True).assign_attrs(
            {"transform": transform}
        )

    # Using ne to get time vector
    ne = (
        _get_abst("nefe", *electron_density)
        if electron_density[1].lower() == "abst"
        else reader.get(*electron_density)["ne"]
    )
    ne = ne.where(ne > ne_min, other=ne_min)
    if len(ne.t) > nt:
        istart = int(np.abs(tstart - ne.t).argmin())
        iend = int(np.abs(tend - ne.t).argmin())
        tfull = ne.t.data[istart : iend + 1]
        tfull = tfull[:: (len(tfull) // nt)]
        tstart, tend = float(tfull[0]), float(tfull[-1])
        dt = tfull[1] - tfull[0]
    else:
        dt = (tend - tstart) / nt

    try:
        assert any((val is not None for val in (highz, midz, lowz)))
    except AssertionError:
        raise UserWarning("Must provied at least one of highz, midz and/or lowz!")
    impurities = [val for val in (highz, midz, lowz) if val is not None]

    plasma = Plasma(
        tstart,
        tend,
        dt,
        impurities=tuple(
            sorted(
                set([*impurities, *concentrations.keys(), *impurity_densities.keys()]),
                key=lambda elem: get_element_info(elem)[0],
                reverse=True,
            )
        ),
        machine="jet",
        full_run=False,
        n_rad=n_rad,
    )

    # Define plasma parameters and profiles
    plasma.set_equilibrium(Equilibrium(reader.get(*equilibrium)))
    ne.transform.set_equilibrium(plasma.equilibrium)
    te = (
        _get_abst("tefe", *electron_temperature)
        if electron_temperature[1].lower() == "abst"
        else reader.get(*electron_temperature)["te"]
    )
    te = te.where(te > te_min, other=te_min)
    te.transform.set_equilibrium(plasma.equilibrium)
    ti = reader.get(*ion_temperature)["ti"]
    ti = ti.where(ti > te_min, other=te_min)
    ti.transform.set_equilibrium(plasma.equilibrium)
    vtor = reader.get(*toroidal_rotation)["vtor"]
    vtor.transform.set_equilibrium(plasma.equilibrium)
    ni = {key: reader.get(*val)["dens"] for key, val in impurity_densities.items()}
    for val in ni.values():
        val.transform.set_equilibrium(plasma.equilibrium)

    inputs = {
        "electron_density": ne,
        "electron_temperature": te,
        "ion_temperature": ti,
        "toroidal_rotation": vtor,
    }

    # Set atomic data
    plasma.set_adf11(ADF11)
    plasma.fract_abu, lz = make_atomic_data(
        pulse=pulse, elements=plasma.elements, adf11=plasma.adf11
    )
    plasma.power_loss_tot = lz["power_loss_tot"]

    times: list[float] = np.array(plasma.time_to_calculate, ndmin=1).tolist()
    for t in times:
        plasma.electron_density.loc[t, :] = reformat_inputs(ne, plasma.rhop, t, avrg)
        plasma.electron_temperature.loc[t, :] = reformat_inputs(
            te, plasma.rhop, t, avrg
        )
        plasma.ion_temperature.loc[t, :] = reformat_inputs(ti, plasma.rhop, t, avrg)
        plasma.toroidal_rotation.loc[t, :] = reformat_inputs(vtor, plasma.rhop, t, avrg)
        for key, val in ni.items():
            # Assume fully ionised measurement from CXRS, fragile assumption though...
            ion_charge = int(plasma.element_z.sel(element=key))
            plasma.impurity_density.loc[key, t, :] = reformat_inputs(
                val, plasma.rhop, t, avrg
            ) * plasma.fz[key].sel(t=t, ion_charge=ion_charge, method="nearest")
        for element, (conc, _, _) in concentrations.items():
            plasma.set_impurity_concentration(element, conc, t)

    diagnostics_processed: Dict[str, Diagnostic] = {}
    for uid, instrument, revision, weight in diagnostics:
        kwargs = {}
        if instrument.lower() in [val.split("_")[-1] for val in lz.keys()]:
            kwargs["power_loss"] = lz[f"power_loss_{instrument.lower()}"]
        elif "kb5" in instrument.lower():
            kwargs["power_loss"] = plasma.power_loss_tot
        quantity, model = DIAGNOSTICS[instrument](**kwargs)
        if "vtor" in instrument:
            instrument = instrument.split("_")[0]
        if instrument.lower() == "sxrv" and pulse >= 99330:
            continue  # We lost SXRV during DT but the PPFs still exist
        if quantity == "conc" and model.element not in plasma.impurities:
            continue
        if quantity == "conc" and model.element in impurity_densities.keys():
            continue
        if any(
            (
                (
                    "cxd6" in instrument.lower()
                    and f"{instrument.replace('cxd6', 'cxg6')}_{quantity}"
                    in diagnostics_processed.keys()
                ),
                (
                    "cxh6" in instrument.lower()
                    and f"{instrument.replace('cxh6', 'cxs6')}_{quantity}"
                    in diagnostics_processed.keys()
                ),
            )
        ):
            continue
        try:
            measurement = reader.get(uid, instrument, revision)[quantity]
            measurement.transform.set_equilibrium(plasma.equilibrium)
            if "sxr" in instrument.lower():
                # SXR channels start at 1, need to fix in the reader at some point...
                measurement = measurement.assign_coords(
                    {"channel": measurement.channel - 1}
                )
                if np.all((measurement.error == 0.0) | measurement.error.isnull()):
                    measurement["error"] = (xr.ones_like(measurement)) + (
                        0.3 * measurement.mean("channel")
                    )
            ic = None
            if "channel" in measurement.coords:
                ic = channels.get(instrument, measurement.channel.data.tolist())
                if len(ic) == 0:
                    ic = measurement.channel.data.tolist()
                ic = set(ic)
                if "sxr" in instrument.lower():
                    te_map_max = (
                        measurement.transform.map_profile_to_los(
                            plasma.electron_temperature,
                            plasma.electron_temperature.t,
                        )
                        .max("los_position")
                        .sum("beamlet")
                    )
                    for c in te_map_max.where(
                        te_map_max < 1.5e3, drop=True
                    ).channel.data:
                        ic.discard(c + 1)
                if "kb5v" in instrument.lower():
                    for c in measurement.transform.x1:
                        if 2.30 <= measurement.transform.x_end[c] <= 2.95:
                            ic.discard(c + 1)
                if "zeff" in instrument.lower() or "base" in instrument.lower():
                    zeff_rho = measurement.transform.convert_to_rho_theta()[0].sum(
                        "beamlet"
                    )
                    zeff_rho_min = zeff_rho.where(
                        (
                            (zeff_rho.where(zeff_rho > 0).min("los_position"))
                            >= zeff_rho_max
                        ),
                        drop=True,
                    ).channel
                    for c in zeff_rho_min.data:
                        ic.discard(c + 1)
            if "wavelength" in measurement.coords:
                measurement = measurement.assign_coords(
                    {
                        "wavelength": np.array(
                            getattr(model, "central_wavelength", 0.0), ndmin=1
                        )
                    }
                )
            model.transform = measurement.transform
            model.set_plasma(plasma)
            diagnostics_processed[f"{instrument}_{quantity}"] = Diagnostic(
                instrument=instrument,
                quantity=quantity.replace("angf", "vtor"),
                measurement=measurement,
                model=model,
                channels=list(ic),
                weight=weight,
                rescale_factor=2.35 if "sxr" in instrument else 1.0,
            )
        except (NodeNotFound, KeyError):
            print(f"No data found for {uid}/{instrument}:{revision}")

    return plasma, diagnostics_processed, inputs


def make_plasma_profiler(
    plasma: Plasma,
    impurities: tuple[str, ...],
    xknots: tuple[float, ...],
    t: Union[int, float],
    n_particles: int = 24,
    no_asym: bool = False,
) -> Tuple[PlasmaProfiler, tuple[NDArray, NDArray], NDArray,]:
    density_lower_bounds: list[float] = []
    density_upper_bounds: list[float] = []
    density_init_pos: list[float] = []
    density_profilers: dict[str, ProfilerCubicSpline] = {}
    ne = plasma.electron_density.sel(t=t, method="nearest")
    for element in impurities:
        ne_max: float = float(ne.max().data)
        z = int(plasma.element_z.sel(element=element).data)
        y_lower = np.log10(1e-6 * ne_max)
        y_upper = np.log10((1 / z) * ne_max)
        density_init_pos.append(
            np.random.uniform(y_lower, y_upper, size=n_particles).tolist()
        )
        if z <= 20:
            density_lower_bounds.extend([y_lower])
            density_upper_bounds.extend([y_upper])
            continue
        for _ in range(len(xknots[1:-1])):
            density_init_pos.append([1.0] * n_particles)
        density_lower_bounds.extend(
            [y_lower, *np.linspace(0.8, 0.5, len(xknots[1:-1]))]
        )
        density_upper_bounds.extend(
            [y_upper, *np.linspace(1.2, 2.0, len(xknots[1:-1]))]
        )
        density_profilers[f"impurity_density:{element}"] = ProfilerCubicSpline(
            datatype="impurity_density",
            xspl=plasma.rhop.to_numpy(),
            parameters={"xknots": xknots, **_profile_parameters(np.ones_like(xknots))},
        )
    vtor_xknots = [0.0, 0.3, 0.6, 0.9, 1.05]
    if no_asym is True:
        rotation_parameters = [
            1.0,
            *([1.0] * len(vtor_xknots[1:-1])),
            0.0,
        ]
        rotation_lower_bounds = [
            (rotation_parameters[0] - 0.01),
            *([0.99] * (len(rotation_parameters) - 2)),
        ]
        rotation_upper_bounds = [
            (rotation_parameters[0] + 0.01),
            *([1.01] * (len(rotation_parameters) - 2)),
        ]
    else:
        vtor = plasma.toroidal_rotation
        rotation_parameters = [
            np.log10(float(vtor.interp(t=t, rhop=0.0).data)),
            *(
                (
                    vtor.interp(rhop=list(vtor_xknots[1:-1]))
                    / vtor.interp(rhop=list(vtor_xknots[1:-1])).data
                )
                .interp(t=t)
                .fillna(0.0)
                .data.tolist()
            ),
            0.0,
        ]
        rotation_lower_bounds = [
            (rotation_parameters[0] - 0.5),
            *([0.5] * (len(rotation_parameters) - 2)),
        ]
        rotation_upper_bounds = [
            (rotation_parameters[0] + 0.5),
            *([2.0] * (len(rotation_parameters) - 2)),
        ]
    rotation_init_pos = [([val] * n_particles) for val in rotation_parameters[:-1]]
    rotation_profiler = ProfilerCubicSpline(
        datatype="toroidal_rotation",
        xspl=plasma.rhop.to_numpy(),
        parameters={
            "xknots": vtor_xknots,
            **_profile_parameters(np.asarray(rotation_parameters)),
        },
    )
    lower_bounds = np.asarray([*density_lower_bounds, *rotation_lower_bounds]).flatten()
    upper_bounds = np.asarray([*density_upper_bounds, *rotation_upper_bounds]).flatten()
    init_pos = np.asarray([*density_init_pos, *rotation_init_pos])
    profiler = PlasmaProfiler(
        plasma, {**density_profilers, "toroidal_rotation": rotation_profiler}
    )
    return profiler, (lower_bounds, upper_bounds), init_pos.T


def optimise_imurities(
    config: Config,
    plasma: Plasma,
    diagnostics: Diagnostics,
    inputs: Inputs,
    t: Optional[Union[int, float]] = None,
    savedir: Path = Path(".").expanduser().resolve(),
    iters: int = 250,
    pso_options: Optional[Dict[str, float]] = None,
    diag_to_scale: Optional[list[str]] = None,
    verbose: bool = False,
) -> Tuple[Config, Plasma, Diagnostics, Inputs, Results, History, Path,]:
    if t is None:
        t = np.array(plasma.time_to_calculate, ndmin=1)[0]
    try:
        assert t is not None
    except AssertionError as e:
        print("Time to optimise not provided and could not be determined, aborting fit")
        raise e
    xknots: tuple[float, ...] = config.get("xknots", deepcopy(DEFAULTS["xknots"]))
    avrg: float = config.get("avrg", deepcopy(DEFAULTS["avrg"]))
    concentrations = config.get("concentrations", {})
    diag_conf = config["diagnostics"]
    for (_, _, _, conf_weight), diag in zip(diag_conf, diagnostics.values()):
        diag.weight = conf_weight
        chan = config["channels"].get(diag.instrument, None)
        if chan is None or len(chan) == 0:
            diag.channels = None
        else:
            diag.channels = chan
    diag_to_scale = [
        diagnostics[key].instrument
        for key in sorted(diagnostics.keys())
        if diagnostics[key].instrument
        in config.get("diag_to_scale", deepcopy(DEFAULTS["diag_to_scale"]))
    ]
    print(f"--- Optimising {t=:.3f}s ---")
    impurities: tuple[str, ...] = tuple(
        val
        for val in (config.get("highz"), config.get("midz"), config.get("lowz"))
        if val is not None
    )
    for element in impurities:
        plasma.set_impurity_concentration(element, 0.0, t=t)
    n_particles = 64
    (profiler, bounds, init_pos,) = make_plasma_profiler(
        deepcopy(plasma),
        impurities=impurities,
        xknots=xknots,
        t=t,
        n_particles=n_particles,
        no_asym=config.get("no_asym", False),
    )
    options = pso_options if pso_options is not None else DEFAULT_OPTIONS
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(init_pos[0]),
        options=options,
        bounds=bounds,
        init_pos=init_pos,
        ftol=0.01,
        ftol_iter=15,
    )
    rho_2d = plasma.equilibrium.flux_coords(plasma.R, plasma.z, t=t)[0]
    R_0 = plasma.equilibrium.rmjo.interp(t=t, rhop=rho_2d).drop_vars("rhop")
    R_lfs = plasma.equilibrium.R_lfs(plasma.rhop, t)[0]
    cost, results = optimizer.optimize(
        costfn,
        iters=iters,
        n_processes=cpu_count(),
        verbose=verbose,
        profiler=profiler,
        diagnostics=diagnostics,
        impurities=impurities,
        xknots=xknots,
        diag_to_scale=diag_to_scale,
        rho_2d=rho_2d,
        R_0=R_0,
        R_lfs=R_lfs,
        t=t,
        avrg=avrg,
    )
    print(f"Optimal {cost = :.3f}")
    print(f"Impurities: {results[:-4]}")
    print(f"Rotation: {results[-4:]}")
    history = dict(
        cost=optimizer.cost_history,
        pos=optimizer.pos_history,
        mean_neighbor=optimizer.mean_neighbor_history,
        mean_pbest=optimizer.mean_pbest_history,
    )
    profiler = recover_profiles(
        profiler,
        results,
        impurities,
        xknots,
        t,
    )
    output = convert_to_dataset(
        config=config,
        profiler=profiler,
        diagnostics=diagnostics,
        inputs=inputs,
        results=results,
        history=history,
        xknots=xknots,
        concentrations=concentrations,
        diag_to_scale=diag_to_scale,
        rho_2d=rho_2d,
        R_0=R_0,
        R_lfs=R_lfs,
        t=t,
        avrg=avrg,
    )
    output.to_netcdf(savedir / f"results_{t:.3f}.nc")
    return (
        config,
        profiler.plasma,
        diagnostics,
        inputs,
        np.asarray(results),
        history,
        savedir,
    )


def _format_for_ppf(data: DataArray, comm: str) -> Optional[Dict[str, NDArray]]:
    try:
        import ppf  # type: ignore
    except ImportError:
        raise ImportError("Could not load PPF module, required to write PPFs")

    t = data.coords[data.dims[0]]
    tunits = t.attrs.get("units", "s" if data.dims[0].lower() == "t" else "")
    tkind = t.to_numpy().dtype.kind
    if not (tkind == "f" or tkind == "i"):
        t = xr.ones_like(t, dtype=int) * np.arange(len(t))
        tunits = ""
        tkind = "i"
    if data.ndim == 1:
        if data.size == 0:
            return None
        return {
            "data": data.to_numpy(),
            "t": t.to_numpy(),
            "irdat": ppf.ppfwri_irdat(1, len(t.to_numpy())),
            "ihdat": ppf.ppfwri_ihdat(
                data.attrs.get("units", "").replace("$", ""),
                "",
                tunits.replace("$", ""),
                data.to_numpy().dtype.kind,
                "I",
                tkind,
                comm,
            ),
        }
    elif data.ndim == 2:
        data = data.dropna(data.dims[-1])
        if data.size == 0:
            print(f"No data to save for {data.name}!")
            return None
        x = data.coords[data.dims[-1]]
        if data.dims[-1] == "channel":
            x = x + 1
        xunits = x.attrs.get(
            "units",
            {
                "R": "m",
                "rhop": "rho",
                "channel": "channel",
            }.get(data.dims[-1], ""),
        )
        xkind = x.to_numpy().dtype.kind
        if not (xkind == "f" or xkind == "i"):
            x = xr.ones_like(x, dtype=int) * np.arange(len(x))
            xunits = ""
            xkind = "i"
        return {
            "data": data.transpose("t", ...).to_numpy(),
            "t": t.to_numpy(),
            "x": x.to_numpy(),
            "irdat": ppf.ppfwri_irdat(len(x.to_numpy()), len(t.to_numpy())),
            "ihdat": ppf.ppfwri_ihdat(
                data.attrs.get("units", "").replace("$", ""),
                xunits.replace("$", ""),
                tunits.replace("$", ""),
                data.to_numpy().dtype.kind,
                xkind,
                tkind,
                comm,
            ),
        }
    else:
        raise UserWarning(f"ndim={data.ndim} ({data.name}) not supported by PPF system")


def dataset_to_ppf(
    data: Dataset,
    dda: str = "idca",
    ppfuid: str = getuser(),
    ddastat: int = 0,
):
    """Write contents of :py:`Dataset` to PPF"""
    try:
        import ppf  # type: ignore
    except ImportError:
        raise ImportError("Could not load PPF module, required to write PPFs")
    ppf.ppf_preference_set_bool(ppf.PPF_PREFS_DDA_LONG_NAMES, True)
    ppf.ppf_preference_set_bool(ppf.PPF_PREFS_DTYPE_LONG_NAMES, True)

    if "t" not in data.dims:
        data = data.expand_dims({"t": [float(data.t.data)]}, axis=0)

    dtypes_comm: dict[str, tuple[str, str]] = {
        "fitp": ("results", "Final fit parameters"),
        "fitr": ("residual", "Final fit residual"),
        "scle": ("rescale_factors", "Diagnostic scale factors"),
        "rhop": ("rhop", "Rho poloidal"),
        "impp": ("rho_2d", "Impact parameter"),
        "rmag": ("rmag", "Magnetic axis (R)"),
        "zmag": ("zmag", "Magnetic axis (z)"),
        "xknt": ("xknots", "x-knots for splines"),
        "te  ": ("electron_temperature", "Electron temperature fit"),
        "ne  ": ("electron_density", "Electron density fit"),
        "ti  ": ("ion_temperature", "Ion temperature fit"),
        "angf": ("toroidal_rotation", "Toroidal rotation fit"),
        "n   ": ("ion_density_2d", "{} density"),
        "c   ": ("concentration_2d", "{} concentration"),
        "a   ": ("asymmetry_parameter", "{} asymmetry parameter"),
        "z   ": ("effective_charge", "{} zeff contrib"),
        "ptot": ("prad_tot", "{} radiated power"),
        "avfl": ("total_radiation", "rho avrg radiated power"),
        "toif": ("total_radiation", "Rad power inside rho"),
        "vuv_dens": ("initial_density", "Density from VUV"),
        "vuv_dens_rho": ("initial_density_rho", "VUV density rho"),
    }
    for diag in data.diagnostics:
        dtypes_comm.update(
            {
                f"{diag}_model": (
                    f"{diag}_model",
                    f"{diag.replace('_', ' ')} model",
                ),
                f"{diag}_measurement": (
                    f"{diag}_measurement",
                    f"{diag.replace('_', ' ')} measurement",
                ),
                f"{diag}_impact_parameter": (
                    f"{diag}_impact_parameter",
                    f"{diag.replace('_', ' ')} impact parameter",
                ),
            }
        )
    to_write: dict[str, Optional[dict[str, Any]]] = {}

    for dtype, (name, comm) in dtypes_comm.items():
        dat = getattr(data, name, None)
        if dat is None:
            continue
        if "R" in dat.dims and "z" in dat.dims:
            dat = dat.interp(z=data.zmag)
        if "beamlet" in dat.dims:
            dat = dat.sum("beamlet")
        if "vuv_dens" in dtype:
            rhop = data.initial_density_rho.where(
                data.initial_density_rho.notnull(), other=0.0
            )
            if "rho" in dtype:
                dat = rhop
            else:
                dat = dat.sel(rhop=rhop)
        if dtype == "toif":
            _dat = xr.zeros_like(dat).transpose("element", ...)
            vol = data.volume
            rhop = vol.rhop.data
            for r in rhop:
                _dat.loc[dict(rhop=r)] = np.trapz(
                    dat.where(_dat.rhop <= r, drop=True).transpose("element", ...),
                    vol.where(vol.rhop <= r, drop=True),
                )
            dat = _dat
        if "element" in dat.dims:
            for element in data.impurities:
                _prefix = dtype.strip()
                _dtype_sep = (
                    _prefix + str(element)
                    if len(_prefix) == 1
                    else f"{_prefix}_{element}"
                )
                to_write[_dtype_sep] = _format_for_ppf(
                    dat.sel(element=element),
                    comm.format(element.capitalize()),
                )
            if "radiation" in name.lower() or "prad" in name.lower():
                to_write[dtype.strip()] = _format_for_ppf(
                    dat.sum("element"),
                    comm.format("Total"),
                )
            elif "effective_charge" in name.lower():
                to_write["zeff"] = _format_for_ppf(
                    dat.sum("element"),
                    "Effective charge",
                )
        else:
            to_write[dtype.strip()] = _format_for_ppf(dat, comm)

    try:
        ppf.ppfuid(ppfuid, "W")
        pulse = int(data.pulse.data)
        time, date, ier = ppf.pdstd(pulse)
        assert ier == 0
        comm = "InDiCA Impurity Profiles"
        assert ppf.ppfopn(pulse, date, time, comm) == 0
        iwdat, ierr, ixv, ixt = {}, {}, {}, {}
        config = deepcopy(data.config)
        for dtype, ddesc in to_write.items():
            if ddesc is None:
                continue
            data = ddesc["data"]
            t = ddesc["t"]
            x = ddesc.get("x")
            irdat = ddesc["irdat"]
            ihdat = ddesc["ihdat"]
            irdat[8] = ixt.get(hash(t.tobytes()), irdat[8])
            if x is not None:
                irdat[7] = ixv.get(hash(x.tobytes()), irdat[7])
            iwdat[dtype], ierr[dtype] = ppf.ppfwri(
                pulse, dda, dtype, irdat, ihdat, data, x, t
            )
            ixt[hash(t.tobytes())] = iwdat[dtype][8]
            if x is not None:
                ixv[hash(x.tobytes())] = iwdat[dtype][7]
        (ierr["ppfwri_json"],) = ppf.ppfwri_dda_json_as_string(dda, config)
        assert ppf.ddaclo(ddastat, comm) == 0
        seq, ier = ppf.ppfclo(pulse, "InDiCA", 1)
        ierr["ppfclo"] = ier
    except Exception as e:
        ppf.ppfabo()
        raise e
    return seq, iwdat, ierr


def main(
    configfile: Union[str, Path] = "config.yaml",
    t: int = 0,
    inputs_only: bool = False,
    verbose: bool = False,
) -> Tuple[
    Dict[str, Any],
    Plasma,
    Diagnostics,
    Inputs,
    Optional[Results],
    Optional[History],
    Optional[Path],
]:
    config, plasma, diagnostics, inputs = load(configfile)

    if inputs_only:
        return config, plasma, diagnostics, inputs, None, None, None

    times: list[float] = np.array(plasma.time_to_calculate, ndmin=1).tolist()

    return optimise_imurities(
        config=config,
        plasma=plasma,
        diagnostics=diagnostics,
        inputs=inputs,
        t=times[t],
        savedir=Path(configfile).expanduser().resolve().parent,
        iters=config.get("n_iters", 250),
        pso_options=config.get("options", None),
        verbose=verbose,
    )
