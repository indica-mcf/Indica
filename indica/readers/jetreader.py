"""Provides implementation of :py:class:`readers.DataReader` for reading PPF data
produced by JET

"""

from copy import deepcopy
from pathlib import Path
import re
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
import scipy.constants as sc
import xarray as xr
from xarray import DataArray

from indica.abstractio import BaseIO
from indica.configs.readers.jetconf import JETConf
from indica.configs.readers.machineconf import MachineConf
from indica.converters import CoordinateTransform
from indica.converters import LineOfSightTransform
from indica.converters import TransectCoordinates
from indica.converters import TrivialTransform
from indica.numpy_typing import ArrayLike
from indica.numpy_typing import RevisionLike
from indica.readers.datareader import DataReader
from indica.readers.salutils import SALUtils
from indica.readers.surfutils import read_surf_los


class JETReader(DataReader):
    """Class to read JET PPF data using SAL"""

    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        machine_conf: MachineConf = JETConf,
        reader_utils: BaseIO = SALUtils,
        server: str = "https://sal.jet.uk",
        verbose: bool = False,
        default_error: float = 0.05,
        *args,
        **kwargs,
    ):
        super().__init__(
            pulse,
            tstart,
            tend,
            machine_conf=machine_conf,
            reader_utils=reader_utils,
            server=server,
            verbose=verbose,
            default_error=default_error,
            **kwargs,
        )
        self.reader_utils = self.reader_utils(pulse, server)

    def _get_equilibrium(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        data["t"] = data["rbnd_dimensions"][0]
        data["index"] = data["rbnd_dimensions"][1]
        data["xpsin"] = data["f_dimensions"][1]
        uid = data["uid"]
        instrument = data["instrument"]
        revision = data["revision"]
        data["psi_r"], data["psi_r_records"] = self.reader_utils.get_signal(
            uid, instrument, "psir", revision
        )
        data["psi_z"], data["psi_z_records"] = self.reader_utils.get_signal(
            uid, instrument, "psiz", revision
        )
        data["psi"] = data["psi"].reshape(
            (len(data["t"]), len(data["psi_z"]), len(data["psi_r"]))
        )
        data["psi_error"] = data["psi_error"].reshape(
            (len(data["t"]), len(data["psi_z"]), len(data["psi_r"]))
        )
        transform = assign_trivial_transform()
        return data, transform

    def _get_thomson_scattering(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        uid = data["uid"]
        instrument = data["instrument"]
        revision = data["revision"]
        hrts_fit_dda = re.match(r"t[0-9]{3}", instrument.lower())
        if hrts_fit_dda is not None:
            data["R"], data["z"], data["t"], data["ne"], data["te"] = _get_abst_fits(
                uid=uid,
                instrument=instrument,
                revision=revision,
                reader=self,
            )
            data["ne_error"] = np.zeros_like(data["ne"])
            data["te_error"] = np.zeros_like(data["te"])
            data["x"] = data["R"]
            data["y"] = np.zeros_like(data["R"])
            data["channel"] = np.arange(len(data["R"]))
            transform = assign_transect_transform(data)
            return data, transform
        data["R"] = data["z_dimensions"][0]
        data["x"] = data["R"]
        data["y"] = np.zeros_like(data["R"])
        data["t"] = data["te_dimensions"][0]
        data["channel"] = np.arange(len(data["R"]))
        transform = assign_transect_transform(data)
        return data, transform

    def _get_interferometry(
        self, data: dict
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        data = _interferometer_polarimeter_coords(data)
        data["t"] = data["LID3_dimensions"][0]
        data["ne"] = np.array(
            [
                data.get("LID{}".format(i + 1), np.zeros_like(data["LID3"]))
                for i in data["channel"]
            ]
        ).T

        transform = assign_lineofsight_transform(data)
        return data, transform

    def _get_polarimetry(
        self, data: dict
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        data = _interferometer_polarimeter_coords(data)
        data["dphi"] = np.array(
            [
                data.get("FAR{}".format(i + 1), np.zeros_like(data["FAR3"]))
                for i in data["channel"]
            ]
        ).T
        data["t"] = data["FAR3_dimensions"][0]

        transform = assign_lineofsight_transform(data)
        return data, transform

    def _get_cyclotron_emissions(
        self, data: dict
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        channel = np.argwhere(data["gen"][0, :] > 0)[:, 0]
        data["freq"] = data["gen"][15, channel] * 1e9
        data["nharm"] = data["gen"][11, channel]
        data["btot"] = 2 * np.pi * data["freq"] * sc.m_e / (sc.e * data["nharm"])
        data["R"] = data["R"].mean(0)  # Time-varying R
        data["x"] = data["R"]
        data["y"] = np.zeros_like(data["R"])
        data["z"] = np.zeros_like(data["R"])
        data["channel"] = channel
        data["t"] = data["te_dimensions"][0]
        transform = assign_transect_transform(data)
        return data, transform

    def _get_density_reflectometer(
        self, data: dict
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        data["R"] = data["R"].mean(0)  # Time-varying R
        data["x"] = data["R"]
        data["y"] = np.zeros_like(data["R"])
        data["z"] = np.ones_like(data["R"]) * data["z"][0]
        data["channel"] = range(len(data["R"]))
        data["t"] = data["ne_dimensions"][0]
        transform = assign_transect_transform(data)
        return data, transform

    def _get_charge_exchange(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        data["R"] = data["R"].mean(0)
        data["x"] = data["R"]
        data["y"] = np.zeros_like(data["R"])
        data["z"] = data["z"].mean(0)
        data["t"] = data["R_dimensions"][0]
        data["channel"] = np.arange(len(data["R"]))
        if data.get("conc") is not None:
            data["conc"] /= 100
        if data.get("angf") is None:
            # Fall back to ANGF if AFCR isn't available
            data["angf"] = self.reader_utils.get_data(
                data["uid"],
                data["instrument"],
                quantity="angf",
                revision=data["revision"],
            )[0]
        if data.get("zeff_avrg") is None:
            # Fall back to ZFEQ if ZFBR isn't available
            data["zeff_avrg"] = self.reader_utils.get_data(
                data["uid"],
                data["instrument"],
                quantity="zfeq",
                revision=data["revision"],
            )[0]
        data["vtor"] = data["angf"] * data["R"]
        if data.get("conc") is not None:
            data["conc"] /= 100
        transform = assign_transect_transform(data)
        return data, transform

    def _get_radiation(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        uid = data["uid"]
        instrument = data["instrument"]
        revision = data["revision"]
        if "sxr" in instrument.lower():
            quantity = instrument[-1]
            _instrument = "sxr"
            luminosities = []
            channels = []
            for i in range(
                1, self.machine_conf._RADIATION_RANGES[_instrument + "/" + quantity] + 1
            ):
                try:
                    qval, q_dims, _, q_path = self.reader_utils.get_data(
                        uid, _instrument, f"{quantity}{i:02d}", revision
                    )
                    luminosities.append(qval)
                    channels.append(i)
                except Exception:
                    continue
                if data.get("t") is None:
                    data["t"] = q_dims[0]
            data["brightness"] = np.array(luminosities).T
            data["channel"] = channels
        elif "kb5" in instrument.lower():
            _instrument = "bolo"
            quantity = instrument
            qval, qval_dimensions, _, qval_records = self.reader_utils.get_data(
                uid=uid, instrument="bolo", quantity=quantity, revision=revision
            )
            data["brightness"] = qval
            data["t"] = qval_dimensions[0]
            data["brightness_records"] = qval_records
            data["channel"] = np.arange(len(qval_dimensions[1]))
        else:
            raise UserWarning(f"{instrument} unsupported for {__class__}")

        xstart, xend, zstart, zend, ystart, yend = read_surf_los(
            self.machine_conf.SURF_PATH,
            self.pulse,
            _instrument.lower() + "/" + quantity.lower(),
        )
        location = np.asarray([xstart, ystart, zstart])
        direction = np.asarray([xend, yend, zend]) - location
        data["location"] = location.transpose()
        data["direction"] = direction.transpose()
        transform = assign_lineofsight_transform(data)
        return data, transform

    def _get_spectrometer(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        uid: str = data["uid"]
        instrument: str = data["instrument"]
        revision: str = data["revision"]
        if "ks3" in instrument.lower():
            quantity = instrument[-1]
            instrument = instrument[:3]
            qval, qval_dimensions, _, qval_path = self.reader_utils.get_data(
                uid=uid, instrument="ks3", quantity="bas" + quantity, revision=revision
            )
            data["spectra"] = np.expand_dims(qval, -1)
            data["spectra_error"] = np.zeros_like(data["spectra"])
            data["t"] = qval_dimensions[0]
            try:
                data["channel"] = np.arange(len(qval_dimensions[1]))
            except IndexError:
                data["channel"] = np.array([0], ndmin=1)
            data["wavelength"] = np.array(523.0, ndmin=1)
            data["spectra_records"] = qval_path
            los, _ = self.reader_utils.get_signal(
                uid,
                self.machine_conf._BREMSSTRAHLUNG_LOS[instrument],
                "los" + quantity,
                revision,
            )
            data["location"] = np.asarray([[(los[1] / 1000), 0, (los[2] / 1000)]])
            data["direction"] = (
                np.asarray([[(los[4] / 1000), 0, (los[5] / 1000)]]) - data["location"]
            )
        elif instrument.lower().startswith("cx"):
            assert self.pulse >= 92671
            spec = {
                "cxs": "ks5a",
                "cxd": "ks5b",
                "cxf": "ks5c",
                "cxg": "ks5d",
                "cxh": "ks5e",
            }[instrument.lower()[:3]]
            qval, qval_dimensions, _, qval_path = self.reader_utils.get_data(
                uid=uid, instrument=instrument, quantity="base", revision=revision
            )
            wcx0, _ = self.reader_utils._get_signal(uid, instrument, "wcx0", revision)
            trck, _ = self.reader_utils._get_signal(uid, instrument, "trck", revision)
            sav_file = _get_cxrs_los_savfile(pulse=self.pulse, spec=spec)
            tracks = _get_cxrs_active_tracks(
                pulse=self.pulse, spec=spec, trck=trck.data
            )
            data["location"], data["direction"] = _get_cxrs_los_geometry(
                sav_file=sav_file, tracks=tracks
            )
            data["spectra"] = np.expand_dims(qval, -1)
            data["spectra_error"] = np.zeros_like(data["spectra"])
            data["spectra_records"] = qval_path
            data["t"] = qval_dimensions[0]
            data["channel"] = np.arange(len(qval_dimensions[1]))
            data["wavelength"] = np.array(wcx0.data, ndmin=1)
        else:
            raise UserWarning(f"{instrument} unsupported for {__class__}")
        transform = assign_lineofsight_transform(data)
        return data, transform

    def _get_zeff(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        uid: str = data["uid"]
        instrument: str = data["instrument"]
        revision: str = data["revision"]
        if "ks3" in instrument.lower():
            quantity = instrument[-1]
            instrument = instrument[:-1]
            qval, qval_dimensions, _, qval_path = self.reader_utils.get_data(
                uid=uid, instrument="ks3", quantity="zef" + quantity, revision=revision
            )
            data["zeff_avrg"] = qval
            data["t"] = qval_dimensions[0]
            try:
                data["channel"] = np.arange(len(qval_dimensions[1]))
            except IndexError:
                data["channel"] = np.array([0], ndmin=1)
            data["zeff_avrg_records"] = qval_path
            los, _ = self.reader_utils.get_signal(
                uid,
                self.machine_conf._BREMSSTRAHLUNG_LOS[instrument],
                "los" + quantity,
                revision,
            )
            data["location"] = np.asarray([[(los[1] / 1000), 0, (los[2] / 1000)]])
            data["direction"] = (
                np.asarray([[(los[4] / 1000), 0, (los[5] / 1000)]]) - data["location"]
            )
        elif instrument.lower().startswith("cx"):
            assert self.pulse >= 92671
            spec = {
                "cxs": "ks5a",
                "cxd": "ks5b",
                "cxf": "ks5c",
                "cxg": "ks5d",
                "cxh": "ks5e",
            }[instrument.lower()[:3]]
            trck, _ = self.reader_utils._get_signal(uid, instrument, "trck", revision)
            data["location"], data["direction"] = _get_cxrs_los_geometry(
                sav_file=_get_cxrs_los_savfile(pulse=self.pulse, spec=spec),
                tracks=_get_cxrs_active_tracks(
                    pulse=self.pulse, spec=spec, trck=trck.data
                ),
            )
            data["t"] = data["zeff_avrg_dimensions"][0]
            data["channel"] = np.arange(len(data["zeff_avrg_dimensions"][1]))
        else:
            raise UserWarning(f"{instrument} unsupported for {__class__}")
        transform = assign_lineofsight_transform(data)
        return data, transform


def assign_lineofsight_transform(database_results: Dict):
    transform = LineOfSightTransform(
        database_results["location"][:, 0],
        database_results["location"][:, 1],
        database_results["location"][:, 2],
        database_results["direction"][:, 0],
        database_results["direction"][:, 1],
        database_results["direction"][:, 2],
        machine_dimensions=database_results["machine_dims"],
        dl=database_results["dl"],
        passes=database_results["passes"],
    )
    return transform


def assign_transect_transform(database_results: Dict):
    transform = TransectCoordinates(
        database_results["x"],
        database_results["y"],
        database_results["z"],
        machine_dimensions=database_results["machine_dims"],
    )

    return transform


def assign_trivial_transform():
    transform = TrivialTransform()
    return transform


def _interferometer_polarimeter_coords(data: dict) -> Dict[str, Any]:
    data = deepcopy(data)
    x_start, z_start = [], []
    for i, (R, z, a) in enumerate(zip(data["R"], data["z"], data["a"])):
        if i < 4:
            x_start.append(R - (2.5 - z) * np.tan(np.pi / 2 - a))
            z_start.append(2.5)
        elif i >= 4:
            x_start.append(4.5)
            z_start.append(z + (4.5 - R) * np.tan(a))
    data["channel"] = np.arange(len(x_start))
    data["location"] = np.asarray([x_start, [0.0] * len(x_start), z_start]).T
    data["direction"] = (
        np.asarray([data["R"], [0.0] * len(data["R"]), data["z"]]).T - data["location"]
    )
    return data


def _get_cxrs_los_geometry(sav_file: Path, tracks: ArrayLike) -> Any:
    """Read IDL save file to get position and direction for KS5 tracks

    Parameters
    ----------

    sav_file:
        Path to IDL save file to load
    tracks:
        ArrayLike of tracks to select from save file, from JET dtype `TRCK`

    Returns
    -------

    :
        Tuple of position and direction arrays
    """
    import scipy.io as io

    data = io.readsav(sav_file)
    fibres = data.ptrfib.fibres[0]
    numfibview = fibres.numfibview[0].astype(str)
    origin, direction = {}, {}
    for name, pos, vec in zip(
        numfibview,
        fibres.losdef[0].virtualposition_roomtemp_rot[0].cartesian_ref[0],
        fibres.losdef[0].virtualdirection_rot[0].cartesian_ref[0],
    ):
        origin[name] = pos / 1000  # mm -> m
        direction[name] = vec
    return (
        np.asarray([origin[tr] for tr in tracks]),
        np.asarray([direction[tr] for tr in tracks]),
    )


def _setup_idl(pulse: int) -> Any:
    import idlbridge as idlb

    idlb.execute(".reset")
    idlb.execute(
        "!PATH=!PATH + ':' + "
        "expand_path( '+~cxs/idl_spectro/' ) + ':' + "
        "expand_path( '+~cxs/idl_spectro/show' ) + ':' + "
        "expand_path( '+~cxs/ks6read/' ) + ':' + "
        "expand_path( '+~cxs/ktread/' ) + ':' + "
        "expand_path( '+~cxs/kx1read/' ) + ':' + "
        "expand_path( '+~cxs/idl_spectro/kt3d' ) + ':' + "
        "expand_path( '+~cxs/utc' ) + ':' + "
        "expand_path( '+~cxs/instrument_data' ) + ':' + "
        "expand_path( '+~cxs/calibration' ) + ':' + "
        "expand_path( '+~cxs/alignment' ) + ':' + "
        "expand_path( '+/usr/local/idl' ) + ':' + "
        "expand_path( '+/home/CXSE/cxsfit/idl/' ) + ':' + "
        "expand_path( '+~/jet/share/lib' ) + ':' + "
        "expand_path( '+~/jet/share/root/lib' ) + ':' + "
        "expand_path( '+~/jet/share/idl' )"
    )

    idlb.execute(
        "!PATH = !PATH + ':' + expand_path('+/u/cxs/utilities',/all_dirs)+ ':'"
    )

    idlb.execute(
        "!PATH = !PATH + ':' + expand_path('+/u/cxs/instrument_data/namelists')+ ':' + "
        "expand_path('+/u/cxs/instrument_data/jpfnodes')"
    )

    idlb.execute("!PATH = !PATH + ':' + expand_path('+/usr/local/idl')")
    idlb.execute("!PATH = !PATH + ':' + expand_path('+/home/CXSE/cxsfit/idl/')")
    idlb.execute("!PATH = !PATH +':/home/CXSE/cxsfit/idl:'")
    idlb.execute(".compile plot")
    idlb.execute(".compile ppfread")
    idlb.execute(".compile cxf_number_to_text")
    idlb.execute(".compile cxf_read_switches")
    idlb.execute(".compile cxf_decompress_history")

    idlb.put("shot", pulse)
    idlb.execute("julian_date = agm_pulse_to_julian(shot, 'DG')")

    return idlb


def _get_cxrs_los_savfile(pulse: int, spec: str) -> Path:
    """Determine correct sav file for given pulse and spectrometer

    Parameters
    ----------
    pulse:
        Pulse to search for
    spec:
        Spectrometer to search for

    Returns
    -------
    :
        Path to save file
    """
    octant = {"ks5a": 7, "ks5b": 1, "ks5c": 7, "ks5d": 7, "ks5e": 1}.get(spec.lower())
    idlb = _setup_idl(pulse=pulse)
    idlb.execute(
        "str1 = periscope_oct{}_hist_align(julian_date=julian_date)".format(str(octant))
    )
    idlb.execute("file_align = str1.file_align")
    savfile = Path(str(idlb.get("file_align")))
    assert savfile.exists() and savfile.is_file()
    return savfile


def _get_cxrs_active_tracks(pulse: int, spec: str, trck: ArrayLike) -> ArrayLike:
    """Translate tracks used to track names as in geometry save file"""
    idlb = _setup_idl(pulse=pulse)
    idlb.execute(
        "str1={}_hist_fibresetup(pulse=pulse,julian_date=julian_date)".format(str(spec))
    )
    idlb.execute("viewing_position=str1.pulse_setup.viewing_position")
    idlb.execute("track_reshuffle=str1.pulse_setup.ks4fit_track_reshuffle")
    viewing_position = idlb.get("viewing_position")
    track_reshuffle = idlb.get("track_reshuffle")
    assert len(viewing_position) == len(track_reshuffle)
    assert len(trck) >= len(viewing_position)
    active = [viewing_position[i - 1] for i, t in zip(track_reshuffle, trck) if t == 1]
    return active[::-1]


def _get_abst_fits(
    uid: str,
    instrument: str,
    revision: RevisionLike,
    reader: JETReader,
) -> DataArray:
    """"""
    abft = int(reader.reader_utils.get_signal(uid, "abst", "abft", revision)[0])
    absq = int(reader.reader_utils.get_signal(uid, "abft", "absq", abft)[0])
    shif = float(reader.reader_utils.get_signal(uid, "aiba", "shif", absq)[0])
    shir = float(reader.reader_utils.get_signal(uid, "aiba", "shir", absq)[0])
    psif = (
        np.delete(
            reader.reader_utils.get_signal(uid, instrument, "psif", 0)[0][0],
            501,
        )
        + shif
    )
    rmdf = (
        np.delete(
            reader.reader_utils.get_signal(uid, instrument, "rmdf", 0)[0][0],
            501,
        )
        + shir
    )
    psin = reader.reader_utils.get_signal_dims(uid, "abst", "nefe", revision)[0][1]
    R = DataArray(rmdf, dims=("psin",), coords={"psin": ("psin", psif)}).interp(
        psin=psin
    )
    eftp_uid: str = reader.reader_utils._client.get(
        f"/pulse/{reader.reader_utils.pulse}/ppf/signal/{uid}/{instrument}/kan1"
    ).description
    eftp_instrument: str = reader.reader_utils._client.get(
        f"/pulse/{reader.reader_utils.pulse}/ppf/signal/{uid}/{instrument}/kan2"
    ).description
    eftp_revision: int = {
        97490: 797,
        97482: 340,
        97481: 982,
        97492: 298,
        97484: 342,
    }.get(
        reader.reader_utils.pulse,
        int(
            reader.reader_utils._client.get(
                f"/pulse/{reader.reader_utils.pulse}/ppf/signal/{uid}/{instrument}/kan3"
            ).description
        ),
    )
    t = reader.reader_utils.get_signal_dims(uid, "abst", "nefe", revision)[0][0]
    zmag = reader.get(eftp_uid, eftp_instrument.lower(), eftp_revision)["zmag"]
    z = xr.ones_like(R) * float(zmag.sel(t=t, method="nearest", drop=True))
    ne = reader.reader_utils.get_signal(uid, "abst", "nefe", revision)[0]
    te = reader.reader_utils.get_signal(uid, "abst", "tefe", revision)[0]
    return R.data, z.data, t, (ne * 1e19), (te * 1e3)
