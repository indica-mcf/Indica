"""Base class for reading in ADAS atomic data."""

import datetime
from pathlib import Path
import re
from typing import List
from typing import Literal
from typing import TextIO
from typing import Union
from urllib.request import pathname2url
from urllib.request import urlretrieve

import numpy as np
import prov.model as prov
from xarray import DataArray

from .. import session
from ..abstractio import BaseIO
from ..datatypes import DATATYPES

# TODO: Evaluate this location
DEFAULT_PATH = Path("")
CACHE_DIR = ".indica"

ADF11: dict = {
    "h": {
        "scd": "96",
        "acd": "96",
        "ccd": "96",
        "plt": "96",
        "prb": "96",
        "prc": "96",
        "pls": "15",
        "prs": "15",
    },
    "he": {
        "scd": "96",
        "acd": "96",
        "ccd": "96",
        "plt": "96",
        "prb": "96",
        "prc": "96",
        "pls": "15",
        "prs": "15",
    },
    "c": {
        "scd": "96",
        "acd": "96",
        "ccd": "96",
        "plt": "96",
        "prb": "96",
        "prc": "96",
        "pls": "15",
        "prs": "15",
    },
    "ar": {
        "scd": "89",
        "acd": "89",
        "ccd": "89",
        "plt": "89",
        "prb": "89",
        "prc": "89",
        "pls": "15",
        "prs": "15",
    },
    "ne": {
        "scd": "96",
        "acd": "96",
        "ccd": "96",
        "plt": "96",
        "prb": "96",
        "prc": "96",
        "pls": "15",
        "prs": "15",
    },
    "mo": {
        "scd": "89",
        "acd": "89",
        "ccd": "89",
        "plt": "89",
        "prb": "89",
        "prc": "89",
    },
    "w": {
        "scd": "89",
        "acd": "89",
        "ccd": "89",
        "plt": "89",
        "prb": "89",
        "prc": "89",
        "pls": "15",
        "prs": "15",
    },
}


class ADASReader(BaseIO):
    """Class for reading atomic data from ADAS files.

    Parameters
    ----------
    path: Union[str, Path]
        Location from which relative paths should be evaluated.
        Default is to download files from OpenADAS, storing them
        in your home directory for later use.
    session: session.Session
        An object representing the session being run. Contains information
        such as provenance data.

    """

    def __init__(
        self,
        path: Union[str, Path] = DEFAULT_PATH,
        sess: session.Session = session.global_session,
    ):
        path = Path(path)
        self.session = sess
        self.openadas = path == DEFAULT_PATH
        if path == DEFAULT_PATH:
            self.namespace = "openadas"
            self.session.prov.add_namespace(
                self.namespace, "https://open.adas.ac.uk/detail/adf11/"
            )
            self.path = Path.home() / CACHE_DIR / "adas"
        else:
            self.path = path
            self.namespace = "localadas"
            self.session.prov.add_namespace(
                self.namespace, "file:/" + str(self.path.resolve())
            )
        self.prov_id = session.hash_vals(path=self.path)
        self.agent = self.session.prov.agent(self.prov_id)
        self.session.prov.delegation(self.session.agent, self.agent)
        self.entity = self.session.prov.entity(
            self.prov_id, {"path": str(self.path.resolve())}
        )
        self.session.prov.generation(
            self.entity, self.session.session, time=datetime.datetime.now()
        )
        self.session.prov.attribution(self.entity, self.session.agent)

    def close(self):
        """Closes connections to database/files. For this class it does not
        need to do anything."""
        pass

    def get_adf11(self, quantity: str, element: str, year: str) -> DataArray:
        """Read data from the specified ADAS file.

        Parameters
        ----------
        quantity
            The type of data to retrieve. Options: scd, acd, plt, prb,
            plsx, prsx.
        element
            The atomic symbol for the element which will be retrieved.
        year
            The two-digit year label for the data.

        Returns
        -------
        :
            The data in the specified file. Dimensions are density and
            temperature. Each members of the dataset correspond to a
            different charge state.

        """
        now = datetime.datetime.now()
        file_component = f"{quantity}{year}"
        filename = Path(file_component) / f"{file_component}_{element.lower()}.dat"
        with self._get_file("adf11", filename) as f:
            header = f.readline().split()
            # z = int(header[0])
            nd = int(header[1])
            nt = int(header[2])
            zmin = int(header[3]) - 1
            zmax = int(header[4]) - 1
            element_name = header[5][1:].lower()
            f.readline()
            densities = np.fromfile(f, float, nd, " ")
            temperatures = np.fromfile(f, float, nt, " ")
            data = np.empty((zmax - zmin + 1, nt, nd))
            date = datetime.date.min
            for i in range(zmax - zmin + 1):
                section_header = f.readline()
                m = re.search(r"Z1=\s*(\d+)", section_header, re.I)
                assert isinstance(m, re.Match)
                assert int(m.group(1)) - 1 == zmin + i
                m = re.search(
                    r"DATE=\s*(\d?\d)[.\-/](\d\d)[.\-/](\d\d)", section_header, re.I
                )
                # assert isinstance(m, re.Match)  # TEMP for PLSX/PRSX reading
                if isinstance(m, re.Match):
                    short_year = int(m.group(3))
                    parsed_year = short_year + (
                        1900 if short_year >= now.year % 100 else 2000
                    )
                    new_date = datetime.date(
                        parsed_year, int(m.group(2)), int(m.group(1))
                    )
                else:
                    new_date = datetime.datetime.now().date()
                if new_date > date:
                    date = new_date
                data[i, ...] = np.fromfile(f, float, nd * nt, " ").reshape((nt, nd))
        long_name, units = DATATYPES["pec"]
        attrs = {
            "long_name": long_name,
            "units": units,
            "date": date,
            "provenance": self.create_provenance(filename, now),
            "element": element_name,
            "year": year,
        }
        return DataArray(
            10 ** (data - 6),
            coords=[
                ("ion_charges", np.arange(zmin, zmax + 1, dtype=int)),
                ("electron_temperature", 10 ** (temperatures)),
                ("electron_density", 10 ** (densities + 6)),
            ],
            attrs=attrs,
        )

    def get_adf15(
        self,
        element: str,
        charge: str,
        filetype: str,
        year="",
    ) -> DataArray:
        """Read data from the specified ADF15 ADAS file.

        Implementation is capable of reading files with compact and expanded formatting
        e.g. pec96][ne_pju][ne9.dat and pec40][ar_cl][ar16.dat respectively

        Parameters
        ----------
        element
            The atomic symbol for the element which will be retrieved.
        charge
            Charge state of the ion (e.g. 16 for Ar 16+), can also include
            other string for more complicated path (transport_llu][ar15ic.dat
            setting charge to "15ic")
        filetype
            The type of data to retrieve. Options: ic, cl, ca, ls, llu, ...
        year
            The two-digit year label for the data. = "transport" if special
            transport path


        Returns
        -------
        :
            The data in the specified file. Dimensions are density and
            temperature. Each members of the dataset correspond to a
            different charge state.

        """

        def explicit_reshape(data_to_reshape, nd, nt):
            data = np.empty((nd, nt))
            for id in range(nd):
                for it in range(nt):
                    data[id, it] = data_to_reshape[id * nt + it]

            return data

        def build_file_component(year, element):
            file_component = "transport"
            if year != "transport":
                file_component = f"pec{year}][{element.lower()}"

            return file_component

        def file_type(identifier):
            identifier_dict = {
                "+": "compact",
                ":": "expanded",
            }
            file_type = identifier_dict.get(identifier)
            if file_type is None:
                raise ValueError(f"Unknown file header identified ({identifier}).")

            return file_type

        def transition_match(transition_line):
            transition_type = "orbitals"
            match = (
                r"c\s+(\d+.)"  # isel
                r"\s+(\d+.\d+)"  # wavelength
                r"\s+(\d+)(\(\d\)\d\(.+\d?.\d\))-"  # transition upper level
                r".+(\d+)(\(\d\)\d\(.+\d?.\d\))"  # transition lower level
            )
            header_re = re.compile(match)
            m = header_re.search(transition_line)
            if not m:
                transition_type = "n_levels"
                match = r"c\s+(\d+.)\s+(\d+.\d+)\s+([n]\=.\d+.-.[n]\=.\d+)"
                header_re = re.compile(match)
                m = header_re.search(transition_line)
                if not m:
                    raise ValueError(f"Unknown transition formatting ({identifier}).")

            return transition_type, match

        now = datetime.datetime.now()
        file_component = build_file_component(year, element)
        filename = Path(pathname2url(file_component)) / pathname2url(
            f"{file_component}_{filetype.lower()}]"
            f"[{element.lower()}{charge.lower()}.dat"
        )

        header_match = {
            "compact": r"(\d+).+/(\S+).*\+(.*)photon",
            "expanded": r"(\d+).+/(\S+).*\:(.*)photon",
        }
        section_header_match = {
            "compact": r"(\d+.\d+).+\s+(\d+)\s+(\d+).+type\s?"
            r"=\s?(\S+).+isel.+\s+(\d+)",
            "expanded": r"(\d+.\d+)\s+(\d+)\s+(\d+).+type\s?="
            r"\s?(\S+).+isel\s+?=\s+?(\d+)",
        }
        with self._get_file("adf15", filename) as f:
            header = f.readline().strip().lower()
            identifier = file_type(header.split("/")[1][2])

            match = header_match[identifier]
            m = re.search(match, header, re.I)
            assert isinstance(m, re.Match)
            ntrans = int(m.group(1))
            element_name = m.group(2).strip().lower()
            charge_state = int(m.group(3))
            assert element_name == element.lower()
            m = re.search(r"(\d+)(\S*)", charge)
            assert isinstance(m, re.Match)
            extracted_charge = m.group(1)
            if charge_state != int(extracted_charge):
                raise ValueError(
                    f"Charge state in ADF15 file ({charge_state}) does not "
                    f"match argument ({charge})."
                )

            # Read first section header to build arrays outside of reading loop
            match = section_header_match[identifier]
            header_re = re.compile(match)
            m = None
            while not m:
                line = f.readline().strip().lower()
                m = header_re.search(line)
            assert isinstance(m, re.Match)
            nd = int(m.group(2))
            nt = int(m.group(3))
            ttype: List[str] = []
            tindex = np.empty(ntrans)
            wavelength = np.empty(ntrans)

            # Read Photon Emissivity Coefficient rates
            data = np.empty((ntrans, nd, nt))
            for i in range(ntrans):
                m = header_re.search(line)
                assert isinstance(m, re.Match)
                assert int(m.group(5)) - 1 == i
                tindex[i] = i + 1
                ttype.append(m.group(4))
                wavelength[i] = float(m.group(1))  # (Angstroms)

                densities = np.fromfile(f, float, nd, " ")
                temperatures = np.fromfile(f, float, nt, " ")
                data_tmp = np.fromfile(f, float, nd * nt, " ")
                data[i, :, :] = explicit_reshape(data_tmp, nd, nt)
                line = f.readline().strip().lower()

            data = np.transpose(np.array(data), (0, 2, 1))

            # Read Transition information from end of file
            file_end_re = re.compile(r"c\s+[isel].+\s+[transition].+\s+[type]")
            while not file_end_re.search(line):
                line = f.readline().strip().lower()
            _ = f.readline()
            if identifier == "expanded":
                _ = f.readline()
            line = f.readline().strip().lower()
            transition_type, match = transition_match(line)
            transition_re = re.compile(match)

            format_transition = {
                "orbitals": lambda m: f"{m.group(4)}-{m.group(6)}".replace(" ", ""),
                "n_levels": lambda m: m.group(3).replace(" ", ""),
            }
            transition = []
            for i in tindex:
                m = transition_re.search(line)
                assert isinstance(m, re.Match)
                assert int(m.group(1)[:-1]) == i
                transition_tmp = format_transition[transition_type](m)
                transition.append(transition_tmp)
                line = f.readline().strip().lower()
        long_name, units = DATATYPES["pec"]
        attrs = {
            "long_name": long_name,
            "units": units,
            "provenance": self.create_provenance(filename, now),
        }

        coords = [
            ("index", tindex),
            ("electron_temperature", temperatures),  # eV
            ("electron_density", densities * 10**6),  # m**-3
        ]

        pecs = DataArray(
            data * 10**-6,
            coords=coords,
            attrs=attrs,
        )

        # Add extra dimensions attached to index
        pecs = pecs.assign_coords(wavelength=("index", wavelength))  # (A)
        pecs = pecs.assign_coords(
            transition=("index", transition)
        )  # (2S+1)L(w-1/2)-(2S+1)L(w-1/2) of upper-lower levels, no blank spaces
        pecs = pecs.assign_coords(type=("index", ttype))  # (excit, recomb, cx)

        return pecs

    def create_provenance(
        self, filename: Path, start_time: datetime.datetime
    ) -> prov.ProvEntity:
        """Create a provenance entity for the given ADAS file.

        Note that this method just creates the provenance data
        appropriate for the arguments it has been provided with. It
        does not check that these arguments are actually valid and
        that the provenance corresponds to actually existing data.

        """
        end_time = datetime.datetime.now()
        entity = self.session.prov.entity(
            session.hash_vals(filename=filename, start_time=start_time)
        )
        activity = self.session.prov.activity(
            session.hash_vals(agent=self.prov_id, date=start_time),
            start_time,
            end_time,
            {prov.PROV_TYPE: "ReadData"},
        )
        self.session.prov.association(activity, self.agent)
        self.session.prov.association(activity, self.session.agent)
        self.session.prov.communication(activity, self.session.session)
        self.session.prov.derivation(entity, f"{self.namespace}:{filename}", activity)
        self.session.prov.generation(entity, activity, end_time)
        self.session.prov.attribution(entity, self.agent)
        self.session.prov.attribution(entity, self.session.agent)
        return entity

    def _get_file(self, dataclass: str, filename: Union[str, Path]) -> TextIO:
        """Retrieves an ADAS file, downloading it from OpenADAS if
        necessary. It will cache any downloads for later use.

        Parameters
        ----------
        dataclass
            The format of ADAS data in this file (e.g., ADF11).
        filename
            Name of the file to get.

        Returns
        -------
        :
            A file-like object from which the data can be read.

        """
        filepath = self.path / dataclass / filename
        if self.openadas and not filepath.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(
                f"https://open.adas.ac.uk/download/{dataclass}/{filename}", filepath
            )
        return filepath.open("r")

    @property
    def requires_authentication(self) -> Literal[False]:
        """Reading ADAS data never requires authentication."""
        return False
