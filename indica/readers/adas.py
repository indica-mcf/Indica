"""Base class for reading in ADAS atomic data."""

import datetime
from pathlib import Path
import re
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
from ..datatypes import ADF11_GENERAL_DATATYPES
from ..datatypes import ADF15_GENERAL_DATATYPES
from ..datatypes import ORDERED_ELEMENTS


# TODO: Evaluate this location
DEFAULT_PATH = Path("")
CACHE_DIR = ".indica"


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
            z = int(header[0])
            nd = int(header[1])
            nt = int(header[2])
            zmin = int(header[3]) - 1
            zmax = int(header[4]) - 1
            element_name = header[5][1:].lower()
            assert ORDERED_ELEMENTS.index(element_name) == z
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
                assert isinstance(m, re.Match)
                short_year = int(m.group(3))
                parsed_year = short_year + (
                    1900 if short_year >= now.year % 100 else 2000
                )
                new_date = datetime.date(parsed_year, int(m.group(2)), int(m.group(1)))
                if new_date > date:
                    date = new_date
                data[i, ...] = np.fromfile(f, float, nd * nt, " ").reshape((nt, nd))
        gen_type = ADF11_GENERAL_DATATYPES[quantity]
        spec_type = ORDERED_ELEMENTS[z]
        name = f"{spec_type}_{gen_type}"
        attrs = {
            "datatype": (gen_type, spec_type),
            "date": date,
            "provenance": self.create_provenance(filename, now),
        }
        return DataArray(
            data - 6,
            coords=[
                ("ion_charges", np.arange(zmin, zmax + 1, dtype=int)),
                ("electron_temperature", 10 ** (temperatures)),
                ("electron_density", 10 ** (densities + 6)),
            ],
            name=name,
            attrs=attrs,
        )

    def get_adf15(
        self,
        element: str,
        charge: str,
        filetype: str,
        year="",
    ) -> DataArray:
        """Read data from the specified ADAS file. Different files are available, e.g.:

        pec96][ne / pec96][ne_pju][ne9.dat
        transport_llu][ar16.dat

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
        now = datetime.datetime.now()
        if year == "transport":
            file_component = "transport"
        else:
            file_component = f"pec{year}][{element.lower()}"

        filename = Path(pathname2url(file_component)) / pathname2url(
            f"{file_component}_{filetype.lower()}]"
            f"[{element.lower()}{charge.lower()}.dat"
        )
        with self._get_file("adf15", filename) as f:
            header = f.readline().strip().lower()
            match = r"(\d+).+/(\S+).*\:(.*)photon"
            m = re.search(match, header, re.I)
            if not isinstance(m, re.Match):
                match = r"(\d+).+/(\S+).*\+(.*)photon"
                m = re.search(match, header, re.I)

            assert isinstance(m, re.Match)
            ntrans = int(m.group(1))
            element_name = m.group(2).strip().lower()
            charge_state = int(m.group(3))
            assert element_name == element.lower()
            try:
                assert charge_state == int(charge)
            except ValueError:
                m = re.search(r"(\d+)(\S+)", charge, re.I)
                charge_tmp = m.group(1)
                assert charge_state is int(charge_tmp)

            # Read first section header to build arrays outside of reading loop
            while True:
                section_header = f.readline().strip().lower()
                match = r"(\d+.\d+)\s+(\d+)\s+(\d+).+type=(\S+)/.+/isel.+=\s+(\d+)"
                m = re.search(match, section_header, re.I)
                if not isinstance(m, re.Match):
                    match = (
                        r"(\d+.\d)\s?\S?\s+(\d+)\s+(\d+).+"
                        r"type\s?=\s?(\S+).+isel\s?=\s+(\d+)"
                    )
                    m = re.search(match, section_header, re.I)
                if isinstance(m, re.Match):
                    break
            section_header_match = match

            assert isinstance(m, re.Match)
            nd = int(m.group(2))
            nt = int(m.group(3))
            data = np.empty((ntrans, nt, nd))
            ttype = [""] * ntrans
            tindex = np.empty(ntrans)
            wavelength = np.empty(ntrans)

            # Read Photon Emissivity Coefficient rates
            for i in range(ntrans):
                if i > 0:
                    section_header = f.readline().strip().lower()
                m = re.search(section_header_match, section_header, re.I)
                assert isinstance(m, re.Match)
                assert int(m.group(5)) - 1 == i
                tindex[i] = i + 1
                ttype[i] = m.group(4)
                wavelength[i] = float(m.group(1))  # (Angstroms)
                densities = np.fromfile(f, float, nd, " ")
                temperatures = np.fromfile(f, float, nt, " ")
                data[i, ...] = np.fromfile(f, float, nd * nt, " ").reshape((nt, nd))

            # Read Transition information from end of file
            match = r"c\s+[isel].+\s+[transition].+\s+[type]"
            while True:
                tmp = f.readline().strip().lower()
                m = re.search(match, tmp, re.I)
                if isinstance(m, re.Match):
                    break
            f.readline().strip().lower()

            while True:
                tmp = f.readline().strip().lower()
                match = (
                    r"c\s+(\d+.)"  # isel
                    r"\s+(\d+.\d+)"  # wavelength
                    r"\s+(\d+)(\(\d\)\d\(.+\d?.\d\))-"  # transition upper level
                    r".+(\d+)(\(\d\)\d\(.+\d?.\d\))"  # transition lower level
                )
                m = re.search(match, tmp, re.I)
                if not isinstance(m, re.Match):
                    match = r"c\s+(\d+.)\s+(\d+.\d+)\s+([n]\=.\d+.-.[n]\=.\d+)"
                    m = re.search(match, tmp, re.I)

                if isinstance(m, re.Match):
                    break
            trans_match = match

            if len(tmp.split(")")) > 3:
                orbitals = True
            else:
                orbitals = False

            transition = []
            for i in tindex:
                if i > 1:
                    tmp = f.readline().strip().lower()
                m = re.search(trans_match, tmp, re.I)
                assert isinstance(m, re.Match)
                assert int(m.group(1)[:-1]) == i
                if orbitals:
                    transition.append(f"{m.group(4)}-{m.group(6)}".replace(" ", ""))
                else:
                    transition.append(m.group(3).replace(" ", ""))

        gen_type = ADF15_GENERAL_DATATYPES[filetype]
        spec_type = element
        name = f"{spec_type}_{gen_type}"
        attrs = {
            "datatype": (gen_type, spec_type),
            "provenance": self.create_provenance(filename, now),
        }

        if nd == nt:
            coords = [
                ("index", tindex),
                ("electron_density", densities * 10 ** 6),  # m**-3
                ("electron_temperature", temperatures),  # eV
            ]
        else:
            coords = [
                ("index", tindex),
                ("electron_temperature", temperatures),  # eV
                ("electron_density", densities * 10 ** 6),  # m**-3
            ]

        pecs = DataArray(
            data * 10 ** -6,
            coords=coords,
            name=name,
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
