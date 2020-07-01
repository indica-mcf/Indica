"""Provides an abstract class defining the interface for writing out data."""

from abc import abstractmethod
from typing import Dict
from typing import Union

from prov.model import ProvDocument
from xarray import DataArray
from xarray import Dataset

from ..abstractio import BaseIO
from ..equilibrium import Equilibrium


class DataWriter(BaseIO):
    """An abstract class defining the interface for writing data to the
    disk or datatbases.

    """

    @abstractmethod
    def write(self, uid: str, name: str, *data: Union[Dataset, DataArray]):
        """Write data out to the desired format/database.

        The exact location will be implementation-dependent but will
        include the ``uid`` and ``name`` arguments.

        This is a wrapper function which performs tasks commons to all
        writer classes, such as converting the xarray data structures
        into a form ammenable to output. It will create a new
        :py:class:`xarray.Dataset` containing all data, with
        attributes reformated as necessary:

        - Uncertainty will be made a member of the dataset, with the name
          ``VARIABLE_uncertainty``, where ``VARIABLE`` is the name of the
          variable it is associated with.
        - Dropped data will be merged into the main data and the attribute will
          be replaced with a list of the indices of the dropped channels and
          ``dropped_dim``, the name of the dimension these indices are for.
        - The coordinate transform will be replaced with a JSON serialisation,
          from which it can be recreated. These serialisations will be stored
          in a dictionary attribute for the Dataset as a whole, with each
          DataArray holding the key for its corresponding transform.
        - The PROV attributes will be replaced by the ID for that entity. The
          complete PROV data for the session will be passed to low-level
          writing routines as a separate argument.
        - Datatypes will be serialised as JSON
        - All data will have an ``equilibrium`` attribute, which provides an
          identifier for the equilibrium data (passed to the low-level writer
          in a dictionary).

        Parameters
        ----------
        uid
            User ID (i.e., user that created or wrote this data)
        name
            Name to store this data under, such as a DDA
        data
            The data to be written out. The data will be written as though it
            had been merged into a single :py:class:`xarray.Dataset`

        """
        # TODO: Implement this

    def _write(
        self,
        uid: str,
        name: str,
        data: Dataset,
        equilibria: Dict[str, Equilibrium],
        prov: ProvDocument,
    ):
        """Perform the low-level writing of data to disk/database. It takes a
        single Dataset, with attributes reformatted as described in the
        documentation of :py:meth:`write`.

        """
        raise NotImplementedError(
            "{} does not implement a `write` method".format(self.__class__.__name__)
        )
