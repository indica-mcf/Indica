Code Design
===========
   
This section of the documentation describes the data structures and
organisation of the code. It is primarily intended for developers.


Data Containers
---------------

I would recommend basing data storage on the `xarrays
<http://xarray.pydata.org/en/stable/>`_ package and, in particular, the
:py:class:`xarray.DataArray` class which it provides. This can be used
to store multidimensional data with labelled dimensions, coordinates
along each dimension, and associated metadata. Standard mathematical
operations are built into these objects. We should be able to use this
class to represent all of the data, without needing to create any
specific subclasses. Additional functionality which we need can be
achieved through using meta-data or by providing a `"custom accessor" 
<http://xarray.pydata.org/en/stable/internals.html#extending-xarray>`_,
as described in the xarray documentation. More details on the required
extensions is provided below.


Coordinate Systems
------------------

When operations are performed on :py:class:`xarray.DataArray` objects,
they are :ref:`automatically aligned <math automatic alignment>`
("alignment" meaning that ticks on their respective axes have the same
locations). Any indices which do not match are discarded; the result
consists only of the union of the two sets of coordinates. When
operating on datasets where some or all dimensions have different
names, it automatically performs :ref:`compute.broadcasting`. However,
note that this would not be physically correct if the coordinates are
not linearly independent.

There is also built-in support of :ref:`interpolating onto a new
coordinate system <interp>`. This
can be for either different grid-spacing on the same axes or for
another set of axes entirely. The latter can be slightly cumbersome to
do and requires some additional information about how coordinates map,
so we will likely want to provide convenience methods for that
purpose.

In order to perform these sorts of conversions, it will be necessary to
provide functions which map from one coordinate system to another. An
arbitrary number of potential coordinate systems could be used and
being able to map between each of them would require $O(n^2)$
different functions. This can be reduced to $O(n)$ if instead we
choose a "master" coordinate system to which all the others can be
converted. A sensible choice for this would be $R, z$, as these axes
are orthogonal and the coordinates remain constant over time (unlike
flux surfaces).

To achieve this, each :py:class:`xarray.DataArray` would contain a piece of metadata
called ``mapToMaster`` and another called ``mapFromMaster``. Both of these
would be functions, each taking 2 arguments (plus an optional third)
and returning a 2-tuple (or 3-tuple, if the optional argument is
provided). The first would accept a coordinate on the system used by
the :py:class:`xarray.DataArray` and return the corresponding location in the master
coordinate system. The second function would perform the inverse
operation. The optional third argument corresponds to time and would
be needed if a coordinate system is not fixed in time.

When two :py:class:`xarray.DataArray` objects use the same coordinate
system with only different grid spacing, the built in
:py:meth:`~xarray.DataArray.interp_like` method already provides a
convenient interface to interpolate one onto the same locations as the
other. For converting to different coordinate systems, I would suggest
extending :py:class:`xarray.DataArray` with a `"custom accessor"
<http://xarray.pydata.org/en/stable/internals.html#extending-xarray>`_
that has a method ``remap_like()``, which would provide the same
behaviour (and simply delegate to ``interp_like()`` if the same
coordinate system is used). The following UML class diagram
illustrates this structure.

.. uml::

   class DataArray {
   + name: str
   + values: ndarray
   + dims: tuple
   + coords: dict
   + attrs: dict
   + wsx: WSXAccessor
   ...
   __
   ...
   + interp(coords): DataArray
   + interp_like(other: DataArray): DataArray
   ...
   }

   class WSXAccessor {
   - _obj: DataArray
   
   + remap_like(other: DataArray): DataArray
   }
   
   DataArray -* WSXAccessor
   DataArray o- WSXAccessor

The next diagram gives an example of some of these attributes in a
:py:class:`xarray.DataArray` object.

.. uml::

   object example_data {
   + name = "W density"
   + values
   + dims = ["rho", "R"]
   + coords = {"rho": [-1.0, ..., 1.0], "R": [2.0, ..., 4.0]}
   + attrs = {"mapToMaster": func(rho, R, t=None) -> (R, z, t),\n\t "mapFromMaster": func(R, z, t=None) -> (rho, R, t), ...}
   + wsx
   }

Custom accessors appear like attributes on
py:class:`xarray.DataArray`, with their own set of methods. This
allows xarray extensions to be "namespaced" (i.e., common
functionality gets grouped into the same accessor). The
use is as follows::

  # array1 and array2 are on different coordinate systems.
  
  # Broadcasting creates a 4D array; probably not what you want
  array3 = array1 + array2

  # Same coordinate system as array1
  array4 = array1 + array2.WSXAccessor.remap_like(array1)

  # Same coordinate system as array2
  array5 = array1.WSXAccessor.remap_like(array2) + array2

Anyone who imports this library will be able to use the accessor with
xarrays in their own code.


Data IO
-------

Reading data should be achieved by defining a standard interface. A
different subclass would then be defined for each data
source/format. These would return :py:class:`xarray.DataArray` objects
with all the necessary metadata. More information should be gathered
on how each source is accessed to determine how best to define a
common interface for them.

A similar approach could be taken for writing data out to different
formats. Presumably we would want to include the formats used by
different research centres. However, it might also be useful to use
some general formats for output, such as NetCDF or HDF5.



Data Value Type System
----------------------

When performing physics operations, arguments have specific physical
meanings associated with them. The most obvious way this manifests
itself is in terms of what units are associated with a
number. However, you may have multiple distinct quantities with the
same units and an operation may require a specific one of those. It is
desirable to be able to detect mistake arising from using the wrong
quantity as quickly as possible. For this reason it would be useful to
allow different operations on data to define what it expects that data
to be and to check that this condition is met.

Beyond catching errors when using this software as a library or
interactively at the command line, this technique could be valuable
when building a GUI interface. It would allow the GUI to limit the
choice of input for each operation to those variables which are
valid. This would simplify use (as your choices would be limited to
those which are appropriate) and make it safer.

This system need not be very complicated. A type would consist of one
mandatory label and a second, optional one. The first label would
indicate the general sort of quantity (e.g., number density,
temperature, luminosity, etc.) and the second would specify what that
quantity applies to (type of ion, electrons, soft X-rays, etc.). This
could be expressed as a 2-tuple, where the first element is a string
and the second is either a string or ``None``. See examples below.

::
   
    # Describes a generic number density of some particle
    ("n", None)
    # Describes number density of electrons
    ("n", "e")
    # Describes number density of Tungsten
    ("n", "W")
    
    ("T", None) # Temperature
    ("T", "e")  # Electron temperature

It can be a matter of discussion whether we should use short symbolic
labels for types or whether they should be slightly longer and more
descriptive, e.g.::

    # Describes a generic number density of some particle
    ("number_density", None)
    # Describes number density of electrons
    ("number_density", "electrons")
    # Describes number density of Tungsten
    ("number_density", "tungsten")
    
    ("temperature", None) # Temperature
    ("temperature", "electrons")  # Electron temperature

Each operation on data would contain information on the types of
arguments it expects to receive and return and would have methods to
confirm that these expectations are met. An operation should always
specify the first element in the type tuple and may choose to specify
the second if appropriate. Each :py:class:`xarray.DataArray` would
contain one of these type-tuples in its metadata, associated to the
key ``"type"`` and this should always specify both elements of the
tuple.

In principal, this is all the infrastructure that would be needed for
the type system. However, it may be useful to keep a global registry
of the types available. This would help to enforce consistent
labelling of types and could add the ability to check for type. It
might also be used to store information on what each type corresponds
to and in what units it should be provided. This information would be
useful documentation for users and could be integrated in a GUI
interface. This could be accomplished using dictionaries::

    general_types = {"n": ("Number density of a particle", "m^-3"),
                     "T": ("Temperature of a species", "keV")}
    specific_types = {"e": "electrons", "W": "Tungsten"}


Provenance Tracking
-------------------

In order to make research reproducible, it is valuable to know exactly
how a data set is generated. For this reason, it is proposed that the
software contains a mechanism for tracking data "provenance". Every
time data is created, either by being read in from the disk, by some
assumption specified by the user, or by a calculation on other data, a
record should also be created describing how this was done. This
record could look something like the ``ProvenanceTree`` class described
in UML below.

.. uml::

   class ProvenanceTree {
   + name: str
   + created: date
   + creator: str 
   + commit: str
   + metadata: dict
   + inputs: list
   }

The meaning of each attribute is

name
   description of what this data represents

created
   the date this data was read from disk or calculated

creator
   the name of the class which produced this data

commit
   the git hash for the version of the code which was used

metadata
   information provided by the creator

inputs
   the ``ProvenanceNode`` objects used to calculate this data

Sufficient information should be provided in the metadata that the
data could be exactly recreated using the same inputs. When writing
data out to disk, the provenance information should be embedded in the
output file. In future it would be possible to implement a function
which uses this data to "replay" a calculation. Breakpoints could be
inserted into this for debugging purposes. Alternatively, metadata or
inputs could be altered prior to replaying the calculation, to see how
doing something differently would affect the results.

The UML object diagram below gives a sense of what a full provenance
tree could look like.

.. uml::
   
   object sxr_data {
   + name = "SXR Camera V"
   + created = "2020-01-01 12:00"
   + creator = "PPFReader"
   + commit = "8a5cf74"
   + metadata = {'pulse': 94672, user_id': 'JETPPF',\n\t 'source': 'EFIT', 'start': 47., 'end': 53.5,\n\t 'smooth': 0.02}
   + inputs = [major_rad_offset]
   }
   
   object major_rad_offset {
   + name = "EFIT Major Radius Offset"
   + created = "2020-01-01 11:59"
   + creator = "EFITOffset"
   + commit = "8a5cf74"
   + meta = {'target_temp': 100.}
   + inputs = [equilibrium, electron_temp]
   }
   
   object electron_temp {
   + name = "HRTS Electron Temperatures"
   + created = "2020-01-01 11:58"
   + creator = "PPFReader"
   + commit = "8a5cf74"
   + metadata = {'pulse': 94672, user_id': 'JETPPF',\n\t 'source': 'EFIT', 'start': 47., 'end': 53.5,\n\t 'smooth': 0.02}
   + inputs = []
   }
   
   object equilibrium {
   + name = "EFIT Equilibrium Data"
   + created = "2020-01-01 11:57"
   + creator = "PPFReader"
   + commit = "8a5cf74"
   + metadata = {'pulse': 94672, user_id': 'JETPPF',\n\t 'source': 'EFIT', 'start': 47., 'end': 53.5,\n\t 'smooth': 0.02}
   + inputs = []
   }
   
   sxr_data o-- major_rad_offset
   major_rad_offset o-- electron_temp
   major_rad_offset o-- equilibrium


Operations on Data
------------------

In the previous two sections I referred to "operations" on data. These
should be seen as something distinct from standard mathematical
operators, etc. Rather, they should be thought of as representing some
discreet, physically meaningful calculation which one wishes to
perform on some data. They take physical quantities as arguments and
return one or more derived physical quantities as a result. It is
proposed that these be represented by callable objects of class
``Operation``. A base class would be provided, containing some utility
methods, which all operators would inherit from. The main purpose of
these utility methods would be to check that types of arguments are
correct and to assemble information on data provenance. The class
might look something like the UML below.

.. uml::

   class Operation {
   - _provenance_input: list
   
   + validate_arguments(*args)
   + set_result_metadata(result: DataArray): DataArray
   - _create_result_provenance()
   }
   
   class ImplementedOperation {
   - {static} input_types: list
   - {static} result_types: list
   - _provenance_metadata: dict
   
   + __init__(...)
   + __call__(...)
   + {static} recreate(provenance: ProvenanceTree): ImplementedOperation
   }
   
   Operation <|-- ImplementedOperation

While performing the calculation they should not make reference to any
global data except for well-established physical constants, for
reasons of reproducibility and data provenance. If it would be
considered too cumbersome to pass all of the required data when
calling the operation, additional parameters could potentially be
provided at instantiation-time; this would be useful if the
operation were expected to be applied multiple times to different data
but using some of the same parameters.
