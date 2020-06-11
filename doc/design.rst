Code Design
===========
   
This section of the documentation describes the data structures and
organisation of the code. It is primarily intended for developers.


Data Containers
---------------

Diagnostics and the results of calculations are stored in the
:py:class:`xarray.DataArray`. This stores multidimensional data with
labelled dimensions, coordinates along each dimension, and associated
metadata. Standard mathematical operations are built into these
objects. Additional bespoke functionality is provided using meta-data
or `"custom accessors"
<http://xarray.pydata.org/en/stable/internals.html#extending-xarray>`_,
as described in the xarray documentation.



Accessing Equilibrium Data
--------------------------

Key to analysing any fusion reactor data is knowing the equilibrium
state of the plasma. This is done using equilibrium
calculations. Multiple models are available for this and it should be
easy to swap one for another in the calculation. It is also desirable
that they use the same interface, regardless of what fusion reactor
the equilibrium is for or what library is used to access the data.

The equilibrium data which is of interest is the total magnetic field
strength, the location of various flux surface (poloidal, toroidal,
and potentially others), the minor radius, the volume enclosed by a
given flux surface, and the minimum flux surface for a
line-of-sight. An abstract class was defined with abstract methods to
access this data. Rather than try to anticipate every type of flux
surface which might be needed, any method which takes or returns a
flux surface has an argument ``kind`` which accepts a string
specifying which one is desired (``"toroidal"``, by default). There is
also a method to convert between different flux surface types.

Unfortunately, equilibrium results are not always entirely accurate
and may need to be adjusted. A multiplier for the magnetic field
strength should be provided in the constructor for the equilibrium
object. Additionally, the location of the flux surfaces will often be
slightly offset along the major axis from the "real" ones. Therefore,
a ``calibrate`` method is provided. This estimates (to the nearest
half centimetre) the offset in R needed for the electron temperature
at the last closed flux surface to be about 100eV. It provides a plot
of the optimal R-shift at different times, with the average value also
draw. This average value is used to reposition the flux surfaces and a
second plot is produced with electron temperature against normalised
flux. The user can choose to accept this offset or to specify a custom
value. If the latter, these plots will be recreated with the new
R-shift and the user will again be asked whether or not to accept it.

The abstract equilibrium class can be represented by the following
UML.

.. uml::

   class Equilibrium {
   + R_ax: DataArray
   + R_sep: DataArray
   + z_ax: DataArray
   + z_sep: DataArray
   + tstart: float
   + tend: float
   + provenance: ProvEntity
   - _session: Session
   
   + calibrate(T_e: DataArray, selector: Callable)
   + {abstract} Btot(R: arraylike, z: arraylike, t: arraylike): (arraylike, arraylike)
   + {abstract} enclosed_volume(rho: array_like, t: array_like, kind: str): (arraylike, arraylike)
   + {abstract} minor_radius(rho: arraylike, theta: arraylike, t: arraylike, kind: str): (arraylike, arraylike)
   + {abstract} flux_coords(R: arraylike, z: arraylike, t: arraylike, kind: str): (arraylike, arraylike, arralylike)
   + {abstract} spatial_coords(rho: arraylike, theta: arraylike, t: arraylike, kind: str): (arraylike, arraylike, arralylike)
   + {abstract} convert_flux_coords(rho: arraylike, theta: arraylike, t: arraylike, kind: str): (arraylike, arraylike, arralylike)
   + R_hfs(rho: arraylike, t: arraylike, kind: str): (arraylike, arraylike)
   + R_lfs(rho: arraylike, t: arraylike, kind: str): (arraylike, arraylike)
   }



Coordinate Systems and Transforms
---------------------------------

Each diagnostic which is used for calculations is stored on a
different coordinate system and/or grid. One of the key challenges is
thus to make it easy to convert between these coordinate systems. This
is further complicated by the fact that many of the coordinate systems
are based on what (time-dependent) equilibrium state was calculated
for the plasma. Transforms between coordinate systems must therefore
be agnostic as to which equilibrium results are used.

When operations are performed on :py:class:`xarray.DataArray` objects,
they are :ref:`automatically aligned <math automatic alignment>`
("alignment" meaning that ticks on their respective axes have the same
locations). Any indices which do not match are discarded; the result
consists only of the union of the two sets of coordinates. When
operating on datasets where some or all dimensions have different
names, it automatically performs :ref:`compute.broadcasting`. However,
note that this would not be physically correct if the coordinates are
not linearly independent.

There is also built-in support for :ref:`interpolating onto a new
coordinate system <interp>`. This
can be either for different grid-spacing on the same axes or for
another set of axes entirely. The latter can be slightly cumbersome to
do and requires some additional information about how coordinates map,
so we will likely want to provide convenience methods for that
purpose.

In order to perform these sorts of conversions, I means is necessary
to map from one coordinate system to another. An arbitrary number of
potential coordinate systems could be used and being able to map
between each of them would require :math:`O(n^2)` different
functions. This can be reduced to :math:`O(n)` if instead we choose a
go-between coordinate system to which all the others can be
converted. A sensible choice for this would be :math:`R, z`, as these
axes are orthogonal, the coordinates remain constant over time, and
libraries to retrieve equilibrium data typically work in these
coordinates.

A :py:class:`convertors.CoordinateTransform` class is defined to handle
this process. This is an abstract class which will have a different
subclass for each type of coordinate system. It has two abstract
methods (both private), for converting coordinates to and from
R-z. These get wrapped by public (non-abstract) methods which provide
default argument values and cache the result for these
defaults. A non-abstract ``convert_to`` method takes
another coordinate system as an argument and will map coordinates
onto it. Finally, the ``distance`` method can provide the spatial
distance between grid-points along a given axis and first grid-point
on that axis.

In addition to doing conversions via R-z coordinates, subclasses of
:py:class:`convertors.CoordinateTransform` may define additional
methods to map directly between coordinate systems. This would be
useful if there is a more efficient way to do the conversion without
going through R-z, if that transformation is expected to be
particularly frequently used, or if that transformation would need to
be done as a step in converting to R-z coordinates.

The :py:class:`convertors.CoordinateTransform` class is agnostic to the
equilibrium data and can be instantiated without any knowledge of
it. However, many subclasses will require equilibrium information to
perform the needed calculations. This can be set using the
``set_equilibrium`` method at any time after instantiation. Calling
this method multiple times with the same equilibrium object will have
no affect. Calling with a different equilibrium object will cause an
error unless specifying the argument ``force=True``.

.. uml::

   class CoordinateTransform {
   + set_equilibrium(equilibrium: Equilibrium, force: bool)
   + convert_to(other: CoordinateTransform, x1: arraylike, x2: arraylike, t: arraylike): (arraylike, arraylike, arraylike)
   + convert_to_Rz(x1: arraylike, x2: arraylike, t: arraylike): (arraylike, arraylike, arraylike)
   + convert_from_Rz(x1: arraylike, x2: arraylike, t: arraylike): (arraylike, arraylike, arraylike)
   + distance(direction: int, x1: arraylike, x2: arraylike, t: arraylike): (arraylike, arraylike)
   - _convert_to_Rz(x1: arraylike, x2: arraylike, t: arraylike): (arraylike, arraylike, arraylike)
   - _convert_from_Rz(x1: arraylike, x2: arraylike, t: arraylike): (arraylike, arraylike, arraylike)
   }

Each DataArray will have a ``transform`` attribute which is one of
these objects. To save on memory and computation, different data from the same
instrument/diagnostic will share a single transform object. This
should not normally be of any concern for the user, unless they area
attempting to use multiple sets of equilibrium data at once.

When two :py:class:`xarray.DataArray` objects use the same coordinate
system with only different grid spacing, the built in
:py:meth:`~xarray.DataArray.interp_like` method already provides a
convenient interface to interpolate one onto the same locations as the
other. For converting to different coordinate systems, I would suggest
extending :py:class:`xarray.DataArray` with a `"custom accessor"
<http://xarray.pydata.org/en/stable/internals.html#extending-xarray>`_
that has a method ``remap_like()``, which would provide the same
behaviour (and simply delegate to ``interp_like()`` if the same
coordinate system is used). More details to follow.

Custom accessors appear like attributes on
:py:class:`xarray.DataArray`, with their own set of methods. This
allows xarray extensions to be "namespaced" (i.e., common
functionality gets grouped into the same accessor). The
use is as follows::

  # array1 and array2 are on different coordinate systems.
  
  # Broadcasting creates a 4D array; probably not what you want
  array3 = array1 + array2

  # Same coordinate system as array1
  array4 = array1 + array2.impurities.remap_like(array1)

  # Same coordinate system as array2
  array5 = array1.impurities.remap_like(array2) + array2

Anyone who imports this library will be able to use the accessor with
xarrays in their own code.

Metadata
~~~~~~~~

The following metadata should be attached to
:py:class:`xarray.DataArray`:

datatype : **(str, str)**
    Information on the type of data stored in this
    :py:class:`xarray.DataArray` object. See :ref:`Data Value Type System`.

provenance : **:py:class:`prov.model.ProvEntity`**
    Information on the process which generated this data. See
    :ref:`Provenance Tracking`.

transform : **:py:class:`convertors.CoordinateTransform`**
    An object describing the coordinate system of this data, with
    methods to map to other coordinate systems.

error (optional) : **ndarray**
    Uncertainty in the value.


Data IO
-------

Reading data should be achieved by defining a standard interface,
:py:class:`readers.DataReader`. A different subclass would then be defined for
each data source/format. These would return
:py:class:`xarray.DataArray` objects with all the necessary metadata.

.. uml::

   abstract class DataReader {
   + {static} available_data: dict
   __
   + get(key: string, revision): DataArray
   + authenticate(name: str, password: str): bool
   + {abstract} close()
   - {abstract} _get_data(key: str, revision): DataArray
   .. «property» ..
   + {abstract} requires_authentication(): bool
   }

   class PPFReader {
   + {static} available_data: dict
   + uid: str
   + seq: int
   - _client: SALClient
   __
   + __init__(uid: str, seq: int)
   + authenticate(name: str, password: str): bool
   + close()
   - _get_data(key: string, revision: int): DataArray
   .. «property» ..
   + {abstract} requires_authentication(): bool
   }

   DataReader <|- PPFReader

Here we see that reader classes contain public methods for getting
data using a key (and optional revision, to specify which version of
data in systems using version control). It also provides methods for
authentication and closing a database connection. Each reader should
feature a dictionary called ``available_data`` where keys are valid
arguments for the :py:meth:`reader.DataReader.get` method and corresponding
values are the type of data which gets returned (see next
section). Python's abstract base class module (:py:mod:`abc`) can be
used when defining ``DataReader``

The :py:meth:`reader.DataReader.get` method is implemented in the parent
class and provides basic functionality for checking whether a key is
valid and that the returned :py:class:`xarray.DataArray` object
contains the expected metadata. The actual process of getting this
data is delegated to the abstract private method
:py:meth:`reader.DataReader._get_data`, which is implementation
dependent. Implementations are free to define additional private
methods if necessary. The form of the constructor for each reader
class is not defined, as this is likely to vary widely.

A complicating factor in all of this is that the ``map_to_master`` and
``map_from_master`` metadata functions can only be created once
equilibrium data has been loaded. Furthermore, ideally one would be
able to change which equilibrium was used on the fly, without having
to reload all other data. The solution to this is to define an
additional metadata function, called ``generate_mappers`` which takes
an :py:class:`xarray.Dataset` object containing equilibrium data
(exact format TBC) as an argument. It would return a tuple made up of
the ``map_to_master`` and ``map_from_master`` functions appropriate
to that equilibrium.

Exactly where and how ``generate_mappers`` should be used can be a
matter of further discussion. One approach would be to do it in one of
the :ref:`Operations on Data` described below. This has the advantage
of keeping a clear separation of concerns between the different
classes in the program, but could be somewhat verbose to use. Another
option would be to allow an equilibrium dataset to be assigned to a
reader object (perhaps even have the reader automatically load the
entire dataset). It would then use this dataset to call the
``generate_mappers`` function when reading in all future data. This
would be more convenient, but somewhat complicates the logic of the
program by mixing functionality.

Should all data need to be remapped to a new set of equilibrium data,
it is proposed that the :ref:`Provenance Tracking` system be used to
automatically do this. This would be part of the broader "replay"
functionality which would be possible with provenance data.

A similar approach of defining an abstract base class could be taken
for writing data out to different formats. Presumably we would want to
include the formats used by different research centres. However, it
might also be useful to use some general formats for output, such as
NetCDF or HDF5. This would probably be an easier operation, as it
would not require so much knowledge of the peculiarities of how data
is stored by different research groups.


Data Value Type System
----------------------

When performing physics operations, arguments have specific physical
meanings associated with them. The most obvious way this manifests
itself is in terms of what units are associated with a
number. However, you may have multiple distinct quantities with the
same units and an operation may require a specific one of those. It is
desirable to be able to detect mistake arising from using the wrong
quantity as quickly as possible. For this reason, operations on data
define what they expects that data to be and to check this.

Beyond catching errors when using this software as a library or
interactively at the command line, this technique will be valuable
when building a GUI interface. It will allow the GUI to limit the
choice of input for each operation to those variables which are
valid. This will simplify use (as your choices will be limited to
those which are appropriate) and make it safer.

This system need not be very complicated. A type consists of one
mandatory label and a second, optional one. The first label
indicates the general sort of quantity (e.g., number density,
temperature, luminosity, etc.) and the second specifies what that
quantity applies to (type of ion, electrons, soft X-rays, etc.). This
is expressed as a 2-tuple, where the first element is a string
and the second is either a string or ``None``. See examples below::

    # Describes a generic number density of some particle
    ("number_density", None)
    # Describes number density of electrons
    ("number_density", "electrons")
    # Describes number density of primary impurity
    ("number_density", "impurity0")
    
    ("temperature", None) # Temperature
    ("temperature", "electrons")  # Electron temperature

Each operation on data contains information on the types of
arguments it expects to receive and return and has a method to
confirm that these expectations are met. An operation should always
specify the first element in the type tuple and may choose to specify
the second if appropriate. Each :py:class:`xarray.DataArray`
contains one of these type-tuples in its metadata, associated to the
key ``"type"`` and this always specifies both elements of the
tuple.

In principal, this is all the infrastructure that would be needed for
the type system. However, it may be useful to keep a global registry
of the types available. This would help to enforce consistent
labelling of types and could add the ability to check for type. It
might also be used to store information on what each type corresponds
to and in what units it should be provided. This information would be
useful documentation for users and could be integrated in a GUI
interface. This could be accomplished using dictionaries::

    general_types = {"number_density": ("Number density of a particle", "m^-3"),
                     "temperature": ("Temperature of a species", "keV")}
    specific_types = {"electrons": "Electron gas in plasma",
                      "impurity0": "Primary impurity, varying in space and time"}

Note that impurities are not specified by their actual
composition. This is because calculations do not depend on a
particular element but rather on the assumptions which have been made
about that impurity. These are indicated by names such as
``"impurity0"``, ``"impurity"``, etc. More details about these
different assumptions will be explained elsewhere.


Provenance Tracking
-------------------

In order to make research reproducible, it is valuable to know exactly
how a data set is generated. For this reason, it is proposed that the
software contains a mechanism for tracking data "provenance". Every
time data is created, either by being read in from the disk, by some
assumption specified by the user, or by a calculation on other data, a
record should also be created describing how this was done.

There already exist standards and library for recording this sort of
information: W3C defines the `PROV standard
<https://www.w3.org/TR/2013/NOTE-prov-overview-20130430/>`_ and the
`PyProv <https://prov.readthedocs.io/en/latest/index.html>`_ library
exists to use it from within Python. In this model, there are the
following types of records:

Entity : :py:class:`prov.model.ProvEntity`
    Something you want to describe the provenance of, such as book,
    piece of artwork, scientific paper, web page, or book.
Activity : :py:class:`prov.model.ProvActivity`
    Something occurring over a period of time which acts on or with
    entities.
Agent : :py:class:`prov.model.ProvAgent`
    Something bearing responsibility for an activity occurring or an
    entity existing.

There are various sorts of relationships between these objects, with
the main ones summarised in the diagram below.

.. image:: _static/provRelationships.png

This software provides a class :py:class:`session.Session` which holds
the :py:class:`provenance document <prov.model.ProvDocument>` as well
as contains information about the user and version of the software. A
global session can be established using
:py:meth:`session.Session.begin` or a context manager. Doing so
requires specifying information about the user, such as an email or
ORCiD ID. The library will then use this global session to record
information or, alternatively, you can provide your own instance when
constructing objects. The latter option allows greater flexibility
and, e.g., running two sessions in parallel.

What follows is a list of the sorts of PROV objects which will be
generated. Each of them should come with an unique identifier. Where
the data is read from some sort of database this could be the key for
the object. Otherwise it should be a hash generated from the metadata
of the object.

Calculations
~~~~~~~~~~~~
A calculation will be represented by an **Activity**. It will be
linked with the data entities it used to do the calculation, the user
or other agent to invoke it, and the Operator object which actually
performed it.

:py:class:`xarray.DataArray` objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each data object will be represented by an **Entity**. This entity will
contain links with the user and piece of software (e.g., reader or
operator) to create it, the reading or calculation activity it was
produced by, and any entities which went into its creation.

:py:class:`reader.DataReader` objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These objects are represented as both an **Entity** and an
**Agent**. The former is used to describe how it was instantiated
(e.g., the user that created it, what arguments were used) while the
latter can be used to indicate when it creates DataArray objects by
reading them in.

Dependency
~~~~~~~~~~
Third-party libraries which are depended on should be represented as
**Entitites** in the provenance data. Information should be provided
on which version was used.

External data
~~~~~~~~~~~~~
External data (e.g., contained in source files or remote databases)
should have a simple representation as an **Entity**. Sufficient
information should be provided to uniquely identify the record.

Operator objects
~~~~~~~~~~~~~~~~
Similar to reader objects, these are represented as both an **Entity**
and an **Agent**. Again, the former provides information on who
created the operator and what arguments were used. The latter
indicates the object's role in performing calculations.

Package
~~~~~~~
The overall library/impurities package is itself represented by an
**Entity**. This should contain information on the version or git
commit. It could also provide information on the authors who wrote it.

Reading data
~~~~~~~~~~~~
Reading data is an **Activity**. It is associated with a reader agent
and a user of the software. It uses external data entities. 

:py:class:`session.Session` object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
An **Activity** representing the current running instance of this
software. It uses the package and dependencies and is associated with
the user to launch it. It contains metadata on the computer being
used, the working directory, etc.

Users
~~~~~
The person using the software is represented as an **Agent**. Data
objects will be attributed to them. They are associated with the
session. Sometimes they will delegate authority to classes or
functions which are themselves agents. Sufficient metadata should be
provided to allow them to be contacted. Ideally they would have some
sort of unique identifier such as an ORCiD ID, but email is also
acceptable.


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

   class Operator {
   - _start_time: datetime
   - _input_provenance: list
   - _session: Session
   + agent: ProvAgent
   + entity: ProvEntity

   + __init__(self, sess, **kwargs)
   + {abstract} __call__(self, *args): DataArray
   + create_provenance()
   + validate_arguments(*args)
   + {static} recreate(provenance: ProvEntity): Operator
   }
   
   class ImplementedOperator {
   + {static} INPUT_TYPES: list
   + {static} RESULT_TYPES: list
   
   + __init__(self, ...)
   + __call__(self, ...): DataArray
   }
   
   Operator <|-- ImplementedOperator

While performing the calculation they should not make reference to any
global data except for well-established physical constants, for
reasons of reproducibility and data provenance. If it would be
considered too cumbersome to pass all of the required data when
calling the operation, additional parameters could potentially be
provided at instantiation-time; this would be useful if the
operation were expected to be applied multiple times to different data
but using some of the same parameters.

We can discuss whether it would be best to have the call return a new
object or to operate on the first argument in-place. I find the former
tidier, more readable, generally less prone to bugs, etc. However, the
second can save memory. Both approaches allow us to avoid operating on
global variables.
