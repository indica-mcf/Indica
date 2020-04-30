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

In order to perform these sorts of conversions, it will be necessary
to provide functions which map from one coordinate system to
another. An arbitrary number of potential coordinate systems could be
used and being able to map between each of them would require
:math:`O(n^2)` different functions. This can be reduced to
:math:`O(n)` if instead we choose a "master" coordinate system to
which all the others can be converted. A sensible choice for this
would be :math:`R, z`, as these axes are orthogonal and the
coordinates remain constant over time (unlike flux surfaces).

To achieve this, each :py:class:`xarray.DataArray` would contain a
piece of metadata called ``map_to_master`` and another called
``map_from_master``. Both of these would be functions, each taking 3
arguments and returning a 3-tuple. The first would accept a coordinate
on the system used by the :py:class:`xarray.DataArray` and return the
corresponding location in the master coordinate system. The second
function would perform the inverse operation. The optional third
argument corresponds to time and would be needed if a coordinate
system is not fixed in time.

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
   + attrs = {"map_to_master": func(rho, R, t=None) -> (R, z, t),\n\t "map_from_master": func(R, z, t=None) -> (rho, R, t), ...}
   + wsx
   }

Custom accessors appear like attributes on
:py:class:`xarray.DataArray`, with their own set of methods. This
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

Metadata
~~~~~~~~

The following metadata should be attached to
:py:class:`xarray.DataArrays`:

equilibrium : **str**
    A string identifying the set of equilibrium data used for this
    object's coordinate system.

generate_mappers : **function(equilibrium) -> (map_to_master, map_from_master)**
    A higher ordered function which can be used as a factory
    for the two mapping functions below.

map_to_master : **function(x1, x2, t) -> (rho, theta, t)**
    A function mapping from the coordinate system of this data to the
    "master" coordinate system. Will be ``None`` immediately after
    read-in.

map_from_master : **func(rho, theta, t) -> (x1, x2, t)**
    A function mapping from the "master" coordinate system to the
    coordinate system of this data. Will be ``None`` immediately after
    read-in.

datatype : **(str, str)**
    Information on the type of data stored in this
    :py:class:`xarray.DataArray` object. See :ref:`Data Value Type System`.

provenance : **:py:class:`prov.model.ProvEntity`**
    Information on the process which generated this data. See
    :ref:`Provenance Tracking`.

error (optional) : **ndarray**
    Uncertainty in the value.


Data IO
-------

Reading data should be achieved by defining a standard interface,
:py:class:`reader.DataReader`. A different subclass would then be defined for
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
record should also be created describing how this was done.

There already exist standards and library for recording this sort of
information, so we should seek to use them. W3C defines the `PROV
standard <https://www.w3.org/TR/2013/NOTE-prov-overview-20130430/>`_
for representing this sort of data and the `PyProv
<https://prov.readthedocs.io/en/latest/index.html>`_ library exists to
use it from within Python. In this model, there are the following
types of records:

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

This software a class :py:class:`session.Session` which holds the
:py:class:`provenance document <prov.model.ProvDocument>` as well as
contains information about the user and
version of the software. A global session can be established using
:py:meth:`session.Session.begin` or a context manager. Doing so requires
specifying information about the user, such as an ORCiD ID (other
options TBC). The library will then use this global session to record
information or, alternatively, you can provide your own instance when
constructing objects. This allows greater flexibility and, e.g.,
running two sessions in parallel.

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
sort of unique identifier such as an ORCiD ID.


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

We can discuss whether it would be best to have the call return a new
object or to operate on the first argument in-place. I find the former
tidier, more readable, generally less prone to bugs, etc. However, the
second can save memory. Both approaches allow us to avoid operating on
global variables.
