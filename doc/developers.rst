How to Develop
==============

When you first come to InDiCA as a developer, you may find the idea of
making even small changes to be quite daunting. This guide will try to
walk you through how to start developing and explain the steps for
adding different types of functionality. Before you continue, it is
strongly recommended that you read about the :ref:`Code Design`.


Obtaining a Copy of the Code
----------------------------

The `git version control system <https://git-scm.com/>`_ is used when
developing InDiCA. If you are not familiar with it, there are many
`tutorials available online
<https://guides.github.com/introduction/git-handbook/>`_. The code is
`hosted on GitHub <https://github.com/ukaea/Indica>`_. You can
download a copy with the command ::

  # If you have permission to make changes to the repository
  git clone git@github.com:ukaea/Indica.git

  # If you do not have permission and won't contribute changes upstream
  git clone https://github.com/ukaea/Indica.git

If you want to contribute your changes upstream but don't have
permission to do so, fork the repository and do your development
there. Then make a pull request (see below) from you forked version.

All development should be performed on a dedicated branch::

  git branch my_new_feature
  git checkout my_new_feature


Setting up the Development Environment
--------------------------------------

InDiCA development uses `Poetry <https://python-poetry.org/>`_ to manage
dependencies, control the testing environment, and handle
packaging. Follow the `instructions on installing poetry
<https://python-poetry.org/docs/#installation>`_. In the repository,
run::

  poetry install

This will install all the necessary dependencies in a virtual
environment. To run a command from this virtual environment, use::

  poetry run <command>

Next you need to set up `pre-commit <https://pre-commit.com/#install>`_
to enable the automatic running of various checks before you can commit your
code. Just run::

  poetry run pre-commit install

Now open your favourite text editor/IDE and start developing!


Adding Features
---------------

InDiCA provides a framework for analysing and performing calculations
with diagnostic data. The vast majority of new features will not
require any fundamental changes to this framework and only the adding
of new classes within it.

Reading from New Databases
~~~~~~~~~~~~~~~~~~~~~~~~~~

To support a new database (e.g., one at another fusion experiment),
you must subclass :py:class:`~indica.readers.DataReader`. The base
class provides all of the functionality for assembling data from
different diagnostics into DataArray objects. It also handles the
creation of provenance. Subclasses must implement logging into the
database (if necessary) and fetching the raw data.
An example of implementing such a reader can be seen in
:py:class:`~indica.readers.PPFReader`.  You should
place your new class in the ``readers`` directory and make it
available in the ``__init__.py`` module of said directory.

Reader Constructor
..................

You should first write the constructor for your class. This must make
a call to the parent's constructor, providing the following
information:

tstart
    Time into the pulse at which to start keeping data.

tend
    Time into the pulse at which to stop keeping data.

max_freq
    The maximum frequency of data collection. If data is collected
    above this frequency, some of it will be dropped.

selector
    A callback allowing the user to select which channels to drop when
    reading data.

sess
    The :py:class:`~indica.session.Session` object for this run of the
    code.

As such, you will want these to be arguments for your new
reader class's constructor. You will also probably want an argument
identifying which pulse to read data for and possibly one indicating
the server holding the database. Extra arguments such of these should
be passed to the parent constructor as keyword arguments, for use when
creating provenance.

Within the constructor you should also do anything required to set up
reading of data, such as instantiating the database client.

Authentication
..............

Most likely your database will require a username and/or password to
log in. If this is the case, you should implement the
:py:meth:`~indica.readers.DataReader.requires_authentication` property
(indicating when this is necessary) and the
:py:meth:`~indica.readers.DataReader.authenticate` method.

Diagnostic Fetching Methods
...........................

Each diagnostic fetcher in the base reader class (e.g.,
:py:meth:`~indica.readers.DataReader.get_radiation`,
:py:meth:`~indica.readers.DataReader.get_thomson_scattering`, etc.)
requires a corresponding private version of the method to be
implemented which returns the raw data as NumPy arrays. Each of these
private methods have docstrings describing the data they must
return. Not all reader classes need to implement all diagnostics.

For each diagnostic you implement, you must provide some information
on the sort of data it can return. First, you should define a
static/class-level attribute ``INSTRUMENT_METHODS``, which is a dictionary
mapping between INSTRUMENT names (in the JET parlance; they are the
"instrument" argument to the getter methods) and the specific
get-method used to read that data. In effect, this is defining what
type of diagnostic each supported instrument provides.

You may also need to provide an ``_IMPLENTATION_QUANTITIES`` static
attribute. This is similar to the ``_AVAILABLE_QUANTITIES`` attribute
in the base class. The latter describes default quantities which are
available for each diagnostic and what :ref:`datatype<Data Value Type
System>` they will have. ``_IMPLEMENTATION_QUANTITIES`` allows you to
override this for specific instruments. It maps from instrument names
to dictionaries. These dictionaries have keys that are the name of
available quantities and values that are the datatype of the quantity.

.. note::
   You may wish to cache the raw data you have fetched from the
   database, to speed up future reading. This is done by
   :py:class:`~indica.readers.PPFReader`. However, it is not mandatory.

Bad Channels
............

Sometimes a channel is known to provide bad data. This might be
because it corresponds to a line of sight which is facing the
divertor. You must implement the private method ``_get_bad_channels``
which will return a list of these channels given a particular
instrument and quantity.

Provenance
..........

Most of the work of generating provenance is handled by the base
class. However, you should provide a ``NAMESPACE`` attribute on the
child class, as either a class or and object attribute. This is tuple
containing a short name for the namespace and a URL. This URL will
likely be that of the server you are fetching the data from. (See
information on `PROV namespaces <https://www.w3.org/TR/2013/REC-prov-dm-20130430/#term-NamespaceDeclaration>`_.)

Additionally, you will see that when implementing the private getter
methods you are required to include ``<quantity_name>_records`` data
in the result. This is a list of strings, each of which should
uniquely identify the database records you have accessed. This does
not need to contain any information on the database URL, however, as
that will be included by the base class.

Supporting New Coordinate Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To implement a converter for a new coordinate system, you need to
subclass :py:class:`~indica.converters.CoordinateTransform`. You will
then need to provide the methods
:py:meth:`~indica.converters.CoordinateTransform.convert_to_Rz`,
:py:meth:`~indica.converters.CoordinateTransform.convert_from_Rz`, and
:py:meth:`~indica.converters.CoordinateTransform.__eq__`.  A good
example to start from when creating a new coordinate transform is
:py:class:`~indica.converters.TransectCoordinates`. You should place
your new class in the ``readers`` directory and make it available in
the ``__init__.py`` module of said directory.


Standard Functionality
......................

You will most likely also need to provide a constructor, although
there are no particular constraints or requirements on what this
does. It should just take whatever configurations are needed for your
coordinate system. You will also need to declare the attributes
``x1_name`` and ``x2_name`` which are the names which should be used
for the first and second spatial coordinates. These may be class
attributes, if they will always be the same for these types of
coordinates (e.g., in the
:py:class:`~indica.converters.TrivialTransform`. Alternatively, they
may be object attributes if each instance can represent a distinct
coordinate system (e.g.,
:py:class:`~indica.converters.TransectCoordinates`).

The ``convert_from_Rz`` and ``convert_to_Rz`` methods are fairly
self-explanatory. They should convert from R-z coordinates to your new
coordinate system and vice versa, respectively. The equality operator
should check whether two transforms describe identical coordinate
systems. It must start with a call to the ``_abstract_equals`` method,
which will check equality of attributes on the base class and the
coordinate names.

Shortcut Methods
................

Some coordinate systems have a natural means of converting between
each other (e.g., :py:class:`~indica.converters.LinesOfSightTransform`
and :py:class:`~indica.converters.ImpactParameterCoordinates`) which
will be more efficient than doing so via the R-z system. Often
calculation of R-z coordinates will require converting to this other
coordinate system first. In those cases you should implement those
calculations in separate methods. You can then override the
:py:meth:`~indica.converters.CoordinateTransform.get_converter` method
to return one of these "shortcut" methods for converting to the other
coordinate system, if such a shortcut is available. Otherwise, it
should just return ``None``.

There are some subtleties to this of which you should be wary. First,
often such a shortcut conversion will only be possible for a
*particular instance* of the other coordinate system (as is the case
for lines of sight and impact parameters: the shortcut only makes
sense of lines of sight coordinates are the same ones for which the
impact parameters were calculated). It should also be noted that the
``get_converter`` method has the argument ``reverse``, which indicates
that you are looking for the reverse conversion (convert from the
other coordinates system to this one, instead of from this one to the
other). If ``reverse == False`` and you could not find a suitable
converter on your object, you should always make a call to
``other.get_converter(self, reverse=True)`` to see if that object has
a suitable conversion method. This is necessary because often the
necessary information for both directions of the conversion is only
held by one of the coordinate systems and it must implement both of
the shortcut methods.

Other Notes
...........

In rare cases it may be necessary to implement a custom
:py:meth:`~indica.converters.CoordinateTransform.distance`
method. This is the case if, for some reason, the distance between
successive points in R-z space does not correspond to the actual
distance along the coordinate. This would happen if the coordinate has
some component in the toroidal direction, as is the case for the
:py:class:`~indica.converters.LinesOfSightTransform`.

Coordinate transforms to not record provenance.

Performing New Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

Most development will likely focus on performing new calculations with
the data. This will require you to create new subclasses of
:py:class:`~indica.operators.Operator`. You will need to implement a
constructor, :py:meth:`~indica.operators.Operator.return_types` and
the :py:meth:`~indica.operators.Operator.__call__`
methods. :py:class:`~indica.operators.CalcZeff` provides a simple
example of an operator which you can examine. New operator classes
should be placed in the ``operators`` directory and made available in
the ``__init__.py`` file in that directory.

Operator Constructor
....................

Your constructor must make a call to the constructor on the parent
class. This requires you to pass a :py:class:`~indica.session.Session`
object, which your subclass's constructor should also take as an
argument. Any other arguments to the subclass's constructor should be
passed as keyword arguments to the superclass's constructor so they
can be included in provenance.

Argument Types and Return Types
...............................

All operators must provide an ``ARGUMENT_TYPES`` attributes, which is
a list of :ref:`datatypes<Data Value Type System>`. This may be either
a class attribute or an instance attribute, as appropriate. Datatypes
in the list may contain ``None`` for the specific datatype and/or, in the
case of data arrays, the general datatype as well. This indicates that
the type is unconstrained. The final element of the list may be an
Ellipsis object (``...``), which indicates that the operator takes
variadic arguments. The type of the variadic arguments will be that of
the penultimate item in the list. If that datatype is unconstrained
(i.e., contains ``None``) then the type of all variadic arguments must
match that of the first variadic argument.

You will also need to implement the
:py:meth:`~indica.operators.Operator.return_types` method. This takes
datatypes as arguments. These correspond to the datatypes of some
hypothetical arguments for the operator. The method will then return a
tuple of the datatypes of the results of the operator. The number and
types of results will often depend on the number and types of
arguments to the operator, hence why a method is needed to determine
them.

The Calculation Itself
......................

The operator's calculation is performed in the
:py:meth:`~indica.operators.Operator.__call__` method. These methods
break strict static typing, as each operator will take a different
number of arguments, with different names. However, all arguments must
be positional and none should be optional. Variadic positional
arguments are allowed. In order to prevent mypy from complaining that
your ``__call__`` method does not match the call signature of the
original on the base class, you should add ``# type:
ignore[override]`` to the method declaration.

The first thing you should do in the method is call
:py:meth:`~indica.operators.Operator.validate_arguments`. This will
check that all arguments are of the expected type. It will also take
note of these arguments for the purpose of generating provenance.

Your operator should then proceed to the calculation. The details of
this will vary greatly from case to case. If your calculation is
expected to take a long time, it may be worth printing some messages
describing the progress. These can also be useful for
debugging.

Remember that many, many mathematical operations are
available in `SciPy <https://www.scipy.org/>`_ and do not need to be
implemented from scratch. When performing coordinate transformations,
make use of the
:py:meth:`~indica.data.InDiCAArrayAccessor.convert_coords` method (and
its equivalent for datasets) so results will be stored in the data
array/dataset and be available for later reuse. If you need to perform
an interpolation, remember that xarray offers this builtin
(:py:meth:`xarray.DataArray.interp`) and, should you need to perform
cubic interpolation over two dimensions, this is available through the
:py:meth:`~indica.data.InDiCAArrayAccessor.interp2d` method.

It is often useful to return intermediate results of the calculation
for reuse elsewhere. If there is some output of the calculation that
is neither a dataset nor a data array, then it can be assigned as
metadata to one of the other results.

Once the calculation is finished, be sure that all of your results
have the necessary metadata, such as datatype, coordinate transform,
and an equilibrium set. You will also need to assign provenance data
to each result. You should do this by calling
:py:meth:`~indica.operators.Operator.assign_provenance` for each
result.


Contributing Changes Upstream
-----------------------------

If you implement new features, you should consider submitting them for
inclusion in the official version of InDiCA. You can do this by
submitting a `pull
request <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_
on GitHub. In your pull request, please explain what you have
implemented, with reference to any of the repository's issues which it
may address.

In order for your pull request to be accepted, it must meet the
following standards:

- pass all pre-commit hooks (e.g., it must obey the
  `black <https://github.com/psf/black) formatting style>`_
- use Python `type-hints <https://www.python.org/dev/peps/pep-0484/>`_ wherever possible
- pass `mypy <https://mypy.readthedocs.io/en/stable/>`_
- provide unit tests (and ideally integration tests as well)
- not introduce any regressions in existing functionality
- provide docstrings for all functions and classes, using the `NumPy
  style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html#google-vs-numpy>`_
- depending on the sort of change you make, explain the new features
  in the `sphinx <https://www.sphinx-doc.org/en/master/>`_ documentation
  held in the ``doc/`` directory.

Most of these will be checked automatically by the continuous
integration system when you create your pull request. You should also
expect your code to undergo review and you may be requested to make
various stylistic changes or adopt a more idiomatic approach to using
InDiCA features.
