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

InDiCA development uses [Poetry](https://python-poetry.org/) to manage
dependencies, control the testing environment, and handle
packaging. Follow the `instructions on installing poetry
<https://python-poetry.org/docs/#installation>`_. In the repository,
run::

  poetry install

This will install all the necessary dependencies in a virtual
environment. To run a command from this virtual environment, use::

  poetry run <command>

Next you need to [install pre-commit](https://pre-commit.com/#install)
to enable the running of various checks before you can commit your
code. This should be done outside


Contributing Changes Upstream
-----------------------------


Adding Features
---------------

InDiCA provides a framework for working with

Reading from New Databases
~~~~~~~~~~~~~~~~~~~~~~~~~~

Supporting New Coordinate Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Performing New Operations
~~~~~~~~~~~~~~~~~~~~~~~~~
