# Script for creating the model tree for ST40 HDA tree

from MDSplus import *
from MDSplus.mdsExceptions import TreeALREADY_THERE, TreeFOPENW, TreeNNF
import numpy as np
from MDSplus import Tree, Float32, Int32, String
import hda.mdsHelpers as mh
import hda.mds_tree_structures as trees
import getpass
import xarray as xr
from xarray import DataArray

from indica.readers import ST40Reader

from typing import Tuple
from typing import List


user = getpass.getuser()


def create(
    pulse: int,
    code_name: str,
    run_name="RUN01",
    descr="",
    tree_name="ST40",
    subtree=True,
    close=True,
    force=False,
    verbose=False,
):
    """
    Create Tree structure for storing HDA data

    Parameters
    ----------
    pulse
        Pulse number
    code_name
        Name of code used for analysis
        (e.g. "HDA")
    run_name
        Name of run under which data should be saved
    descr
        Description of analysis whose data is going to be saved
    tree_name
        Main tree name under which the data should be saved
        (e.g. "HDA" or "ST40", the latter possible only if subtree = True)
    subtree
        Create code_name tree as a subtree of tree_name
    close
        Close Tree after creation

    Returns
    -------
    t
        Tree object
    run_path
        Full path of tree for writing data
    """

    run_name = run_name.upper().strip()
    code_name = code_name.upper().strip()

    # Create code Tree if it doesn't exist
    code_path = rf"\{code_name}::TOP"
    write_path = f"{code_path}.{run_name}"

    try:
        t = Tree(code_name, pulse, "EDIT")
    except TreeFOPENW:
        print(f"\n Creating new tree {code_name}")
        t = Tree(code_name, pulse, "NEW")

    # Create/Overwrite RUN structure
    try:
        t.getNode(rf"{write_path}")
        if force:
            answer = "yes"
        else:
            run_name = next_run(t, code_path, run_name)
            print(f"\n {pulse}:{write_path} tree already exists")
            answer = input(
                f"\n # Write to next available run {run_name} * yes/(no) * ?  "
            )
            if answer.lower() == "no":
                return None

        if answer.lower() == "yes":
            delete(t, write_path, force=force)
    except TreeNNF:
        _user = user

    # Create RUN structure
    write_path = f"{code_path}.{run_name}"
    mh.createNode(t, f"{write_path}", "STRUCTURE", descr)
    print(f"\n {pulse}: {write_path} structure created")
    t.write()
    t.close()

    # If code_name should be a subtree, create main Tree if it doesn't exist
    if subtree:
        tree_path = rf"\{tree_name}::TOP"
        try:
            t = Tree(tree_name, pulse, "EDIT")
        except TreeFOPENW:
            t = Tree(tree_name, pulse, "NEW")

        # Make code_name a Subtree
        full_path = f"{tree_path}.{code_name}"
        try:
            t.getNode(full_path)
        except TreeNNF:
            t.addNode(full_path, "SUBTREE")
            t.write()
            t.close()

    # Define full path of tree structure and get node-names
    nodes = trees.hda()

    # Create nodes
    t = Tree(code_name, pulse, "EDIT")
    create_nodes(t, write_path, nodes, verbose=verbose)

    # Write and close Tree
    t.write()
    if close:
        t.close()

    return t, write_path


def delete(t, full_path, force=False):
    # Second warning to confirm delete
    print("#####################################################")
    print("#  *** WARNING ***                                  #")
    print("   You are about to delete data   ")
    print(f"{full_path}")
    print("#####################################################")

    if not force:
        answer = input("\n Confirm delete ? * yes/(no) *  ")
        if answer.lower() != "yes":
            return

    # Delete
    t.deleteNode(full_path)
    t.write()
    print("\n Data deleted")


def next_run(t, code_path, run_name):
    """
    Find next available run
    """
    write_path = f"{code_path}.{run_name}"
    n = t.getNode(rf"{write_path}")
    while n is not None:
        next = str(int(run_name[3:]) + 1)
        if len(next) == 1:
            next = "0" + next

        run_name = "RUN" + next
        write_path = f"{code_path}.{run_name}"
        try:
            n = t.getNode(rf"{write_path}")
        except TreeNNF:
            n = None

    return run_name


def create_nodes(t, run_path, nodes, verbose=False):

    t.addNode(f"{run_path}.METADATA", "STRUCTURE")
    t.addNode(f"{run_path}.GLOBAL", "STRUCTURE")
    t.addNode(f"{run_path}.PROFILES", "STRUCTURE")
    t.addNode(f"{run_path}.PROFILES.PSI_NORM", "STRUCTURE")

    for sub_path, quantities in nodes.items():
        for node_name, node_info in quantities.items():
            node_path = f"{run_path}{sub_path}:{node_name}"
            if verbose:
                print(node_path, node_info[0], node_info[1])
            n = mh.createNode(t, node_path, node_info[0], node_info[1])
            if len(node_info) == 3:
                if verbose:
                    print(f"   {node_info[2]}")
                n.putData(node_info[2])


def write(
    plasma,
    pulse: int,
    code_name: str,
    data: dict = None,
    bckc: dict = None,
    run_name="RUN01",
    descr="",
    tree_name="ST40",
    subtree=True,
    force=False,
    verbose=True,
):
    """
    Write HDA data to MDS+

    Parameters
    ----------
    pulse
        Pulse number
    code_name
        Name of code used for analysis
        (e.g. "HDA")
    data
        diagnostic data dictionary
    bckc
        diagnostic back-calculated data dictionary
    run_name
        Name of run under which data should be saved
        (e.g. "RUN01")
    descr
        Description of analysis whose data is going to be saved
    subtree
        Create code_name tree as a subtree of tree_name

    """

    text = "Saving data to "
    text += f"{pulse} {run_name}? (yes)/no   "

    if not (force):
        answer = input(text)
        if answer.lower().strip() == "no":
            print("\n ...writing aborted...")
            return

    t, write_path = create(
        pulse,
        code_name,
        run_name=run_name,
        descr=descr,
        tree_name=tree_name,
        subtree=subtree,
        close=False,
        force=force,
        verbose=verbose,
    )

    print(f"\n {code_name}: Writing results for {pulse} to {write_path}")

    data_to_write = organise_data(plasma, data=data, bckc=bckc)

    write_data(t, write_path, data_to_write, verbose=verbose)

    t.close()


def read(
    pulse: int,
    revision: int,
    uid: str = "",
    instrument: str = "HDA",
    tstart: float = 0.0,
    tend: float = 0.2,
    verbose=False,
):
    """
    Read HDA data from MDS+

    Parameters
    ----------

    Returns
    -------
        Dictionary of quantities contained in HDA MDS+ database

    """
    reader = ST40Reader(pulse, tstart, tend)
    nodes = trees.hda()

    time, dims = reader._get_data(uid, instrument, ":TIME", revision)
    rhop, dims = reader._get_data(uid, instrument, ".PROFILES.PSI_NORM:RHOP", revision)
    data = {}
    for sub_path, quantities in nodes.items():
        for node_name, node_info in quantities.items():
            quantity = f"{sub_path}:{node_name}"
            # TODO: fix MdsCheck in st40reader for strings
            # TODO: fix conn reading of pointers
            _data, _dims = reader._get_data(uid, instrument, quantity, revision)
            if verbose:
                print(quantity)

            if (
                node_name == "RHOP"
                or node_name == "XPSN"
                or node_name == "TIME"
                or np.array_equal(_data, "FAILED")
            ):
                continue

            if sub_path == "" or sub_path == ".METADATA":
                data[node_name] = _data
            elif sub_path == ".GLOBAL":
                data[node_name] = DataArray(_data, coords=[("t", time)])
            elif sub_path == ".PROFILES.PSI_NORM":
                data[node_name] = DataArray(
                    _data, coords=[("t", time), ("rho_poloidal", rhop)]
                )
            else:
                print(f"No known coordinates for sub_path == {sub_path}")

    return data


def organise_data(plasma, data={}, bckc={}):
    """
    Organise HDA data in a dictionary ready to be written to MDS+

    Parameters
    ----------
    plasma
        HDA plasma object

    Returns
    -------
    nodes
        HDA data dictionary
    """

    tev = []
    tiv = []
    nev = []
    niv = []
    zeffv = []
    for t in plasma.time.values:
        tev.append(
            np.trapz(plasma.el_temp.sel(t=t).values, plasma.volume.sel(t=t).values)
            / plasma.volume.sel(t=t).sel(rho_poloidal=1).values,
        )
        tiv.append(
            np.trapz(
                plasma.ion_temp.sel(element=plasma.main_ion).sel(t=t).values,
                plasma.volume.sel(t=t).values,
            )
            / plasma.volume.sel(t=t).sel(rho_poloidal=1).values,
        )
        nev.append(
            np.trapz(plasma.el_dens.sel(t=t).values, plasma.volume.sel(t=t).values)
            / plasma.volume.sel(t=t).sel(rho_poloidal=1).values,
        )
        niv.append(
            np.trapz(
                plasma.ion_dens.sel(element=plasma.main_ion).sel(t=t).values,
                plasma.volume.sel(t=t).values,
            )
            / plasma.volume.sel(t=t).sel(rho_poloidal=1).values,
        )
        zeffv.append(
            np.trapz(
                plasma.zeff.sum("element").sel(t=t).values,
                plasma.volume.sel(t=t).values,
            )
            / plasma.volume.sel(t=t).sel(rho_poloidal=1).values,
        )
    tev = np.array(tev)
    tiv = np.array(tiv)
    nev = np.array(nev)
    niv = np.array(niv)
    zeffv = np.array(zeffv)

    opt_equil = plasma.optimisation["equil"]
    opt_el_dens = plasma.optimisation["el_dens"]
    opt_el_temp = plasma.optimisation["el_temp"]
    opt_ion_temp = plasma.optimisation["ion_temp"]
    opt_stored_en = plasma.optimisation["stored_en"]

    elements = [plasma.main_ion]
    impurities = [""] * 3
    for i, elem in enumerate(plasma.impurities):
        impurities[i] = elem
    elements.extend(impurities)

    ion_conc = []
    ion_meanz = []
    ion_meanz_hi = []
    ion_meanz_lo = []
    ion_dens = []
    ion_dens_hi = []
    ion_dens_lo = []
    ion_temp = []
    ion_temp_hi = []
    ion_temp_lo = []
    ion_zeff = []
    ion_zeff_hi = []
    ion_zeff_lo = []
    for elem in elements:
        if len(elem) > 0:
            conc = (plasma.ion_dens.sel(element=elem) / plasma.el_dens).mean(
                "rho_poloidal"
            )
            meanz = plasma.meanz.sel(element=elem)
            if hasattr(plasma, "meanz_hi"):
                meanz_hi = plasma.meanz_hi.sel(element=elem)
                meanz_lo = plasma.meanz_lo.sel(element=elem)
            else:
                meanz_hi = xr.zeros_like(plasma.meanz.sel(element=elements[0]))
                meanz_lo = xr.zeros_like(plasma.meanz.sel(element=elements[0]))

            dens = plasma.ion_dens.sel(element=elem)
            if hasattr(plasma, "ion_dens_hi"):
                dens_hi = plasma.ion_dens_hi.sel(element=elem)
                dens_lo = plasma.ion_dens_lo.sel(element=elem)
            else:
                dens_hi = xr.zeros_like(plasma.ion_dens.sel(element=elements[0]))
                dens_lo = xr.zeros_like(plasma.ion_dens.sel(element=elements[0]))

            temp = plasma.ion_temp.sel(element=elem)
            if hasattr(plasma, "ion_temp_hi"):
                temp_hi = plasma.ion_temp_hi.sel(element=elem)
                temp_lo = plasma.ion_temp_lo.sel(element=elem)
            else:
                temp_hi = xr.zeros_like(plasma.ion_temp.sel(element=elements[0]))
                temp_lo = xr.zeros_like(plasma.ion_temp.sel(element=elements[0]))

            zeff = plasma.zeff.sel(element=elem)
            if hasattr(plasma, "ion_temp_hi"):
                zeff_hi = plasma.zeff_hi.sel(element=elem)
                zeff_lo = plasma.zeff_lo.sel(element=elem)
            else:
                zeff_hi = xr.zeros_like(plasma.zeff.sel(element=elements[0]))
                zeff_lo = xr.zeros_like(plasma.zeff.sel(element=elements[0]))
        else:
            conc = xr.zeros_like(plasma.t)
            meanz = xr.zeros_like(plasma.meanz.sel(element=elements[0]))
            zeff = xr.zeros_like(plasma.zeff.sel(element=elements[0]))
            dens = xr.zeros_like(plasma.ion_dens.sel(element=elements[0]))
            temp = xr.zeros_like(plasma.ion_temp.sel(element=elements[0]))
        ion_conc.append(conc)
        ion_temp.append(temp)
        ion_temp_hi.append(temp_hi)
        ion_temp_lo.append(temp_lo)
        ion_dens.append(dens)
        ion_dens_hi.append(dens_hi)
        ion_dens_lo.append(dens_lo)
        ion_meanz.append(meanz)
        ion_meanz_hi.append(meanz_hi)
        ion_meanz_lo.append(meanz_lo)
        ion_zeff.append(zeff)
        ion_zeff_hi.append(zeff_hi)
        ion_zeff_lo.append(zeff_lo)

    # TODO: distinguish between Zeff from different elements in MDS+
    if hasattr(plasma, "zeff_hi"):
        zeff_hi = plasma.zeff_hi
        zeff_lo = plasma.zeff_lo
    else:
        zeff_hi = xr.zeros_like(plasma.zeff)
        zeff_lo = xr.zeros_like(plasma.zeff)

    if hasattr(plasma, "el_dens_hi"):
        el_dens_hi = plasma.el_dens_hi
        el_dens_lo = plasma.el_dens_lo
    else:
        el_dens_hi = xr.zeros_like(plasma.el_dens)
        el_dens_lo = xr.zeros_like(plasma.el_dens)

    if hasattr(plasma, "el_temp_hi"):
        el_temp_hi = plasma.el_temp_hi
        el_temp_lo = plasma.el_temp_lo
    else:
        el_temp_hi = xr.zeros_like(plasma.el_temp)
        el_temp_lo = xr.zeros_like(plasma.el_temp)

    if hasattr(plasma, "neutral_dens_hi"):
        neutral_dens_hi = plasma.neutral_dens_hi
        neutral_dens_lo = plasma.neutral_dens_lo
    else:
        neutral_dens_hi = xr.zeros_like(plasma.neutral_dens)
        neutral_dens_lo = xr.zeros_like(plasma.neutral_dens)

    if hasattr(plasma, "pressure_th_hi"):
        pressure_th_hi = plasma.pressure_th_hi
        pressure_th_lo = plasma.pressure_th_lo
    else:
        pressure_th_hi = xr.zeros_like(plasma.pressure_th)
        pressure_th_lo = xr.zeros_like(plasma.pressure_th)

    glob_coord = ["TIME"]
    prof_coord = ["PROFILES.PSI_NORM.RHOP", "TIME"]
    nodes = {
        "": {
            "TIME": (Float32(plasma.time.values), "s", []),
        },  # (values, units, coordinate node name)
        ".METADATA": {
            "PULSE": (Float32(plasma.pulse), "", []),
            "EQUIL": (String(opt_equil), "", []),
            "EL_DENS": (String(opt_el_dens), "", []),
            "EL_TEMP": (String(opt_el_temp), "", []),
            "ION_TEMP": (String(opt_ion_temp), "", []),
            "STORED_EN": (String(opt_stored_en), "", []),
            "MAIN_ION": (String(plasma.main_ion), "", []),
            "IMPURITY1": (String(impurities[0]), "", []),
            "IMPURITY2": (String(impurities[1]), "", []),
            "IMPURITY3": (String(impurities[2]), "", []),
        },
        ".GLOBAL": {
            "CR0": (Float32(plasma.cr0.values), "m", glob_coord,),
            "RMAG": (Float32(plasma.rmag.values), "m", glob_coord,),
            "ZMAG": (Float32(plasma.zmag.values), "m", glob_coord,),
            "VOLM": (
                Float32(plasma.volume.sel(rho_poloidal=1).values),
                "m^3",
                glob_coord,
            ),
            "IP": (Float32(plasma.ipla.values), "A", glob_coord,),
            "TE0": (
                Float32(plasma.el_temp.sel(rho_poloidal=0).values),
                "eV",
                glob_coord,
            ),
            "TI0": (Float32(ion_temp[0].sel(rho_poloidal=0).values), "eV", glob_coord,),
            "TI0_Z1": (
                Float32(ion_temp[1].sel(rho_poloidal=0).values),
                "eV",
                glob_coord,
            ),
            "TI0_Z2": (
                Float32(ion_temp[2].sel(rho_poloidal=0).values),
                "eV",
                glob_coord,
            ),
            "TI0_Z3": (
                Float32(ion_temp[3].sel(rho_poloidal=0).values),
                "eV",
                glob_coord,
            ),
            "NE0": (
                Float32(plasma.el_dens.sel(rho_poloidal=0).values),
                "m^-3 ",
                glob_coord,
            ),
            "NI0": (
                Float32(ion_dens[0].sel(rho_poloidal=0).values),
                "m^-3 ",
                glob_coord,
            ),
            "ZEFF": (
                Float32(plasma.zeff.sum("element").sel(rho_poloidal=0).values),
                "",
                glob_coord,
            ),
            "TEV": (Float32(tev), "eV", glob_coord),
            "TIV": (Float32(tiv), "eV", glob_coord),
            "NEV": (Float32(nev), "m^-3", glob_coord),
            "NIV": (Float32(niv), "m^-3", glob_coord),
            "ZEFFV": (Float32(zeffv), "", glob_coord,),
            "WTH": (Float32(plasma.wth.values), "J", glob_coord),
            "WP": (Float32(plasma.wp.values), "J", glob_coord),
            "UPL": (Float32(plasma.vloop.values), "V", glob_coord),
            "CION": (Float32(ion_conc[0].values), "", glob_coord),
            "CIM1": (Float32(ion_conc[1].values), "", glob_coord),
            "CIM2": (Float32(ion_conc[2].values), "", glob_coord),
            "CIM3": (Float32(ion_conc[3].values), "", glob_coord),
        },
        ".PROFILES.PSI_NORM": {
            "RHOP": (Float32(plasma.rho.values), "", ()),
            "XPSN": (Float32(plasma.rho.values ** 2), "", ()),
            "P": (Float32(plasma.pressure_th.values), "Pa", prof_coord,),
            "P_HI": (Float32(pressure_th_hi.values), "Pa", prof_coord,),
            "P_LO": (Float32(pressure_th_lo.values), "Pa", prof_coord,),
            "VOLUME": (Float32(plasma.volume.values), "m^3", prof_coord,),
            "NE": (Float32(plasma.el_dens.values), "m^-3", prof_coord,),
            "NE_HI": (Float32(el_dens_hi.values), "m^-3", prof_coord,),
            "NE_LO": (Float32(el_dens_lo.values), "m^-3", prof_coord,),
            "NI": (Float32(ion_dens[0].values), "m^-3", prof_coord,),
            "NI_HI": (Float32(ion_dens_hi[0].values), "m^-3", prof_coord,),
            "NI_LO": (Float32(ion_dens_lo[0].values), "m^-3", prof_coord,),
            "NIZ1": (Float32(ion_dens[1].values), "", prof_coord,),
            "NIZ1_HI": (Float32(ion_dens_hi[0].values), "m^-3", prof_coord,),
            "NIZ1_LO": (Float32(ion_dens_lo[0].values), "m^-3", prof_coord,),
            "NIZ2": (Float32(ion_dens[2].values), "", prof_coord,),
            "NIZ2_HI": (Float32(ion_dens_hi[2].values), "m^-3", prof_coord,),
            "NIZ2_LO": (Float32(ion_dens_lo[2].values), "m^-3", prof_coord,),
            "NIZ3": (Float32(ion_dens[3].values), "", prof_coord,),
            "NIZ3_HI": (Float32(ion_dens_hi[3].values), "m^-3", prof_coord,),
            "NIZ3_LO": (Float32(ion_dens_lo[3].values), "m^-3", prof_coord,),
            "NNEUTR": (Float32(plasma.neutral_dens.values), "m^-3", prof_coord,),
            "NNEUTR_HI": (Float32(neutral_dens_hi.values), "m^-3", prof_coord,),
            "NNEUTR_LO": (Float32(neutral_dens_lo.values), "m^-3", prof_coord,),
            "TE": (Float32(plasma.el_temp.values), "eV", prof_coord,),
            "TE_HI": (Float32(el_temp_hi.values), "eV", prof_coord,),
            "TE_LO": (Float32(el_temp_lo.values), "eV", prof_coord,),
            "TI": (Float32(ion_temp[0].values), "eV", prof_coord,),
            "TI_HI": (Float32(ion_temp_hi[0].values), "eV", prof_coord,),
            "TI_LO": (Float32(ion_temp_lo[0].values), "eV", prof_coord,),
            "TIZ1": (Float32(ion_temp[1].values), "eV", prof_coord,),
            "TIZ1_HI": (Float32(ion_temp_hi[1].values), "eV", prof_coord,),
            "TIZ1_LO": (Float32(ion_temp_lo[1].values), "eV", prof_coord,),
            "TIZ2": (Float32(ion_temp[2].values), "eV", prof_coord,),
            "TIZ2_HI": (Float32(ion_temp_hi[2].values), "eV", prof_coord,),
            "TIZ2_LO": (Float32(ion_temp_lo[2].values), "eV", prof_coord,),
            "TIZ3": (Float32(ion_temp[3].values), "eV", prof_coord,),
            "TIZ3_HI": (Float32(ion_temp_hi[3].values), "eV", prof_coord,),
            "TIZ1_LO": (Float32(ion_temp_lo[3].values), "eV", prof_coord,),
            "ZEFF": (Float32(plasma.zeff.sum("element").values), "", prof_coord,),
            "ZEFF_HI": (Float32(zeff_hi.sum("element").values), "", prof_coord,),
            "ZEFF_LO": (Float32(zeff_lo.sum("element").values), "", prof_coord,),
            "ZI": (Float32(ion_meanz[0].values), "", prof_coord,),
            "ZIM1": (Float32(ion_meanz[1].values), "", prof_coord,),
            "ZIM2": (Float32(ion_meanz[2].values), "", prof_coord,),
            "ZIM3": (Float32(ion_meanz[3].values), "", prof_coord,),
        },
    }
    # "RHOT": (
    #     Float32(plasma.rhot.values),
    #     "",
    #     prof_coord,
    # ),

    return nodes


def write_data(t, write_path, data, verbose=False):
    """
    Write HDA data to MDS+

    Parameters
    ----------
    t
        Tree object
    write_path
        Full path to MDS for writing data
    data
        HDA data dictionary as built by organise_data
    """

    print(f"\n Writing data to {write_path}")
    for sub_path, quantities in data.items():
        for node_name, node_info in quantities.items():
            node_path = f"{write_path}{sub_path}:{node_name}"
            build_str = "build_signal(build_with_units($1,$2), * "

            if verbose:
                print(node_path)
            n = t.getNode(node_path)

            node_data, node_units, node_dims = node_info

            # Distinguish if dimensions are present or not
            if len(node_dims) == 0:
                n.putData(node_data.setUnits(node_units))
            else:
                for dim in node_dims:
                    build_str += f", {dim}"
                build_str += ")"
                n.putData(t.tdiCompile(build_str, node_data, node_units))
    print("\n Writing completed")
