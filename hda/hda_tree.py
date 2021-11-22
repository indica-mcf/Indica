# Script for creating the model tree for ST40 HDA tree

from MDSplus import *
from MDSplus.mdsExceptions import TreeALREADY_THERE, TreeFOPENW, TreeNNF
import numpy as np
from MDSplus import Tree, Float32, Int32, String
import hda.mdsHelpers as mh
import getpass

user = getpass.getuser()


def create(
    pulse: int,
    code_name: str,
    run_name="RUN01",
    descr="",
    tree_name="ST40",
    subtree=True,
    close=True,
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
    try:
        t = Tree(code_name, pulse, "EDIT")
    except TreeFOPENW:
        print(f"\n Creating new tree {code_name}")
        t = Tree(code_name, pulse, "NEW")

    # Create/Overwrite RUN structure
    write_path = f"{code_path}.{run_name}"

    try:
        t.getNode(rf"{write_path}")
        print(f"\n {pulse}:{write_path} tree already exists")
        answer = input(f"\n # Overwrite {run_name} * yes/(no) * ?  ")
        if answer.lower() == "yes":
            delete(t, write_path)
        else:
            run_name = next_run(t, code_path, run_name)
            answer = input(
                f"\n # Write to next available run {run_name} * yes/(no) * ?  "
            )
            if answer.lower() == "no":
                return None
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
    nodes = get_tree_structure()

    # Create nodes
    t = Tree(code_name, pulse, "EDIT")
    create_nodes(t, write_path, nodes)

    # Write and close Tree
    t.write()
    if close:
        t.close()

    return t, write_path


def delete(t, full_path):
    # Second warning to confirm delete
    print("#####################################################")
    print("#  *** WARNING ***                                  #")
    print("   You are about to delete data   ")
    print(f"{full_path}")
    print("#####################################################")

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


def get_tree_structure():

    nodes = {
        "": {"TIME": ("NUMERIC", "time vector, s"),},  # (type, description)
        ".METADATA": {
            "USER": ("TEXT", "Username of owner", user),
            "PULSE": ("NUMERIC", "Pulse number analysed"),
            "EQUIL": ("TEXT", "Equilibrium used"),
            "EL_DENS": ("TEXT", "Electron density diagnostic used for optimization"),
            "EL_TEMP": (
                "TEXT",
                "Electron temperature diagnostic used for optimization",
            ),
            "ION_TEMP": ("TEXT", "Ion temperature diagnostic used for optimization"),
            "STORED_EN": ("TEXT", "Stored energy diagnostic used for optimization"),
            "MAIN_ION": ("TEXT", "Main ion element"),
            "IMPURITY1": ("TEXT", "Impurity element chosen for Z1"),
            "IMPURITY2": ("TEXT", "Impurity element chosen for Z2"),
            "IMPURITY3": ("TEXT", "Impurity element chosen for Z3"),
        },
        ".GLOBAL": {
            "CR0": ("SIGNAL", "Minor radius = (R_LFS - R_HFS)/2 at midplane, m"),
            "RMAG": ("SIGNAL", "Magnetic axis R, m"),
            "ZMAG": ("SIGNAL", "Magnetic axis z, m"),
            "VOLM": ("SIGNAL", "Plasma volume z, m^3"),
            "IP": ("SIGNAL", "Plasma current, A"),
            "TE0": ("SIGNAL", "Central electron temp, eV"),
            "TI0": ("SIGNAL", "Central main ion temp, eV"),
            "TI0_Z1": ("SIGNAL", "Central impurity1 ion temp, eV"),
            "TI0_Z2": ("SIGNAL", "Central impurity2 ion temp, eV"),
            "TI0_Z3": ("SIGNAL", "Central impurity3 ion temp, eV"),
            "NE0": ("SIGNAL", "Central electron density, m^-3 "),
            "NI0": ("SIGNAL", "Central main ion density, m^-3 "),
            "TEV": ("SIGNAL", "Volume average electron temp, eV"),
            "TIV": ("SIGNAL", "Volume average ion temp, eV"),
            "NEV": ("SIGNAL", "Volume average electron density m^-3"),
            "NIV": ("SIGNAL", "Volume average main ion density m^-3"),
            "WP": ("SIGNAL", "Total stored energy, J"),
            "WTH": ("SIGNAL", "Thermal stored energy, J"),
            "UPL": ("SIGNAL", "Loop Voltage, V"),
            "P_OH": ("SIGNAL", "Total Ohmic power, W"),
            "ZEFF": ("SIGNAL", "Effective charge at the plasma center"),
            "CION": ("SIGNAL", "Average concentration of main ion"),
            "CIM1": ("SIGNAL", "Average concentration of impurity IMP1"),
            "CIM2": ("SIGNAL", "Average concentration of impurity IMP2"),
            "CIM3": ("SIGNAL", "Average concentration of impurity IMP3"),
        },
        ".PROFILES.PSI_NORM": {
            "RHOP": ("NUMERIC", "Radial vector, Sqrt of normalised poloidal flux"),
            "XPSN": ("NUMERIC", "x vector - fi_normalized"),
            "P": ("SIGNAL", "Pressure,Pa"),
            "VOLUME": ("SIGNAL", "Volume inside magnetic surface,m^3"),
            "NE": ("SIGNAL", "Electron density, m^-3"),
            "NI": ("SIGNAL", "Ion density, m^-3"),
            "TE": ("SIGNAL", "Electron temperature, eV"),
            "TI": ("SIGNAL", "Ion temperature of main ion, eV"),
            "TIZ1": ("SIGNAL", "Ion temperature of impurity IMP1, eV"),
            "TIZ2": ("SIGNAL", "Ion temperature of impurity IMP2, eV"),
            "TIZ3": ("SIGNAL", "Ion temperature of impurity IMP3, eV"),
            "NIZ1": ("SIGNAL", "Density of impurity IMP1, m^-3"),
            "NIZ2": ("SIGNAL", "Density of impurity IMP2, m^-3"),
            "NIZ3": ("SIGNAL", "Density of impurity IMP3, m^-3"),
            "NNEUTR": ("SIGNAL", "Density of neutral main ion, m^-3"),
            "ZI": ("SIGNAL", "Average charge of main ion, "),
            "ZIM1": ("SIGNAL", "Average charge of impurity IMP1, "),
            "ZIM2": ("SIGNAL", "Average charge of impurity IMP2, "),
            "ZIM3": ("SIGNAL", "Average charge of impurity IMP3, "),
            "ZEFF": ("SIGNAL", "Effective charge, "),
        },
    }
    # "RHOT": ("SIGNAL", "Sqrt of normalised toroidal flux, xpsn"),

    return nodes


def create_nodes(t, run_path, nodes):

    t.addNode(f"{run_path}.METADATA", "STRUCTURE")
    t.addNode(f"{run_path}.GLOBAL", "STRUCTURE")
    t.addNode(f"{run_path}.PROFILES", "STRUCTURE")
    t.addNode(f"{run_path}.PROFILES.PSI_NORM", "STRUCTURE")

    for sub_path, quantities in nodes.items():
        for node_name, node_info in quantities.items():
            node_path = f"{run_path}{sub_path}:{node_name}"
            print(node_path, node_info[0], node_info[1])
            n = mh.createNode(t, node_path, node_info[0], node_info[1])
            if len(node_info) == 3:
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
    )

    print(f"\n {code_name}: Writing results for {pulse} to {write_path}")

    data_to_write = organise_data(plasma, data=data, bckc=bckc)

    write_data(t, write_path, data_to_write)

    t.close()


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
    tev = np.array(tev)
    tiv = np.array(tiv)
    nev = np.array(nev)
    niv = np.array(niv)

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
    ion_dens = []
    ion_temp = []
    for elem in elements:
        if len(elem) > 0:
            conc = (plasma.ion_dens.sel(element=elem) / plasma.el_dens).mean(
                "rho_poloidal"
            )
            meanz = plasma.meanz.sel(element=elem)
            dens = plasma.ion_dens.sel(element=elem)
            temp = plasma.ion_temp.sel(element=elem)
        else:
            conc = np.zeros_like(plasma.t)
            meanz = np.zeros_like(plasma.meanz.sel(element=elements[0]))
            dens = np.zeros_like(plasma.ion_dens.sel(element=elements[0]))
            temp = np.zeros_like(plasma.ion_temp.sel(element=elements[0]))
        ion_conc.append(conc)
        ion_meanz.append(meanz)
        ion_dens.append(dens)
        ion_temp.append(temp)

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
            "TEV": (Float32(tev), "eV", glob_coord),
            "TIV": (Float32(tiv), "eV", glob_coord),
            "NEV": (Float32(nev), "m^-3", glob_coord),
            "NIV": (Float32(niv), "m^-3", glob_coord),
            "WTH": (Float32(plasma.wth.values), "J", glob_coord),
            "WP": (Float32(plasma.wp.values), "J", glob_coord),
            "UPL": (Float32(plasma.vloop.values), "V", glob_coord),
            "ZEFF": (
                Float32(plasma.zeff.sum("element").sel(rho_poloidal=0).values),
                "",
                glob_coord,
            ),
            "CION": (Float32(ion_conc[0].values), "", glob_coord),
            "CIM1": (Float32(ion_conc[1].values), "", glob_coord),
            "CIM2": (Float32(ion_conc[2].values), "", glob_coord),
            "CIM3": (Float32(ion_conc[3].values), "", glob_coord),
        },
        ".PROFILES.PSI_NORM": {
            "RHOP": (Float32(plasma.rho.values), "", ()),
            "XPSN": (Float32(plasma.rho.values ** 2), "", ()),
            "P": (Float32(plasma.pressure_th.values), "Pa", prof_coord,),
            "VOLUME": (Float32(plasma.volume.values), "m^3", prof_coord,),
            "NE": (Float32(plasma.el_dens.values), "m^-3", prof_coord,),
            "NI": (Float32(ion_dens[0].values), "m^-3", prof_coord,),
            "NIZ1": (Float32(ion_dens[1].values), "", prof_coord,),
            "NIZ2": (Float32(ion_dens[2].values), "", prof_coord,),
            "NIZ3": (Float32(ion_dens[3].values), "", prof_coord,),
            "NNEUTR": (Float32(plasma.neutral_dens.values), "m^-3", prof_coord,),
            "TE": (Float32(plasma.el_temp.values), "eV", prof_coord,),
            "TI": (Float32(ion_temp[0].values), "eV", prof_coord,),
            "TIZ1": (Float32(ion_temp[1].values), "eV", prof_coord,),
            "TIZ2": (Float32(ion_temp[2].values), "eV", prof_coord,),
            "TIZ3": (Float32(ion_temp[3].values), "eV", prof_coord,),
            "ZEFF": (Float32(plasma.zeff.sum("element").values), "", prof_coord,),
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


def write_data(t, write_path, data):
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
