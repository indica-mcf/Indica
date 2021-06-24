# Script for creating the model tree for ST40 HDA tree
# Script for creating the model tree for ST40 EQUIL tree
# Alexei -- 07/2019
# Peter Buxton -- added username and checks before deleting -- Feb / 2021
# Peter Buxton -- added  copy_runs and warning_message  -- Feb / 2021
# Marco Sertoli -- Modified to write HDA tree  -- Jun / 2021

from MDSplus import *
from numpy import *
import numpy as np
import hda.mdsHelpers as mh
from importlib import reload

reload(mh)
import getpass

user = getpass.getuser()
# MDSplus_IP_address = '192.168.1.7:8000'  # smaug IP address


def test():
    pulseNo = 18999999
    run_name = "RUN01"
    tree_name = "HDA"
    descr = "HDA test tree"
    create(pulseNo, run_name, descr, tree_name)

def create(pulseNo, run_name: str, descr, tree_name="HDA", subtree_name=""):

    ###############################################################
    ####################    Create the tree    ####################
    ##############################################################

    run_name = run_name.upper().strip()
    tree_name = tree_name.upper().strip()
    subtree_name = subtree_name.upper().strip()
    if len(subtree_name) == 0:
        print(f"\n # Writing to {tree_name} requires subtree_name != "" # \n")
        tree_path = f"\{tree_name}::TOP:{subtree_name}"
    else:
        tree_path = f"\{tree_name}::TOP"

    try:
        mode = "EDIT"
        t = Tree(tree_name, pulseNo, mode)

        try:
            n = t.getNode(rf"{tree_name_full}.{run_name}.METADATA:USER")
            user_already_written = n.data()
        except:
            user_already_written = user

        if not (user_already_written == user):
            print(f"\n #  You are about to overwrite {run_name}! # \n")
            print(" Proceed yes/(no)? ")
            yes_typed = input(">>  ")
            if not(yes_typed.lower() == "yes"):
                return
    except:
        mode = "NEW"
        if tree_name=="ST40":
            if pulseNo<1.e6:
                print(f"\n # Creating new ST40 tree enabled only for modelling pulses ! # \n")
                return
        t = Tree(tree_name, pulseNo, mode)

    if len(subtree_name) > 0:
        t.addNode(tpath, "SUBTREE")

    t.setDefault(mh.createNode(t, branches[0], "STRUCTURE", "Metadata of analysis"))
    t.addNode(f"{tree_path}.METADATA", "STRUCTURE")
    n = t.addNode(f"{tree_path}.METADATA:USER", "TEXT")
    n.putData(user)
    n = t.addNode(f"{tree_path}.METADATA:PULSE", "TEXT")
    n = t.addNode(f"{tree_path}.METADATA:EFIT_RUN", "TEXT")

    n = mh.createNode(t, f"{tree_path}.TIME", "NUMERIC", "time vector, s")
    n = mh.createNode(t, f"{tree_path}.GLOBAL.TE0 ", "SIGNAL", "Central electron temp, eV")
    n = mh.createNode(t, f"{tree_path}.GLOBAL.TI0 ", "SIGNAL", "Central ion temp, eV")
    n = mh.createNode(t, f"{tree_path}.GLOBAL.NE0 ", "SIGNAL", "Central electron density, m^-3 ")
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.NEL ","SIGNAL","Line aver electron density m^-3 ");
    n = mh.createNode(t, f"{tree_path}.GLOBAL.NEV ", "SIGNAL", "Volume aver electron density m^-3 ")
    n = mh.createNode(t, f"{tree_path}.GLOBAL.TEV ", "SIGNAL", "Volume aver electron temp, eV")
    n = mh.createNode(t, f"{tree_path}.GLOBAL.TIV ", "SIGNAL", "Volume aver ion temp, eV")
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.VLOOP ", "SIGNAL", "Loop voltage, V")
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.P_OH","SIGNAL","Total Ohmic power, M");
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.UPL ","SIGNAL","Loop Voltage,V          ");
    n = mh.createNode(t, f"{tree_path}.GLOBAL.WTH ", "SIGNAL", "Thermal energy, J       ")
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.Li3 ","SIGNAL","Internal inductance     ");
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.BetP","SIGNAL","Poloidal beta           ");
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.BetT","SIGNAL","Toroidal beta           ");
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.BetN","SIGNAL","Beta normalized  ");
    n = mh.createNode(t,  f"{tree_path}.GLOBAL.ZEFF", "SIGNAL", "Z effective at the plasma center")
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.Res',"SIGNAL","Total plasma resistance Qj/Ipl^2, Ohm");
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.I_BS","SIGNAL","Total bootstrap current,A")
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.F_BS","SIGNAL","Bootstrap current fraction")
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.I_OH","SIGNAL","Total Ohmic current,A")
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.F_NI","SIGNAL","Non-inductive current fraction")
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.P_AUX","SIGNAL","Total external heating power,W")
    # n = mh.createNode(t, f"{tree_path}.GLOBAL.VTOR0","SIGNAL","Central toroidal velocity, m/s")

    n = mh.createNode(
        t,  f"{tree_path}.PROFILES.PSI_NORM.RHOP", "NUMERIC", "radial vector, Sqrt of normalised poloidal flux"
    )
    n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.XPSN", "NUMERIC", "x vector - fi_normalized")
    # n = mh.createNode(t,f"{tree_path}.PROFILES.PSI_NORM.Q","SIGNAL","Q_PROFILE(PSI_NORM)");
    n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.P", "SIGNAL", "Pressure,Pa")
    n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.VOLUME", "SIGNAL", "Volume inside magnetic surface,m^3")

    n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.RHOT", "SIGNAL", "Sqrt of normalised toroidal flux, xpsn")
    n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.TE", "SIGNAL", "Electron temperature, eV")
    n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.NE", "SIGNAL", "Electron density, m^-3")
    n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.NI", "SIGNAL", "Main ion density, m^-3")
    n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.TI", "SIGNAL", "Ion temperature, eV")
    n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.ZEFF", "SIGNAL", "Effective ion charge")
    # n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.CC","SIGNAL","Parallel current conductivity, 1/(Ohm*m)");
    # n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.N_D","SIGNAL","Deuterium density,1/m^3")
    # n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.N_T","SIGNAL","Tritium density	,1/m^3")
    # n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.T_D","SIGNAL","Deuterium temperature,eV")
    # n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.T_T","SIGNAL","Tritium temperature,eV")
    # n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.J_BS","SIGNAL","Bootstrap current density,M/m2")
    # n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.J_OH","SIGNAL","Ohmic current density,A/m2")
    # n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.J_TOT","SIGNAL","Total current density,A/m2")
    # n = mh.createNode(t, f"{tree_path}.PROFILES.PSI_NORM.OMEGA_TOR","SIGNAL","Toroidal rotation frequency, 1/s");
    t.write()
    t.close

def warning_message(pulseNo, run_name):
    run_name = run_name.upper().strip()

    pulseNo_str = str(pulseNo)
    print("#####################################################")
    print("#  *** WARNING ***                                  #")
    print("#  You are about to overwrite data                  #")
    spaces = " " * (41 - len(pulseNo_str))
    print("#  pulseNo=" + pulseNo_str + spaces + "#")
    spaces = " " * (49 - len(run_name))
    print("#  " + node + spaces + "#")
    print("#####################################################")
    print(" Proceed yes/no?")
    yes_typed = input(">>  ")
    if (yes_typed.lower() == "no") or (yes_typed.lower() == "n"):
        return
    while not (not (yes_typed.lower() == "yes") or not (yes_typed.lower() == "y")):
        print(" Error try again")
        yes_typed = input(">>  ")

# def modifyhelp(pulseNo, run_name: str, descr, tree_name="HDA"):
#
#     run_name = run_name.upper().strip()
#     tree_name = tree_name.upper().strip()
#     try:
#         t = Tree(tree_name, pulseNo, "edit")
#     except:
#         t = Tree(tree_name, pulseNo, "New")
#     hda = t.getDefault()
#     t.setDefault(hda)
#     descr0 = t.getNode(run_name + ":HELP").getData()
#     print(descr0)
#     t.getNode(run_name + ":HELP").putData(descr)
#     t.write()
#     descr1 = t.getNode(run_name + ":HELP").getData()
#     print(descr1)
#     t.close

# def addglobal(pulseNo, run_name: str, addnode, descr, tree_name="HDA"):
#     run_name = run_name.upper().strip()
#     tree_name = tree_name.upper().strip()
#     try:
#         t = Tree(tree_name, pulseNo, "edit")
#     except:
#         t = Tree(tree_name, pulseNo, "New")
#     t.setDefault(t.getNode("\\TOP." + run_name + ".GLOBAL"))
#     n = mh.createNode(t, addnode, "SIGNAL", descr)
#     t.write()
#     t.close


# def copy_runs(pulseNo_from, run_from, pulseNo_to, run_to, tree_name):
#     run_from = run_from.upper().strip()
#     run_to = run_to.upper().strip()
#     tree_name = tree_name.upper().strip()
#
#     # Example usage:
#     # move_runs(314, 'RUN1', 1000004, 'RUN1', 'ASTRA')
#
#     path_from = "\\" + tree_name + "::TOP." + run_from
#     path_to = "\\" + tree_name + "::TOP." + run_to
#     print(path_from)
#
#     # Read what we want to move:
#     t_from = Tree(tree_name, pulseNo_from)
#     command = "GETNCI('\\" + path_from + "***','FULLPATH')"
#     fullpaths_from = t_from.tdiExecute(command).data().astype(str, copy=False).tolist()
#     command = "GETNCI('\\" + path_from + "***','USAGE')"
#     usages_from = t_from.tdiExecute(command).data()
#
#     # Read where we want to
#     try:
#         t_to = Tree(tree_name, pulseNo_to, "EDIT")
#         print("editing...")
#     except:
#         t_to = Tree(tree_name, pulseNo_to, "NEW")
#         print("new...")
#
#     # Add the run if needed
#     try:
#         run_node_to = t_to.getNode(path_to)
#
#         # Command line warning_message
#         warning_message(pulseNo_to, path_to)
#
#         # Delete node
#         t_to.deleteNode(run_node_to)
#     except:
#         pass
#     # Add a new fully empty node
#     t_to.addNode(path_to)
#
#     for i in range(0, len(fullpaths_from)):
#         fullpath_from = fullpaths_from[i].strip()
#         fullpath_to = fullpaths_from[i].replace(path_from, path_to).strip()
#         usage = usages_from[i]
#         if usage == 1:
#             datatype = "STRUCTURE"
#         elif usage == 5:
#             datatype = "NUMERIC"
#         elif usage == 6:
#             datatype = "SIGNAL"
#         elif usage == 8:
#             datatype = "TEXT"
#         elif usage == 11:
#             datatype = "SUBTREE"
#         else:
#             print("UNKNOWN DATA TYPE!!")
#         # Make the node
#         n = t_to.addNode(fullpath_to, datatype)
#
#         # Move NUMBER, SIGNAL or TEXT
#         if (usage == 5) or (usage == 6) or (usage == 8):
#             n_from = t_from.getNode(fullpath_from)
#             n_to = t_to.getNode(fullpath_to)
#             n_to.putData(n_from.getRecord())
#
#     t_to.write()
#     t_to.close()
#     t_from.close()
#
#     print("Data successfully moved")
#

## look at /home/ops/mds_trees/ for inspiration
# def delete(pulseNo, run_name: str):
#     t = Tree("HDA", pulseNo, "edit")
#
#     run_name = run_name.upper().strip()
#
#     # get the username of who wrote this run
#     try:
#         n = t.getNode(rf"\HDA::TOP.{run_name}.CODE_VERSION:USER")
#         user_already_written = n.data()
#     except:
#         user_already_written = user
#
#     # First warning if you are going to delete someone else' run
#     if not (user_already_written == user):
#         print("#####################################################")
#         print("#  *** WARNING ***                                  #")
#         print("#  You are about to delete a different user's run!  #")
#         nspaces = 49 - len(user_already_written)
#         spaces = " " * nspaces
#         print("#  " + user_already_written + spaces + "#")
#         print("#####################################################")
#
#         print(" Proceed yes/no?")
#         yes_typed = input(">>  ")
#         if (yes_typed.lower() == "no") or (yes_typed.lower() == "n"):
#             return
#         while not (not (yes_typed.lower() == "yes") or not (yes_typed.lower() == "y")):
#             print(" Error try again")
#             yes_typed = input(">>  ")
#
#         print(' To confirm type in: "' + user_already_written + '"')
#         user_typed = input(">>  ")
#         while not (user_already_written == user_typed):
#             print(" Error try again")
#             user_typed = input(">>  ")
#         print(" ")
#
#     # Second warning to confirm delete
#     print("#####################################################")
#     print("#  *** WARNING ***                                  #")
#     print("#  You are about to delete data                     #")
#     nspaces = 49 - len(user_already_written)
#     spaces = " " * nspaces
#     print(f"# {pulseNo} {tree_name} {run_name}" + spaces + "#")
#     print("#####################################################")
#     print(" Proceed yes/no?")
#     yes_typed = input(">>  ")
#     if (yes_typed.lower() == "no") or (yes_typed.lower() == "n"):
#         return
#     while not (not (yes_typed.lower() == "yes") or not (yes_typed.lower() == "y")):
#         print(" Error try again")
#         yes_typed = input(">>  ")
#
#     # Delete
#     t.deleteNode(run_name)
#     t.write()
#     t.close
#     print(" Data deleted")
#
