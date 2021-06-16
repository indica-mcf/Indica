# June 2021
# Module to write to HDA data to MDSplus
# Adapted from script /ppac/xrc_spectrometer/hda_write.py by Jon Wood

import numpy as np
from MDSplus import Tree, Float32, Int32, String
from hda import HDAdata
from indica.readers import ST40Reader

def test(hdadata):
    pulseNo = 18999999
    whichRun = "RUN01"
    write_to_mdsplus(pulseNo, hdadata, whichRun=whichRun)


def write_to_mdsplus(pulseNo: int, hdadata: HDAdata, whichRun="RUN01", whichTree="HDA"):

    print(f"HDA: Writing results to {pulseNo} run number {whichRun}")

    t = Tree(whichTree, pulseNo)

    if whichTree == "HDA":
        prefix = whichRun + "."
    else:
        prefix = "HDA." + whichRun + "."

    print("...METADATA...")
    n = t.getNode(prefix + "METADATA.EFIT_RUN")
    data = hdadata.raw_data["efit"]["run"]
    n.putData(str(data))

    n = t.getNode(prefix + "METADATA.PULSE")
    data = hdadata.pulse
    n.putData(str(data))

    time_node = prefix + "TIME"
    n = t.getNode(time_node)
    data_time = hdadata.time.values
    n.putData(Float32(data_time).setUnits("s"))

    print("...GLOBAL...")
    build_str = f"build_signal(build_with_units($1,$2), *, {time_node})"

    n = t.getNode(prefix + "GLOBAL.NE0")
    data = hdadata.el_dens.sel(rho_poloidal=0).values
    n.putData(t.tdiCompile(build_str, data, "m^-3"))

    n = t.getNode(prefix + "GLOBAL.NEV")
    data = []
    for tt in data_time:
        data.append(
            np.trapz(hdadata.el_dens.sel(t=tt).values, hdadata.volume.sel(t=tt).values)
            / hdadata.volume.sel(t=tt).sel(rho_poloidal=1).values,
        )
    data = np.array(data)
    n.putData(t.tdiCompile(build_str, data, "m^-3"))

    n = t.getNode(prefix + "GLOBAL.TE0")
    data = hdadata.el_temp.sel(rho_poloidal=0).values
    n.putData(t.tdiCompile(build_str, data, "eV"))

    n = t.getNode(prefix + "GLOBAL.TEV")
    data = []
    for tt in data_time:
        data.append(
            np.trapz(hdadata.el_temp.sel(t=tt).values, hdadata.volume.sel(t=tt).values)
            / hdadata.volume.sel(t=tt).sel(rho_poloidal=1).values,
        )
    data = np.array(data)
    n.putData(t.tdiCompile(build_str, data, "eV"))

    n = t.getNode(prefix + "GLOBAL.TI0")
    data = hdadata.ion_temp.sel(element=hdadata.main_ion).sel(rho_poloidal=0).values
    n.putData(t.tdiCompile(build_str, data, "eV"))

    n = t.getNode(prefix + "GLOBAL.TIV")
    data = []
    for tt in data_time:
        data.append(
            np.trapz(
                hdadata.ion_temp.sel(element=hdadata.main_ion).sel(t=tt).values,
                hdadata.volume.sel(t=tt).values,
            )
            / hdadata.volume.sel(t=tt).sel(rho_poloidal=1).values,
        )
    data = np.array(data)
    n.putData(t.tdiCompile(build_str, data, "eV"))

    n = t.getNode(prefix + "GLOBAL.VLOOP")
    data = hdadata.vloop.values
    n.putData(t.tdiCompile(build_str, data, "V"))

    n = t.getNode(prefix + "GLOBAL.WTH")
    data = hdadata.wmhd.values
    n.putData(t.tdiCompile(build_str, data, "J"))

    n = t.getNode(prefix + "GLOBAL.ZEFF")
    data = hdadata.zeff.sum("element").sel(rho_poloidal=0).values
    n.putData(t.tdiCompile(build_str, data, ""))

    print("...PROFILES...")
    rhop_node = prefix + "PROFILES.PSI_NORM.RHOP"
    n = t.getNode(rhop_node)
    data = hdadata.rho.values
    n.putData(Float32(data).setUnits(""))

    n = t.getNode(prefix + "PROFILES.PSI_NORM.XPSN")
    data = hdadata.rho.values ** 2
    n.putData(Float32(data).setUnits(""))

    build_str = f"build_signal(build_with_units($1,$2), *, {rhop_node}, {time_node})"
    n = t.getNode(prefix + "PROFILES.PSI_NORM.CC")
    data = hdadata.conductivity.values
    n.putData(t.tdiCompile(build_str, data, "1/(Ohm*m)"))

    n = t.getNode(prefix + "PROFILES.PSI_NORM.J_TOT")
    data = hdadata.j_phi.values
    n.putData(t.tdiCompile(build_str, data, "A/m^2"))

    n = t.getNode(prefix + "PROFILES.PSI_NORM.NE")
    data = hdadata.el_dens.values
    n.putData(t.tdiCompile(build_str, data, "m^-3"))

    n = t.getNode(prefix + "PROFILES.PSI_NORM.NI")
    data = hdadata.ion_dens.sel(element=hdadata.main_ion).values
    n.putData(t.tdiCompile(build_str, data, "m^-3"))

    n = t.getNode(prefix + "PROFILES.PSI_NORM.P")
    data = hdadata.pressure_th.values
    n.putData(t.tdiCompile(build_str, data, "Pa"))

    # n = t.getNode(prefix + "PROFILES.PSI_NORM.RHOT")
    # data = hdadata.rhot.values
    # units = "m^-3"
    # n.putData(t.tdiCompile(
    #     build_str,
    #     data, units, xdata, xunits, ydata, yunits
    # ))

    n = t.getNode(prefix + "PROFILES.PSI_NORM.TE")
    data = hdadata.el_temp.values
    n.putData(t.tdiCompile(build_str, data, "eV"))

    n = t.getNode(prefix + "PROFILES.PSI_NORM.TI")
    data = hdadata.ion_temp.sel(element=hdadata.main_ion).values
    n.putData(t.tdiCompile(build_str, data, "eV"))

    n = t.getNode(prefix + "PROFILES.PSI_NORM.VOLUME")
    data = hdadata.volume.values
    n.putData(t.tdiCompile(build_str, data, "m^3"))

    n = t.getNode(prefix + "PROFILES.PSI_NORM.ZEFF")
    data = hdadata.zeff.sum("element").values
    n.putData(t.tdiCompile(build_str, data, ""))

    t.close()

    print("HDA: Finished Writing data to MDSplus")

    return


# Creates a dictionary of all keys and values in the NeXus tree
def iterate_children(node, nodeDict={}):
    """ iterate over the children of a neXus node """
    if node.type() == dec.DNeXusNode.GROUP:
        for kid in node.children():
            nodeDict = iterate_children(kid, nodeDict)
    else:
        nodeDict[node.path()] = node.value()
    return nodeDict

