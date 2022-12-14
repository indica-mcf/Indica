#! Module for MDSplus helper functions
#! Otto Asunta -- 02/03/2018


from MDSplus import *
from numpy import *

# import openpyxl
# import pandas as pd
# import matplotlib.pyplot as plt


def getOrCreateNode(t, nodeName, nodeType, nodeHelp, **kwargs):
    """
    ################################################################
    Function gets the node or creates new one if it does not exist.
    ################################################################"""
    try:
        return t.getNode(nodeName)
    except:
        print("Node " + nodeName + " is created.")
        return createNode(t, nodeName, nodeType, nodeHelp, **kwargs)


def createNode(t, nodeName, nodeType, nodeHelp, **kwargs):
    """################################################################
    Function createNode is used for creating a node and its
    subnode "HELP".

    Input parameters:
    t        = Tree object
    nodeName = name of the node to be added
    nodeType = type of the node to be added
    nodeHelp = help string to be written to .HELP

    Optional parameters:
    dataIn   = data to be written into the node.
               NOTE: Arrays of data need to be defined as array([]),
               i.e. numpy.ndarray, otherwise putData() will fail.
    units    = units of the data (only works if data is given too)
    ################################################################"""

    # Save the current location in the tree
    defaultNode = t.getDefault()
    #
    if "dataIn" in kwargs:
        dataIn = kwargs["dataIn"]
    else:
        dataIn = None

    if "units" in kwargs:
        units = kwargs["units"]
    else:
        units = None

    # Add and prefill the node
    n = t.addNode(nodeName.upper(), nodeType.upper())
    if nodeType.upper() == "SIGNAL":
        if dataIn is None:
            dataIn = t.tdiCompile("build_signal($ROPRAND,*,$ROPRAND)")
            n.putData(dataIn)
        else:
            if units is None:
                n.putData(dataIn)
            else:
                n.putData(dataIn.setUnits(units))

    elif nodeType.upper() == "TEXT":
        if dataIn is None:
            dataIn = " "
            n.putData(dataIn)
        else:
            if units is None:
                n.putData(dataIn)
            else:
                n.putData(String(dataIn).setUnits(units))
    #        d = String(dataIn);

    elif nodeType.upper() == "NUMERIC":
        if dataIn is None:
            dataIn = NaN
            n.putData(dataIn)
        else:
            if units is None:
                n.putData(dataIn)
            else:
                n.putData(Float32(dataIn).setUnits(units))

    #        d = Float32(dataIn);
    elif nodeType.upper() == "STRUCTURE" or nodeType.upper() == "SUBTREE":
        # No more actions needed
        pass
    else:
        #        n = t.setDefault(t.addNode(nodeName,nodeType));
        print("Unknown node type.")
        print("No data filled to node " + nodeName + ".")

    t.setDefault(n)
    #    if(all(~isnan(dataIn))): # If it is not NaN, try to write the data
    # if 'dataIn' in kwargs:
    #     if 'units' in kwargs:
    #         units = kwargs['units'];
    #     else:
    #         units = '';

    #     d.setUnits(units);
    #     print d.units_of()
    #     try:
    #         print 'dataIn=',d.data(), 'units=',d.units
    #         n.putData(d);
    #         #d.setUnits(units);
    #     except:
    #         print "createNode() Failed to write given data into ",n.getFullPath()
    #         print "Continuing regardless"

    #    print t.getDefault()
    t.addNode("HELP", "TEXT").putData(nodeHelp)
    # Return to the point in the tree from which the function was called.
    t.setDefault(defaultNode)
    return t.getNode(nodeName)


# Function for adding a list of nodes to existing trees
def addNodes(pulseNo, which_tree, nodes_to_add, quiet=True):
    """
    Inputs:
        -> pulseNo: integer. Shot number you want to add nodes to.
        -> which_tree: string. Which tree do you want to add nodes to. e.g. 'spectrom'.
        -> nodes_to_add: list. List of nodes you want to add.

    Notes:
        -> The nodes you want to add to MDSplus must be in -1 tree for this function to work.
        -> HELP nodes are added automatically.
        -> Cannot add to 'spectrom' tree if which_tree='st40'.
        -> If a node of the same name already exists - they are skipped

    Example:
        import mdsHelpers as mh
        pulseNo = 5691
        which_tree = 'st40'
        add_nodes = [
            'summary.diag_quality',
            'summary.diag_quality.princeton',
            'summary.diag_quality.princeton.ti_q',
            'summary.diag_quality.princeton.v_q',
            'summary.diag_quality.avantes',
            'summary.diag_quality.avantes.data_q',
            'summary.diag_quality.ocean',
            'summary.diag_quality.ocean.data_q',
            'summary.diag_quality.spectr_lines',
            'summary.diag_quality.spectr_lines.data_q',
            'summary.diag_quality.xrcs',
            'summary.diag_quality.xrcs.ti_q',
            'summary.diag_quality.xrcs.te_q',
        ]
        mh.addNodes(pulseNo, which_tree, add_nodes)

    """

    # Connect to -1 tree
    t1 = Tree(which_tree, -1)
    nodes = []
    for node in nodes_to_add:
        n = t1.getNode(node)
        nodes.append(n)

    # Connect to the tree
    try:
        t = Tree(which_tree, pulseNo, "edit")
    except:
        print("No " + str(which_tree) + " --> or read/write access not granted")
        t1.close()
        return

    # Extract data from dictionary 'nodes_dict'
    for n in nodes:
        if not (quiet):
            print(n)
        try:
            t.getNode(n.getPath())
            status = False
            print(n.getPath() + " already exists. Skipped.")
        except:
            status = True

        if status:
            # Get help node
            try:
                n_help = n.HELP
                help_status = True
            except:
                help_status = False
                print("No HELP node")

            # Add node
            n_temp = t.addNode(n.getPath(), n.getUsage())
            if n.getUsage() != "STRUCTURE":
                n_temp.putData(n.getData())

            # Add help node
            if help_status:
                n_temp = t.addNode(n_help.getPath(), n_help.getUsage())
                n_temp.putData(n_help.getData())

    # Write to the tree
    t.write()
    t.close()

    # Disconnect from -1 tree
    t1.close()
    return
