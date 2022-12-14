from MDSplus import Float32
from MDSplus import String
from numpy import NaN

# Module for MDSplus helper functions
# Otto Asunta -- 02/03/2018


def getOrCreateNode(t, nodeName, nodeType, nodeHelp, **kwargs):
    """
    ################################################################
    Function gets the node or creates new one if it does not exist.
    ################################################################"""
    try:
        return t.getNode(nodeName)
    except ValueError:
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
