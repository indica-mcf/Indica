#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:51:41 2021

@author: sundaresan.sridhar
"""

#TEST BED

from MDSplus import *
from st40_phys_viewer.utility.MDSplus_IP_address import MDSplus_IP_address
import numpy as np

whichTree = 'ASTRA'
pulseNo = 25008383
whichRun = 'RUN101'

#OPENING THE ASTRA TREE
conn = Connection(MDSplus_IP_address)
conn.openTree("ASTRA",pulseNo)

#FUNCTION TO GET ALL THE NODES IN A TREE
def get_all_nodes(whichTree,pulseNo):
    #OPENING THE MDSPLUS CONNECTION
    conn = Connection(MDSplus_IP_address)
    #OPENING THE TREE
    conn.openTree(whichTree, pulseNo)
    #GETTING ALL THE NODES
    fullpath = conn.get("getnci('***','fullpath')").data().astype(str, copy=False).tolist()
    #RETURNING THE NODE INFORMATION
    return fullpath

#FUNCTION TO SELECT THE NODES
def select_nodes(data,query_text):
    #SELECTING THE NODES
    nodes_sel = []
    for sel_node in data:
        if query_text in sel_node:
            nodes_sel += [sel_node]
    #RETURNING THE SELECTED NODES
    return nodes_sel

#FUNCTION TO ELEMINATE THE NODES
def eliminate_nodes(data,query_text):
    #SELECTING THE NODES
    nodes_sel = []
    for sel_node in data:
        if query_text not in sel_node:
            nodes_sel += [sel_node]
    #RETURNING THE SELECTED NODES
    return nodes_sel

#FUNCTION TO REMOVE THE TREE NAME FROM NODES
def remove_tree_name(data,tree_name):
    #RETURN DATA DECLARATION
    return_data = []
    #SWEEP OF DATA
    for sel_data in data:
        return_data += [sel_data[len(tree_name):len(sel_data)]]
    #RETURNING THE DATA
    return return_data

#FUNCTION TO GET THE COLON INDEX
def get_index(text,delimiter=':'):
    #ASCII VALUE OF THE TEXT
    ascii_text = np.array([ord(c) for c in text])
    #INDEX OF THE COLON
    colon_index = np.where(ascii_text==ord(delimiter))[0]
    #RETURNING THE COLON INDEX
    return colon_index

#ALL NODES
all_nodes = get_all_nodes(whichTree,pulseNo)
#SELECTED NODES
sel_nodes = []
#SELECTED NODES
for sel_node in all_nodes:
    if whichRun in sel_node:
        sel_nodes += [sel_node]
        

#BRANCH DETAILS
branch_details = {
    # 'GLOBAL' : {
    #         "select_branch"     : ["GLOBAL"],
    #         "nodePosition"      : 0,
    #         },
    # 'CONSTRAINTS' : {
    #         "select_branch"     : ["CONSTRAINTS","CVALUE"],
    #         "nodePosition"      : 1,
    #         },    
    'PROFILES_ASTRA' : {
            "select_branch"     : ["PROFILES","ASTRA"],
            "nodePosition"      : 0,
            },
    'PROFILES_PSI_NORM' : {
            "select_branch"     : ["PROFILES","PSI_NORM"],
            "nodePosition"      : 0,
            },
    # 'PSI2D' : {
    #         "select_branch"     : ["PSI2D"],
    #         "nodePosition"      : 0,
    #         },
    # 'PSU' : {
    #         "select_branch"     : ["PSU"],
    #         "nodePosition"      : 2,
    #         },
    }


#QUALITIES MDS TEXT
text_qualities_mds =  ""

#RETURN DATA
all_return_data      = {}
all_return_data_help = {}

#SWEEP OF KEYS
for key in branch_details.keys():
    #TREE NAME
    tree_name = '\\' + whichTree + '::TOP.'+whichRun+'.'   
    
    #SELECTING THE NODES
    data = sel_nodes
    for select_branch in branch_details[key]['select_branch']:
        data = select_nodes(data,select_branch)
    #ELIMINATING THE HELP NODES
    data = eliminate_nodes(data,'HELP')
    
    #REMOVING THE TRE NAME       
    data = remove_tree_name(data,tree_name)
    #SELECTING THE DATA NODES
    data = select_nodes(data,':')
    
    #RETURN DATA
    return_data = {}
    return_data_help = {}
    for sel_data in data:
        #REMOVING THE EMPTY SPACES
        sel_data = sel_data.replace(" ","")
        #COLON INDEX
        colon_index = get_index(sel_data,':')
        dot_indices = get_index(sel_data,'.')
        #NAME OF THE NODE
        if len(colon_index)>0:
            if branch_details[key]['nodePosition']==0:
                nodeName = sel_data[colon_index[0]+1:len(sel_data)].lower()
            if branch_details[key]['nodePosition']==1:
                nodeName = sel_data[np.nanmax(dot_indices)+1:colon_index[0]].lower()
            if branch_details[key]['nodePosition']==2:
                nodeName = sel_data[colon_index[0]+1:len(sel_data)].lower() + '_' + sel_data[np.nanmax(dot_indices)+1:colon_index[0]].lower()
            nodePath = "."+sel_data.lower()
            return_data[nodeName] = nodePath
            help_node = nodePath.upper()
            if branch_details[key]['nodePosition']==1:
                help_node = help_node[0:get_index(help_node,':')[0]]
            if branch_details[key]['nodePosition']==2:
                help_node2 = help_node[0:get_index(help_node,':')[0]]         
            return_data_help[nodeName] =  str(conn.get(tree_name[0:len(tree_name)-1]+help_node+':HELP').data())
            if branch_details[key]['nodePosition']>0:
                return_data_help[nodeName] += ' - ' + str(conn.get(tree_name[0:len(tree_name)-1]+help_node2+':HELP').data())
            # print(help_node,help_node2)
    #INCLUDING THE TIME
    if 'times' not in return_data:
        return_data['times'] = ':time'
        return_data_help['times'] = str(conn.get(tree_name[0:len(tree_name)-1]+':TIME:HELP').data())
    #UPDATING THE RETURN DATA
    all_return_data.update(return_data)
    all_return_data_help.update(return_data_help)
    
    
#PRINTING THE DATA
#MAXIMUM LENGTH OF THE KEYS
max_len_keys   = len(max(list(all_return_data.keys()),key=len))+2
max_len_values = len(max(list(all_return_data.values()),key=len))+2
return_data_text =  "\t\t{}".format("'astra' :{ #GETTING DATA FROM ASTRA\n")

for key,value in all_return_data.items():
    return_data_text += "\t\t\t"
    return_data_text += "{:{max_len}s}".format("'"+key+"'",max_len=max_len_keys)
    return_data_text += " : "
    return_data_text += "{:{max_len}}".format("'"+value+"'",max_len=max_len_values)
    return_data_text += " ,"
    return_data_text += "\t#"+all_return_data_help[key]
    return_data_text += "\n"