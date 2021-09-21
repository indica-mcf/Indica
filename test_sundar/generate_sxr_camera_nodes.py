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

#FILTERS AND CHANNEL NUMBERS
no_filter = 4
no_channels = 20 #per filter

#NODE DATA DECLARATION
node_data = {}

#SWEEP OF FILTER
for i_filter in range(0,no_filter):
<<<<<<< HEAD
    key_channel = 'filter_'+str(i_filter+1)
    node_data[key_channel] = '.middle_head.filter_'+str(i_filter+1)+':'
    #TIME DATA
    node_data['filter_'+str(i_filter+1)+'_time'] = '.middle_head.filter_'+str(i_filter+1)+':time'
    #FILTER AVERAGE
    node_data['filter_'+str(i_filter+1)+'_average'] = '.middle_head.filter_'+str(i_filter+1)+':average'
=======
    #SWEEP OF CHANNELS
    for i_ch in range(0,no_channels):
        #CHANNEL NUMBER
        ch_no = (i_filter*no_channels)+i_ch+1
        #CHANNEL KEY
        key_channel = 'ch'+str(ch_no).zfill(3)
        node_data[key_channel] = '.middle_head.filter_'+str(i_filter+1)+':ch'+str(ch_no).zfill(3)
    #TIME DATA
    node_data['time_'+str(i_filter+1)] = '.middle_head.filter_'+str(i_filter+1)+':time'
    #FILTER AVERAGE
    node_data['average_'+str(i_filter+1)] = '.middle_head.filter_'+str(i_filter+1)+':average'
>>>>>>> 46fcc44c1bb3d6ab98bb6c88e6ff81b883245def
#GEOMETRY DATA
node_data['location'] = '.middle_head.geometry:location'
node_data['direction'] = '.middle_head.geometry:direction'

<<<<<<< HEAD
# #SWEEP OF FILTER
# for i_filter in range(0,no_filter):
#     #SWEEP OF CHANNELS
#     for i_ch in range(0,no_channels):
#         #CHANNEL NUMBER
#         ch_no = (i_filter*no_channels)+i_ch+1
#         #CHANNEL KEY
#         key_channel = 'ch'+str(ch_no).zfill(3)
#         node_data[key_channel] = '.middle_head.filter_'+str(i_filter+1)+':ch'+str(ch_no).zfill(3)
#     #TIME DATA
#     node_data['time_'+str(i_filter+1)] = '.middle_head.filter_'+str(i_filter+1)+':time'
#     #FILTER AVERAGE
#     node_data['average_'+str(i_filter+1)] = '.middle_head.filter_'+str(i_filter+1)+':average'
# #GEOMETRY DATA
# node_data['location'] = '.middle_head.geometry:location'
# node_data['direction'] = '.middle_head.geometry:direction'

=======
>>>>>>> 46fcc44c1bb3d6ab98bb6c88e6ff81b883245def


    
#PRINTING THE DATA FOR ST40 READER
#MAXIMUM LENGTH OF THE KEYS
max_len_keys   = len(max(list(node_data.keys()),key=len))+2
max_len_values = len(max(list(node_data.values()),key=len))+2
return_data_text =  "\t\t{}".format("'diode_arrays' :{ #GETTING THE DATA OF THE SXR CAMERA \n")
for key,value in node_data.items():   
    return_data_text += "\t\t\t"
    return_data_text += "{:{max_len}s}".format("'"+key+"'",max_len=max_len_keys)
    return_data_text += " : "
    return_data_text += "{:{max_len}}".format("'"+value+"'",max_len=max_len_values)
    return_data_text += " ,"
    # return_data_text += "\t#"+all_return_data_help[key]
    return_data_text += "\n"
return_data_text +=  "\t\t},\n"

#PRINTING THE DATA FOR ABSTRACT READER
data_text = "\t\t{}".format("'get_radiation' :{ #GETTING THE DATA OF THE SXR CAMERA \n")
#SWEEP OF CHANNELS
for i_ch in range(1,81): 
    data_text += "\t\t\t"
    data_text += "'ch"+str(i_ch).zfill(3)+"'"
    data_text += " : "
    data_text += "('total_radiated_power','line_integrated')"
    data_text += " ,"
#     # return_data_text += "\t#"+all_return_data_help[key]
    data_text += "\n"
data_text +=  "\t\t},\n"




































































































































