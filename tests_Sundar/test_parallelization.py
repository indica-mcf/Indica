#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:43:42 2021

@author: sundaresan.sridhar
"""
#MODULE TO TEST PARALLELIZATION
import time as tt
import numpy as np
import multiprocessing as mp

#EXECUTION FUNCTION DEFINITION
def execution_function(input_data,args):
    tt.sleep(2)
    return np.array([input_data])

input_data = np.arange(0,10)

# #SERIAL EXECUTION
# #RESULTS DEFINITION
# results_serial = np.nan * np.ones(len(input_data))
# #STARTING TIME
# start_time_serial = tt.time()
# for i,inp_data in enumerate(input_data):
#     results_serial[i] = execution_function(inp_data,args=0)
# #PRINTING STATUS
# print('Serial execution took '+str(tt.time()-start_time_serial)+' seconds')

# #PARALLEL EXECUTION
# pool = mp.Pool(mp.cpu_count())
# results_parallel = pool.starmap(execution_function,[(inp_data,0) for inp_data in input_data])
# pool.close()
# #STARTING TIME
# start_time_parallel = tt.time()
# #PRINTING STATUS
# print('Parallel execution took '+str(tt.time()-start_time_parallel)+' seconds')





