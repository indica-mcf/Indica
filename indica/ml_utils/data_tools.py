import numpy as np
import config_tools as cft
from MDSplus import Connection
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed

#from st40_phys_viewer.utility.mdsCheck import mdsCheck
#from st40_phys_viewer.utility.MDSplus_IP_address import MDSplus_IP_address


MDSplus_IP_address = '192.168.1.7:8000'


def data_range(tmin: float, tmax: float, time_array: np.ndarray, data: np.ndarray) -> (np.ndarray, np.ndarray):
    
    mask = (time_array >= tmin) & (time_array <= tmax)
    time = time_array[mask]
    data_new = data[mask]
    return time, data_new

def match_data_in_time(ts_time: np.ndarray, sxr_time: np.ndarray, sxr_data: np.ndarray) -> (np.ndarray, np.ndarray):

    array1 = sxr_time
    array_sxr = sxr_data
    array2 = ts_time

    # Find the indices of array1 that match the values in array2
    indices_float = np.interp(array2, array1, np.arange(len(array1)))
    indices = np.asarray(np.round(indices_float), dtype=int)
    
    new_sxr_time = array1[indices]
    new_sxr_data = array_sxr[indices]

    return new_sxr_time, new_sxr_data

def lowpass_filter(t, y, f_cutoff=10e3):
    """
    """
    uRe = np.fft.rfft(y - np.mean(y))
    freqRe = np.fft.rfftfreq(t.shape[-1], np.gradient(t)[0])
    # freq bins up to 1 / (2 * dt)

    # filter
    cut = np.ones_like(uRe)
    cut[freqRe > f_cutoff] = 0.0
    uRe_cut = cut * uRe
    y_cut = np.fft.irfft(uRe_cut)

    return y_cut

def remove_dc_offset(t, y, t_start=-0.35):
	ids = (t <= t_start)
	offset = np.median(y[ids])
	return y - offset

def sxrc_reader(pulseNo: int, MDSplus_node: str, ) -> (np.ndarray):
	"""
	Minimum working example.

	Given pulseNo and full path to MDSplus node, function reads in the 
	SXRC data.

	Returns np.array of shape (N_t, 2)
	Where the first column is timestamp, and the second column is the data.

	This data has been frequency filtered to below 4 kHz
	This data has been corrected for dc offset (so GND = 0.0)
	"""
    	# Setup
	# -----
	pulseNo = int(pulseNo)
	MDSplus_node = str(MDSplus_node)

	# defaulted to if anything fails
	output = np.array([np.nan, np.nan])

	# Copy workflow from MDSplus_tools.get_MDSplus_node_data here
	tree = MDSplus_node.split('::')[0]  # get first entry in address
	tree = tree[1:]  # remove leading \\
	# Connect to MDSPlus
	conn = Connection(MDSplus_IP_address)
	conn.openTree(tree, pulseNo)
	data = conn.get(MDSplus_node).data()
	t = conn.get('dim_of(' + MDSplus_node + ')')
	#conn.disconnect()
	# freq filter data
	f_cutoff = 4e3 # kHz - cutoff freq for lowpass filter
	data = lowpass_filter(t, data, f_cutoff=f_cutoff)

	# remove dc offset
	data = remove_dc_offset(t, data, t_start=-0.35)

	# get into correct format
	# desired output shape is (N_t, 2)
	output = np.array([t, data])
	output = np.transpose(output)

	return output

def spd_reader(pulseNo: int, MDSplus_node: str) -> (np.ndarray):

    pulseNo = int(pulseNo)
    MDSplus_node = str(MDSplus_node)

    # defaulted to if anything fails
    output = np.array([np.nan, np.nan])

    # Copy workflow from MDSplus_tools.get_MDSplus_node_data here
    tree = MDSplus_node.split('::')[0]  # get first entry in address
    tree = tree[1:]  # remove leading \\

    # Connect to MDSPlus
    conn = Connection(MDSplus_IP_address)
    conn.openTree(tree, pulseNo)
    data = conn.get(MDSplus_node).data()
    t = conn.get('dim_of(' + MDSplus_node + ')')
    #conn.disconnect()
    output = np.array([t, data])
    output = np.transpose(output)
    
    return output

def ts_reader(pulseNo: int, MDSplus_node: str) -> (np.ndarray):
    pulseNo = int(pulseNo)
    MDSplus_node = str(MDSplus_node)

    # defaulted to if anything fails
    output = np.array([np.nan, np.nan])

    # Copy workflow from MDSplus_tools.get_MDSplus_node_data here
    tree = MDSplus_node.split('::')[0]  # get first entry in address
    tree = tree[1:]  # remove leading \\
    # Connect to MDSPlus
    erebor_ip = "192.168.1.21"
    conn = Connection(MDSplus_IP_address)
    conn.openTree(tree, pulseNo)
    data = conn.get(MDSplus_node).data()
    time = conn.get('\ST40::TOP.TS.BEST:TIME').data()
    ts_r = conn.get('\ST40::TOP.TS.BEST:R').data()
    output = np.array(data)
    output = np.transpose(output)
    
    return time, ts_r, output

def sxrc_xy_reader(pulseNo: int, MDSplus_node: str, ) -> (np.ndarray):
    """
    Minimum working example.

    Given pulseNo and full path to MDSplus node, function reads in the 
    SXRC data.

    Returns np.array of shape (N_t, 2)
    Where the first column is timestamp, and the second column is the data.

    This data has been frequency filtered to below 4 kHz
    This data has been corrected for dc offset (so GND = 0.0)
    """
    # Setup
    # -----
    pulseNo = int(pulseNo)
    MDSplus_node = str(MDSplus_node)

    # defaulted to if anything fails
    output = np.array([np.nan, np.nan])

    # Copy workflow from MDSplus_tools.get_MDSplus_node_data here
    tree = MDSplus_node.split('::')[0]  # get first entry in address
    tree = tree[1:]  # remove leading \\

    # Connect to MDSPlus
    conn = Connection(MDSplus_IP_address)
    print(tree, pulseNo)
    conn.openTree(tree, pulseNo)
    data = conn.get(MDSplus_node).data()
    t = conn.get('\ST40::TOP.SXRC_XY2.BEST:TIME_BIN').data()
    sxr_r = conn.get('\ST40::TOP.SXRC_XY2.BEST:R').data()
    output = np.array(data)
    output = np.transpose(output)

    return t, sxr_r, output

def ece_reader(pulseNo: int, MDSplus_node: str) -> (np.ndarray):
    
    pulseNo = int(pulseNo)
    MDSplus_node = str(MDSplus_node)

    # defaulted to if anything fails
    output = np.array([np.nan, np.nan])

    # Copy workflow from MDSplus_tools.get_MDSplus_node_data here
    tree = MDSplus_node.split('::')[0]  # get first entry in address
    tree = tree[1:]  # remove leading \\

    # Connect to MDSPlus
    conn = Connection(MDSplus_IP_address)
    conn.openTree(tree, pulseNo)
    
    output = []
    R = []
    
    n_los = 7
    
    for los in range(n_los):
        
        MDSplus_node = f'\ST40::TOP.ECE.BEST.GLOBAL:TE_{los+1}'
        MDSplus_node_R = f'\ST40::TOP.ECE.BEST.GLOBAL:R_2CYC_{los+1}'
        
        data = conn.get(MDSplus_node).data()
        r_data = conn.get(MDSplus_node_R).data()

        output.append(data)
        R.append(r_data)
    
    t = conn.get('\ST40::TOP.ECE.BEST:TIME').data()
    
    output = np.array(output)
    output = np.transpose(output)
    
    R = np.array(R)
    R = np.transpose(R)
    
    return t, R, output

def smm_reader(pulseNo: int, MDSplus_node: str) -> (np.ndarray):

    pulseNo = int(pulseNo)
    MDSplus_node = str(MDSplus_node)

    # defaulted to if anything fails
    output = np.array([np.nan, np.nan])

    # Copy workflow from MDSplus_tools.get_MDSplus_node_data here
    tree = MDSplus_node.split('::')[0]  # get first entry in address
    tree = tree[1:]  # remove leading \\

    # Connect to MDSPlus
    conn = Connection(MDSplus_IP_address)
    conn.openTree(tree, pulseNo)

    return t, smm_data

#################################################################
#          Contstruct input/output data dictionaries
#################################################################

def read_data(index_1, diagnostic_name, index_2, pulse_no, data_read_function, mdsplus_path):
    # This function will be executed in a separate process
    return (f"{diagnostic_name}_{index_1}", f"pulse_{pulse_no}", input_function_dict[data_read_function](pulse_no, mdsplus_path))

def data_constructor(nn_layer, plasma_info, input_function_dict):

    data = {}

    with ProcessPoolExecutor() as executor:
        futures = []

        for index_1, row in nn_layer.iterrows():
            data_read_function = row['Data reader']
            diagnostic_name = row['Diagnostic name']
            mdsplus_path = row['MDS+ path']
            data[f"{diagnostic_name}_{index_1}"] = {}

            for index_2, row in plasma_info.iterrows():
                pulse_no = int(row['Pulse no.'])

                # Submit the task to the process pool
                future = executor.submit(read_data, index_1, diagnostic_name, index_2, pulse_no, data_read_function, mdsplus_path)
                futures.append(future)

        for future in as_completed(futures):
            # As the processes complete, retrieve the result and update the dictionary
            diag, pulse, result = future.result()
            data[diag][pulse] = result

    return data
     

"""
def data_constructor(nn_layer, plasma_info, input_function_dict):
    
    data = {}
        
    for index_1, row in nn_layer.iterrows():
        data_read_function = row['Data reader']
        diagnostic_name = row['Diagnostic name']
        mdsplus_path = row['MDS+ path']
        data[f"{diagnostic_name}_{index_1}"] = {}

        for index_2, row in plasma_info.iterrows():
            pulse_no = int(row['Pulse no.'])

            # Assuming functions are stored in a dictionary
            data[f"{diagnostic_name}_{index_1}"][f"pulse_{pulse_no}"] = input_function_dict[data_read_function](pulse_no, mdsplus_path)
            
    return data
"""

def create_data_nodes_dict():

    settings_file_path = '../settings.ini'
    settings = cft.read_settings(settings_file_path)
    
    data_nodes_dict = {}
    data_nodes_dict['SXR'] = settings['MDSplus']['input_node']
    data_nodes_dict['TS_ne'] = settings['MDSplus']['output_node_ne']
    data_nodes_dict['TS_Te'] = settings['MDSplus']['output_node_te']
    data_nodes_dict['TS_ne_err'] = settings['MDSplus']['output_node_ne_err']
    data_nodes_dict['TS_Te_err'] = settings['MDSplus']['output_node_te_err']
    data_nodes_dict['ECE'] = settings['MDSplus']['output_node_ece']
    print('Directories exist and data relevant items retrieved!')

    data_nodes_dict['density_norm'] = float(settings['Normalisation']['density_norm']) # normalise density down to 0 - 10 range
    data_nodes_dict['temp_norm'] = float(settings['Normalisation']['temperature_norm'])      # normalise temperature down to 0 - 10 keV range
    data_nodes_dict['sxr_norm'] = float(settings['Normalisation']['sxr_norm'])       # normalise sxr down to 0 - 1 range

    print('Normalisations exist and retrieved!')


    return data_nodes_dict

def get_input_data(pulseNo):

    data_nodes_dict = create_data_nodes_dict()

    try:
        time_sxr, r_sxr, data_sxr = sxrc_xy_reader(pulseNo, MDSplus_node= data_nodes_dict['SXR'])
        sxr_data_exists = True
    except:
        print('no SXR data!')
        sxr_data_exists = False

    try:
        _, _, y_te = ts_reader(pulseNo, MDSplus_node = data_nodes_dict['TS_Te']) # load TS te data
        nan_indices = np.where(np.isnan(y_te))
        y_te = np.array(y_te)
        y_te[nan_indices] = 0

        _, _, y_te_err = ts_reader(pulseNo, MDSplus_node = data_nodes_dict['TS_Te_err']) # load TS te data
        nan_indices = np.where(np.isnan(y_te_err))
        y_te_err = np.array(y_te_err)
        y_te_err[nan_indices] = 0
    
        time_array, _, y_ne = ts_reader(pulseNo, MDSplus_node = data_nodes_dict['TS_ne']) # load TS ne data
        nan_indices = np.where(np.isnan(y_ne))
        y_ne_err = np.array(y_ne)
        y_ne[nan_indices] = 0
    
        _, r_ts, y_ne_err = ts_reader(pulseNo, MDSplus_node = data_nodes_dict['TS_ne_err']) # load TS te data
        nan_indices = np.where(np.isnan(y_ne_err))
        y_ne_err = np.array(y_ne_err)
        y_ne_err[nan_indices] = 0

        ts_data_exists = True
    
    except:
        print('no TS data!')
        ts_data_exists = False
    
    
    try:
        ece_time, ece_R, ece_output = ece_reader(pulseNo, MDSplus_node = data_nodes_dict['ECE'])
        ece_data_exists = True
    except:
        print('no ECE data!')
        ece_data_exists = False

    data_dict = {}

    if ts_data_exists:
        y_ne = y_ne / data_nodes_dict['density_norm']
        y_te = y_te / data_nodes_dict['temp_norm']
        y_te_err = y_te_err / data_nodes_dict['temp_norm']
        y_ne_err = y_ne_err / data_nodes_dict['density_norm']

        data_dict['y_ne'] = y_ne
        data_dict['y_ne_err'] = y_ne_err
        data_dict['y_te'] = y_te
        data_dict['y_te_err'] = y_te_err
        data_dict['ts_time'] = time_array
        data_dict['ts_R'] = r_ts


    if sxr_data_exists:
        refined_data = np.asarray(data_sxr/data_nodes_dict['sxr_norm'])
        refined_data = refined_data.transpose()
        
        data_dict['sxr_data'] = refined_data
        data_dict['sxr_time'] = time_sxr
        data_dict['sxr_R'] = r_sxr
    
    if ece_data_exists:
        data_dict['ece_time'] = ece_time
        data_dict['ece_signal'] = ece_output
        data_dict['ece_R'] = ece_R

    return data_dict


# Define the model function
def model_function(x: np.ndarray, a: float, c: float) -> np.ndarray:
    """Model function y = a*sqrt(x) + c.

    Args:
        x (np.ndarray): The x data.
        a (float): Parameter a of the model.
        c (float): Parameter c of the model.

    Returns:
        np.ndarray: The computed y values based on the model.
    """
    return a * np.sqrt(x) + c

def fit_and_plot(x_data: np.ndarray, y_data: np.ndarray):
    """Fits the model to the data, extracts parameters, and plots the original data with the fitted function.

    Args:
        x_data (np.ndarray): The x data points.
        y_data (np.ndarray): The y data points.
    """
    # Use curve_fit to find the best parameters a and c for the model function
    params, _ = curve_fit(model_function, x_data, y_data)

    # Extract fitted parameters
    a_fitted, c_fitted = params
    print(f"Fitted Parameters: a = {a_fitted}, c = {c_fitted}")

    # Generate a smooth line for the fitted function
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = model_function(x_fit, a_fitted, c_fitted)

    return x_fit, y_fit, a_fitted, c_fitted
