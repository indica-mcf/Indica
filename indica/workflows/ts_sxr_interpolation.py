from typing import Callable

import matplotlib.pylab as plt
import time
import numpy as np
import random
from indica.defaults.load_defaults import load_default_objects
from indica.models import Bolometer
from indica.models import BremsstrahlungDiode
from indica.models import ChargeExchange
from indica.models import EquilibriumReconstruction
from indica.models import HelikeSpectrometer

from sklearn.preprocessing import MinMaxScaler

from prefect import task, flow, runtime
from teml import Data, Models, Visualisations

@task
def ingest_data():
    data_nodes=['SXR']
    ##Shots as interval or one
    
    #shots=list(range(11415,11430))
    #shots=list(range(12190,12200))
    #THe campaign:P2.3C1 started with shot 10131,P2.3C1 ended on shot 11582
    shots=list(range(10131,11582))
    shots=[11389,11415,11416,11417,11418,11419,11420,11421,11422,11424,11425,11426,11468,11469,11472,11522,11523
,11525,11540,11560]
    return Data.Ingestion.offline_ingestion.fetch_data(data_nodes,shots)

@task
def normalize(sensor1_data,sensor1_timestamps,sensor2_data,sensor2_timestamps):

    sensor1_data = np.nan_to_num(sensor1_data, nan=0)
    sensor2_data = np.nan_to_num(sensor2_data, nan=0)

    # Normalize sensor 1 timestamps
    sensor1_timestamps = (sensor1_timestamps - np.min(sensor1_timestamps)) / \
                                    (np.max(sensor1_timestamps) - np.min(sensor1_timestamps))

    # Normalize sensor 2 timestamps
    sensor2_timestamps = (sensor2_timestamps - np.min(sensor2_timestamps)) / \
                                    (np.max(sensor2_timestamps) - np.min(sensor2_timestamps))

    # Normalize sensor 1 and sensor 2 values
    sensor1_data = (sensor1_data - np.min(sensor1_data)) / (np.max(sensor1_data) - np.min(sensor1_data))
    sensor2_data = (sensor2_data - np.min(sensor2_data)) / (np.max(sensor2_data) - np.min(sensor2_data))

    return sensor1_data, sensor1_timestamps, sensor2_data, sensor2_timestamps


@task
def trim_times(sensor2_data, sensor1_timestamps, sensor2_timestamps):
    """
    Align sensor 2 timestamps to the range of sensor 1 timestamps and remove invalid intervals.
    
    Args:
        sensor2_data (np.array): Sensor 2 data (1D or 2D across pulses).
        sensor1_timestamps (np.array): Sensor 1 timestamps across pulses.       
        sensor2_timestamps (np.array): Sensor 2 timestamps across pulses.
    
    Returns:
        trimmed_sensor2_data (np.array): Trimmed and cleaned sensor 2 data.
        trimmed_sensor2_timestamps (np.array): Corresponding trimmed timestamps for sensor 2.
    """
    trimmed_sensor2_data = []
    trimmed_sensor2_timestamps = []
    valid_count=0
    sensor2_data = np.array(sensor2_data, dtype=np.float64)

    for pulse_idx in range(len(sensor1_timestamps)):
        time_s1 = sensor1_timestamps[pulse_idx]
        time_s2 = sensor2_timestamps[pulse_idx]
        data_s2 = sensor2_data[pulse_idx]

        # Ensure 2D structure for slicing if needed
        if data_s2.ndim == 1:
            data_s2 = data_s2[np.newaxis, :]  # Add a channel dimension for consistency


        # Find overlap range
        start_idx = np.searchsorted(time_s2, time_s1[0])
        end_idx = np.searchsorted(time_s2, time_s1[-1], side='right')



        if end_idx <= start_idx:
            print(f"Pulse {pulse_idx} removed due to no overlap between TS and SXR timestamps.")
            continue

        trimmed_time_s2 = time_s2[start_idx:end_idx]
        trimmed_data_s2 = data_s2[:, start_idx:end_idx]  # Safe slicing for 2D



        # Filter intervals with NaNs at either endpoint
        valid_intervals = []
        valid_time_intervals = []

        for i in range(trimmed_data_s2.shape[1] - 1):  # Iterate through intervals
            if not (np.isnan(trimmed_data_s2[:, i]).any() or np.isnan(trimmed_data_s2[:, i + 1]).any()):
                valid_intervals.append(trimmed_data_s2[:, i:i + 2])  # Keep valid interval
                valid_time_intervals.append(trimmed_time_s2[i:i + 2])
            else:
                pass



        
        # Concatenate valid intervals into a single array
        valid_intervals = np.concatenate(valid_intervals, axis=1) if len(valid_intervals) > 1 else valid_intervals[0]
        valid_time_intervals = np.concatenate(valid_time_intervals) if len(valid_time_intervals) > 1 else valid_time_intervals[0]

        # Ensure flattened or consistent shapes for appending
        valid_intervals = valid_intervals.squeeze()  # Remove extra dimensions
        valid_time_intervals = valid_time_intervals.flatten()  # Ensure 1D for time

        trimmed_sensor2_data.append(valid_intervals)
        trimmed_sensor2_timestamps.append(valid_time_intervals)
        valid_count+=len(valid_time_intervals)
    print(f"Total valid intervals across all pulses: {valid_count}")
    return np.array(trimmed_sensor2_data, dtype=object), np.array(trimmed_sensor2_timestamps, dtype=object)








@task 
def create_compute_loss_TESXR(input_sensor2_features, scaling_params, lambda_smooth=0.1, lambda_similarity=0.1):
    def compute_loss_TESXR(y_true, y_pred):
        """
        Custom loss function for TS-SXR interpolation.

        Args:
            y_true: True values (ground truth).
            y_pred: Predicted values.

        Returns:
            float: Total loss.
        """
        import tensorflow as tf

        # Convert inputs to TensorFlow tensors for compatibility
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        # Endpoint Loss (MSE)
        endpoint_mask = tf.math.logical_not(tf.math.is_nan(y_true))
        endpoint_loss = tf.reduce_mean(tf.square(tf.boolean_mask(y_pred, endpoint_mask) - tf.boolean_mask(y_true, endpoint_mask)))

        # Smoothness Penalty
        smoothness_penalty = tf.reduce_mean(tf.square(y_pred[1:] - y_pred[:-1]))

        # Sensor 2 Similarity Penalty
        interval_slope = input_sensor2_features[:, 2]  # Assume slope is the third feature

        # Extract start and end of scaling params
        if len(scaling_params.shape) > 1:  # If batched
            scaling_start = scaling_params[:, 0]  # Batch-aware start
            scaling_end = scaling_params[:, -1]  # Batch-aware end
        else:
            scaling_start = scaling_params[0]
            scaling_end = scaling_params[-1]

        scaling_diff = tf.cast(scaling_end - scaling_start, tf.float32)  # Ensure dtype matches y_pred
        scaling_diff = tf.maximum(scaling_diff, 1e-8)  # Avoid division by zero

        predicted_slope = (y_pred[-1] - y_pred[0]) / scaling_diff

        similarity_penalty = tf.reduce_mean(tf.square(predicted_slope - interval_slope))

        # Total Loss
        total_loss = endpoint_loss + lambda_smooth * smoothness_penalty + lambda_similarity * similarity_penalty
        return total_loss

    return compute_loss_TESXR




@task
def trim_times_single_pulse(combined_sxr,combined_time_sxr,combined_ts,combined_time_ts):

    """
    Construct intervals from combined time ts st. each interval is time between each TS observation.

    Then for each interval:
        -Go through sxr times and combined. If timestamp is greater than first element of interval or less than second element, add to that
        intervals collection
        <
        >

    """
    intervals_sxr_data_bins=[]
    intervals_sxr_time_bins=[]



    for i in range(len(combined_time_ts)-1):
        interval_sxr_times_bin=[]
        interval_sxr_datas_bin=[]
        #print(f"Interval start and end: {combined_time_ts[i]}--{combined_time_ts[i+1]}")

        for timestamp, datapoint in zip(combined_time_sxr,combined_sxr):
            if (timestamp>=combined_time_ts[i] and timestamp<=combined_time_ts[i+1]):
                #print(f"timestamp {timestamp} with data point {datapoint} belongs in the interval.")
                interval_sxr_datas_bin.append(datapoint)
                interval_sxr_times_bin.append(timestamp)
        intervals_sxr_data_bins.append(interval_sxr_datas_bin)
        intervals_sxr_time_bins.append(interval_sxr_times_bin)

    return intervals_sxr_data_bins,intervals_sxr_time_bins




@task
def extract_single_pulse_model_data(sxr_interval_datas,sxr_interval_times,combined_ts,combined_time_ts):
    """

    Args:
        sxr_interval_datas (_type_): List of lists for interval SXR datapoints
        sxr_interval_times (_type_): List of lists for interval SXR timepoints
        combined_ts (_type_): A flat list of TS datapoints
        combined_time_ts (_type_): A flat list of TS timepoints
    """


    model_inputs=[]
    model_targets=[]
    for i in range(len(combined_time_ts)-1):
        #For each TS interval
        ts_interval_start=combined_time_ts[i]
        ts_interval_end=combined_time_ts[i+1]

        sxr_data_for_this_interval=sxr_interval_datas[i]
        sxr_time_for_this_interval=sxr_interval_times[i]

        interval_mean=np.nanmean(sxr_data_for_this_interval)
        interval_variance=np.nanvar(sxr_data_for_this_interval)
        interval_slope= (sxr_data_for_this_interval[-1]-sxr_data_for_this_interval[0])/(sxr_time_for_this_interval[-1]-sxr_time_for_this_interval[0])
        
        #extract a sample from the SXR data
        sxr_resampled=np.interp(np.linspace(0,len(sxr_data_for_this_interval)-1,20),np.arange(len(sxr_data_for_this_interval)),sxr_data_for_this_interval)

 

        for data,time in zip(sxr_data_for_this_interval,sxr_time_for_this_interval):
            scaling_param= (time - ts_interval_start) / (ts_interval_end - ts_interval_start)
            start_value=combined_ts[i]
            end_value=combined_ts[i+1]


            if np.isnan(combined_ts[i]) or np.isnan(combined_ts[i+1]):
                pass
            else:
            #print(scaling_param,start_value,end_value,interval_mean,interval_variance)
                add_list=[scaling_param,start_value,end_value,interval_mean,interval_variance,interval_slope]
                add_list.extend(sxr_resampled)

                #model_inputs.append([scaling_param,start_value,end_value,interval_mean,interval_variance,interval_slope])
                model_inputs.append(add_list)


                target=combined_ts[i]*(1-scaling_param)+combined_ts[i+1]*scaling_param
                model_targets.append(target)

            #print(target)
    return model_inputs,model_targets


            






@flow
def tssxrflow(log_prints=True):
    


    # Initialize storage for inputs and targets
    model_inputs = []
    model_targets = []



    run_identifier=runtime.flow_run.name+str(runtime.flow_run.id)[:5]
    print(run_identifier)
    # Ingest data
    combined_sxr, combined_time_sxr, combined_ts, combined_time_ts = ingest_data()


    print("NaN counts after data ingestion:")
    print("Combined SXR Data NaNs:", np.isnan(combined_sxr).sum())
    print("Combined Time SXR NaNs:", np.isnan(combined_time_sxr).sum())
    print("Combined TS Data NaNs:", np.isnan(combined_ts).sum())
    print("Combined Time TS NaNs:", np.isnan(combined_time_ts).sum())

    points_per_interval=0
    for i in range(len(combined_sxr)):


        sxr_interval_datas,sxr_interval_times=trim_times_single_pulse(combined_sxr[i],combined_time_sxr[i],combined_ts[i],combined_time_ts[i])
        print("points per interval: ",len(sxr_interval_datas[0]))
        points_per_interval=len(sxr_interval_datas[0])







        #Model input is: scaling param, start ts of interval, end ts of interval, slope, mean, variance.

        new_inputs,new_targets=extract_single_pulse_model_data(sxr_interval_datas,sxr_interval_times,combined_ts[i],combined_time_ts[i])
        model_inputs.extend(new_inputs)
        model_targets.extend(new_targets)



    print("Model target and input count: ",len(model_inputs))

    ts_scaling_factor=1e19

    # Convert lists to NumPy arrays
    model_inputs = np.array(model_inputs)
    model_targets = np.array(model_targets)



    # Extract individual input features
    input_t = model_inputs[:, 0].reshape(-1, 1)



    input_obs1_obs2 = model_inputs[:, 1:3]
    TS_min=np.min(input_obs1_obs2)
    TS_max=np.max(input_obs1_obs2)
    input_obs1_obs2=(input_obs1_obs2-TS_min)/(TS_max-TS_min)


    input_sensor2_features = model_inputs[:, 3:6]
    scaler_sxr=MinMaxScaler()
    input_sensor2_features=scaler_sxr.fit_transform(input_sensor2_features)


    #Last input sxr:min max scale
    input_sxr_sample=model_inputs[:,6:]
    scaler_sxr_sample=MinMaxScaler()
    input_sxr_sample=scaler_sxr_sample.fit_transform(input_sxr_sample)







    # Clean targets by replacing NaNs
    model_targets_cleaned = np.nan_to_num(model_targets, nan=0)
    model_targets_cleaned=(model_targets_cleaned-TS_min)/(TS_max-TS_min)



#Currently not used actually
    custom_loss = create_compute_loss_TESXR(
        input_sensor2_features=input_sensor2_features,  # Sensor 2 features
        scaling_params=input_t,                        # Normalized t values for each point
        lambda_smooth=0.1,
        lambda_similarity=0.1
    )



        # Train the model
    model = Models.Train.train_models.build_TS_SXR_Interpolation_model()


    history = model.fit(
        [input_t, input_obs1_obs2,input_sxr_sample],  # Inputs
        model_targets_cleaned,  # Targets   
        batch_size=16,
        epochs=20,
        verbose=1
    )


    # Evaluate on test data
    test_loss, test_mae, test_mse = model.evaluate([input_t, input_obs1_obs2,input_sxr_sample], model_targets_cleaned, verbose=1)

    print(f"Test Loss: {test_loss}")
    print(f"Test MAE: {test_mae}")



    #plot
    interval_idx=6
    for interval_idx in [5,10,15]:

        start_idx=interval_idx*points_per_interval
        end_idx=start_idx+points_per_interval


        t_vals=input_t[start_idx:end_idx].flatten()

        obs_1_obs2_vals=input_obs1_obs2[start_idx:end_idx]
        sxr_features=input_sensor2_features[start_idx:end_idx]

        sorted_indices=np.argsort(t_vals)
        t_vals_sorted=t_vals[sorted_indices]

        obs1_obs2_sorted=obs_1_obs2_vals[sorted_indices]

        sxr_features_sorted=sxr_features[sorted_indices]


        #Sample the used SXR sample to a higher resolution again
        sxr_interval_sample=input_sxr_sample[start_idx]
        upsampled_sxr_interval_sample=np.linspace(0,1,len(sxr_interval_sample))
        target_x=t_vals_sorted
        upsampled_sxr_interval_sample=np.interp(target_x,sxr_interval_sample,input_sxr_sample[start_idx])








        predicted_ts=model.predict([t_vals_sorted.reshape(-1,1),
                                    obs1_obs2_sorted, np.tile(sxr_interval_sample,(len(t_vals_sorted),1))]).flatten()
        """
        predicted_ts=model.predict([t_vals_sorted.reshape(-1,1),
                                    obs1_obs2_sorted,
                                    
                                    sxr_features_sorted]).flatten()"
                                    """

        ts_start=input_obs1_obs2[start_idx,0]
        ts_end=input_obs1_obs2[start_idx,1]
        expected_linear=ts_start*(1-t_vals_sorted)+ts_end*t_vals_sorted


        plt.figure(figsize=(8,5))
        plt.plot(t_vals_sorted,expected_linear,label="Model Linear", linestyle="--")
        plt.plot(t_vals_sorted,predicted_ts,label="Model Prediction", linewidth=2)
        plt.plot(t_vals_sorted,upsampled_sxr_interval_sample,label="Sxr",linestyle=":")



        plt.scatter([0,1],[ts_start,ts_end],color='black',label="TS ENDPOINTS", zorder=5)
        plt.xlabel("Scaling parameter (t)")
        plt.ylabel("TS Value (scaled)")
        plt.title(f"Model prediction vs Linear TS interpolation for interval {interval_idx}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"sampledsxr_ts_interpolation_interval_{interval_idx}.png",dpi=300)
        plt.close()

    print(f"Used {len(combined_sxr)} pulses with {len(input_obs1_obs2)} intervals.")
        








# Schedule the workflow to run continuously or at regular intervals
tssxrflow()