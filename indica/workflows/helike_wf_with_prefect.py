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

from prefect import task, flow, runtime
from teml import Data, Models, Visualisations

def set_model(machine,instrument,diag):
    machine = "st40"
    instrument = "xrcs"
    model = diag
    transforms = load_default_objects(machine, "geometry")
    equilibrium = load_default_objects(machine, "equilibrium")
    plasma = load_default_objects(machine, "plasma")

    plasma.set_equilibrium(equilibrium)
    transform = transforms[instrument]
    transform.set_equilibrium(equilibrium)
    model = model(instrument)
    model.set_transform(transform)
    model.set_plasma(plasma)
    return model


@task
def ingest_data():
    model=set_model(machine="st40",instrument="xrcs",diag=HelikeSpectrometer)
    #return Data.Ingestion.online_ingestion.fetch_new_data(model,num_batches=50,observed_var="[\"raw_spectra\"].data")
    return Data.Ingestion.online_ingestion.fetch_new_data(model)

@task
def ingest_larger_dataset(n=30):
    print("Flow run params:",runtime.flow_run.parameters)
    model=set_model(machine="st40",instrument="xrcs",diag=HelikeSpectrometer)

    all_X = []
    all_Y = []
    for i in range(n):
        raw_data =Data.Ingestion.online_ingestion.fetch_new_data(model) 
        X = raw_data["features"]
        Y = raw_data["outputs"]
        all_X.append(X)
        all_Y.append(Y)
        print("Collected "+str(i+1)+ " sets of samples")
    
    all_X=np.vstack(all_X)
    all_Y=np.vstack(all_Y)
    data={}
    data["features"]=all_X
    data["outputs"]=all_Y
    return data

@task
def normalise_data(raw_data):
    return Data.Treatment.normalise.normalize_data_min_max(raw_data)

@task
def reduce_output(output_data,components=20, method="reduce_dimensionality_pca"):
    #return Data.Treatment.reduce.select_best_technique_and_components(output_data)
    reduce_method=getattr(Data.Treatment.reduce,method,None)
    return reduce_method(output_data,components)

@task
def rolling_average_output(output_data,window=2):
    return Data.Treatment.reduce.rolling_average_reconstruction(output_data,window)

@task
def savgol_filter_output(output_data,window=5,polyorder=2):
    return Data.Treatment.reduce.savitzky_golay_reconstruction(output_data,window,polyorder)



@task
def setup_a_model(input_dim=3,output_dim=1030):
    ml_model=Models.Train.train_models.build_mlp_cnn_model(input_dim,output_dim)
    print(ml_model.summary())
    return ml_model

@task   
def train_a_model(X,Y,model):
    hist=Models.Train.train_models.train_model(model,X,Y)
    model.save('/home/jussi.hakosalo/Indica/indica/ml_models/my_model.keras')
    return hist

@task
def train_plots(history):
    return Visualisations.visualisebasic.plot_learning_curve(history)


@task
def example_prediction_plot(reconstr,actual):
    return Visualisations.visualisebasic.plot_reconstructed_vs_actual(reconstr,actual)

"""

@task
def load_model():
    return model_handling.load_or_initialize_model()

@task
def train_model(model, processed_data):
    updated_model = training.incremental_train(model, processed_data)
    return updated_model

@task
def evaluate_model(model, data):
    return evaluation.evaluate_performance(model, data)

@task
def deploy_model(model):
    model_handling.deploy_model(model)
"""


def predict_single_point(model, X_new, scaler_X, scaler_Y, inverse_Y):
    X_new_normalized = scaler_X.transform([X_new])
 
    Y_pred_reduced = model.predict(X_new_normalized)
 
    Y_pred_normalized = inverse_Y.inverse_transform(Y_pred_reduced)
 
    Y_pred = scaler_Y.inverse_transform(Y_pred_normalized)
 
    return Y_pred


@flow
def mainflow(log_prints=True):
    

    run_identifier=runtime.flow_run.name+str(runtime.flow_run.id)[:5]
    print(run_identifier)
    
        
    data=ingest_larger_dataset(n=4)

    X, norm_scalerX = normalise_data(data["features"])

    Y, norm_scalerY=normalise_data(data["outputs"])


    Y,inverse,pca=reduce_output(Y,components=20)

    print(Y.shape)
    print(X.shape)

    model=setup_a_model(4,20)
    hist=train_a_model(X,Y,model)

    fg,ax=train_plots(hist)
    fg.savefig(run_identifier+"_learning.png")


    
    #Other plot: get a random data point from the set we used
    rn=random.randint(0,len(data["features"]-1))
    X_ref=data["features"][rn]
    Y_ref=data["outputs"][rn]
    Y_newpred=predict_single_point(model,X_ref,norm_scalerX,norm_scalerY,pca).flatten()
    print(X_ref.shape,Y_ref.shape,Y_newpred.shape)
    fg2,ax2=example_prediction_plot(Y_newpred,Y_ref)
    fg2.savefig(run_identifier+"_prediction.png")









# Schedule the workflow to run continuously or at regular intervals
mainflow()