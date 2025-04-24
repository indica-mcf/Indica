from typing import Callable

import matplotlib.pylab as plt

import numpy as np

from indica.defaults.load_defaults import load_default_objects
from indica.models import Bolometer
from indica.models import BremsstrahlungDiode
from indica.models import ChargeExchange
from indica.models import EquilibriumReconstruction
from indica.models import HelikeSpectrometer



def generator(model,equilibrium,plasma):



    plasma.set_equilibrium(equilibrium)


    #base_case=[plasma.t,plasma.Te,plasma.Ne,plasma.Nh,plasma.Fz,plasma.Ti,plasma,Nimp]
    #print(base_case)

    #What is updating the self-electron_temperature.for instance? in the initialization func?
    #for all of these, get a timepoint with collection.sel(t=t)
    #plasma.t = quantities.get("t", plasma.t)9x1
    #plasma.Te = quantities.get("Te", plasma.Te) 9x41, plasma.electron_temperature
    #plasma.Ne = quantities.get("Ne", plasma.Ne) 9x41, plasma.electron_density
    #plasma.Nh = quantities.get("Nh", plasma.Nh) 9x41, plasma.neutral_density
    #plasma.Fz = quantities.get("Fz", plasma.Fz) dict (len 4, keys h c ar he). Each 9x41x2, plasma.fz elements
    #plasma.Ti = quantities.get("Ti", plasma.Ti) 9x41 plasma.ion_temperature
    #plasma.Nimp = quantities.get("Nimp", plasma.Nimp)3x9x41, plasma.impurity_density

    model.set_plasma(plasma)
    bckc1 = model(sum_beamlets=False)
    plasma.electron_temperature=plasma.electron_temperature*1.05;
    print(np.max(plasma.electron_temperature))
    print(np.min(plasma.electron_temperature))
    print(plasma.electron_temperature.shape)
    ata
    bckc2 = model(sum_beamlets=False)


    raw_spectra1=bckc1["raw_spectra"].data
    raw_spectra2=bckc2["raw_spectra"].data
    correlation = np.corrcoef(raw_spectra2.flatten(), raw_spectra1.flatten())[0, 1]
    print(correlation)


def run_example_nn(
    machine: str, instrument: str, model: Callable, plot: bool = False
):
    plasma = load_default_objects(machine, "plasma")
    transforms = load_default_objects(machine, "geometry")
    equilibrium = load_default_objects(machine, "equilibrium")

    transform = transforms[instrument]
    transform.set_equilibrium(equilibrium)
    model = model(instrument)
    model.set_transform(transform)

    generator(model, equilibrium, plasma)




def run_core():
    machine = "st40"
    plot=False
    instrument = "xrcs"
    _model = HelikeSpectrometer
    run_example_nn(machine, instrument, _model, plot=plot)
"""

#from prefect import task, Flow
#from teml import data_ingestion, preprocessing, model_handling, training, evaluation

@task
def ingest_data():
    return data_ingestion.fetch_new_data(model,num_pairs=1000)

@task
def preprocess_data(raw_data):
    return preprocessing.clean_and_transform(raw_data)

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

with Flow("Online Training Workflow") as flow:
    raw_data = ingest_data()
    processed_data = preprocess_data(raw_data)
    model = load_model()
    trained_model = train_model(model, processed_data)
    performance = evaluate_model(trained_model, processed_data)
    deploy_model(trained_model)

# Schedule the workflow to run continuously or at regular intervals
#flow.run()
"""

run_core()