from copy import deepcopy
import numpy as np

from xarray import DataArray
from indica.models.diode_filters import BremsstrahlungDiode
from indica.numpy_typing import LabeledArray
import indica.readers.read_st40 as read_st40
from indica.models.plasma import example_run as example_plasma
from scipy.interpolate import interp1d
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
from indica.workflows import run_tomo_1d


def example_run(
        pulse,  
        plasma=None, 
): 
    
    if plasma is None:
        plasma = example_plasma(pulse=pulse)

    st40 = read_st40.ReadST40(pulse) 
    st40(["pi"]) 

    # Initialise Diagnostic Models
    diagnostic_name = "pi"
    los_transform = st40.binned_data["pi"]["spectra"].transform
    st40.binned_data["pi"]["spectra"].transform.set_equilibrium(
        st40.binned_data["pi"]["spectra"].transform.equilibrium
    )
    model = BremsstrahlungDiode(diagnostic_name)
    model.plasma = plasma
    model.set_los_transform(los_transform)
    bckc = model()
    return plasma, model, bckc


def Bremsstrahlung(
        pulse,  
        channels=np.linspace(18,35,18), 
        tstart: float = 0.020,
        tend: float = 0.10,
        dt: float = 0.010, 
        wavelength_start=531,
        wavelength_end=532,
        instrument="pi",        
):
    
    st40 = read_st40.ReadST40(pulse) 
    st40([instrument]) 

    length=(tend-tstart)/dt+1 #+1 so it will take into account that the last point also should be considered
    times=np.linspace(float(tstart), float(tend), int(length), endpoint=True)

    y = example_run(pulse)[1].transmission
    xdata = np.linspace(wavelength_start, wavelength_end, int(len(y)))
    transmission_inter = interp1d(xdata, y)
    
    bckgemission_full=[]

    for chan in channels:
        for t in times:
        
            reader=st40.binned_data[instrument]["spectra"].sel(t=t, method="nearest").sel(channel=chan, wavelength=slice(wavelength_start, wavelength_end)) 

            y_values=reader.where(reader<0.05)
            x_values=reader.where(reader<0.05).coords["wavelength"]
            y_data=np.array(y_values)
            x_data=np.array(x_values)

            xdata_new=np.linspace(wavelength_start, wavelength_end, len(y_values))
            transmission=transmission_inter(xdata_new)

            yfit=[]
            fit, cov=np.polyfit(x_data, y_data,1, cov=True)
            for i in range(0, len(x_data)):
                yfit.append(fit[0]*x_data[i]+fit[1])
            yfit=np.array(yfit)
            yfit=yfit*transmission

            bckgemission = np.mean(yfit)

            coefficient=len(y_values)
            bckgemission=bckgemission*coefficient
            bckgemission_full.append(bckgemission)
            
    background = [bckgemission_full[i:i + len(times)] for i in range(0, len(bckgemission_full), len(times))]
    brem=DataArray(background, coords={'channel': channels,'t':times}, dims=["channel", "t"])
    brem.attrs = st40.binned_data["pi"]["spectra"].attrs

    data = {}
    data["bremsstrahlung"]=(brem)
    return data, brem

run_tomo_1d.pi(10968)