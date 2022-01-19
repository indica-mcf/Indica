import sys
sys.path.insert(0, "../../")
from hda.diagnostics.CXSpectrometer import CXSpectrometer

# Run CXSpectrometer
cx_spectrometer_object = CXSpectrometer()
cx_spectrometer_object.test_flow()



