import sys
sys.path.insert(0, "../../")
from hda.diagnostics.CXSpectrometer import CXSpectrometer
from hda.diagnostics.spectrometer import XRCSpectrometer

# # Run CXSpectrometer
# xrcs_spectrometer_object = XRCSpectrometer()
# xrcs_spectrometer_object.test_flow()

# Run CXSpectrometer
cx_spectrometer_object = CXSpectrometer()
cx_spectrometer_object.test_flow()

print(cx_spectrometer_object.adf11)
print(cx_spectrometer_object.adf15)

