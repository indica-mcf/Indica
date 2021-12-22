from indica.readers import ADASReader
import numpy as np

# ADAS reader
adas_reader = ADASReader()


# Read ADF11 file
ADF11 = {"ar": {"scd": "89", "acd": "89", "ccd": "89"}}
scd = adas_reader.get_adf11("scd", 'ar', ADF11['ar']["scd"])
print(scd)

# Get ADF12 file
dummy = adas_reader.get_adf12("H", "0", "C", "6", "99")
