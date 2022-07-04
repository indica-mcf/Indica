import unittest

import hda.diagnostics.CrystalSpectrometer as CrystalSpectrometer

class TestAcceptance(unittest.TestCase):

    def test_dielectronic_recombination_shape(self):
        Te = [1]
        data = []
        CrystalSpectrometer.diel_calc(data, Te, label="He")


if __name__ == "__main__":
    unittest.main()