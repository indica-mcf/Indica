import unittest
import numpy as np

import hda.diagnostics.CrystalSpectrometer as CrystalSpectrometer



class TestDielectronicRecombination(unittest.TestCase):

    def test_input_wrong_dim_atomicdata(self):
        Te = np.array([1])
        atomic_data = np.ones(shape=(1,1))
        with self.assertRaises(IndexError):
            result = CrystalSpectrometer.diel_calc(atomic_data, Te, label="he")

    def test_check_output_shape(self):
        dim1 = 2
        dim2 = 4
        Te = np.ones(shape=(dim1,))
        atomic_data = np.ones(shape=(dim2, 5))
        result = CrystalSpectrometer.diel_calc(atomic_data, Te, label="he")
        self.assertEqual((dim1,dim2), result.shape, f"Should have shape ({dim1}, {dim2})")

    def test_wrong_label_given(self):
        label="He"
        Te = np.ones(shape=(1,))
        atomic_data = np.ones(shape=(1, 5))
        with self.assertRaises(UnboundLocalError):
            result = CrystalSpectrometer.diel_calc(atomic_data, Te, label=label)

class TestCrystalSpectrometer(unittest.TestCase):

    def test_initialise(self):
        obj = CrystalSpectrometer.CrystalSpectrometer()

    # def test_workflow(self):
    #     obj = CrystalSpectrometer.CrystalSpectrometer()
    #     obj.test_workflow()



if __name__ == "__main__":
    unittest.main()