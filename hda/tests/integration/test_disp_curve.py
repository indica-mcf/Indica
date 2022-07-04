import unittest
from hda.snippets.BayesData import disp_curve
import numpy as np


class TestDispCurve(unittest.TestCase):

    def test_disp_curve_returns_numpy(self):
        p=disp_curve(np.arange(1,1031))
        self.assertIsInstance(p, np.ndarray, "Should return numpy.ndarray")

    def test_disp_curve_shape(self):
        p = disp_curve(np.arange(1,1031))
        self.assertEqual(p.shape, (1030,), "Should have 1030 pixels")

    def test_if_not_called_with_array(self):
        with self.assertRaises(TypeError):
            p = disp_curve()



if __name__ == "__main__":
    unittest.main()