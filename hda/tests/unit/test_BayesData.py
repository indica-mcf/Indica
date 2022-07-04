import unittest

import hda.snippets.BayesData as BayesData

class TestAcceptance(unittest.TestCase):

    def test_accept_if_new_value_is_greater(self):
        current_value = 0.01
        new_value = 0.1
        self.assertTrue(BayesData.acceptance(current_value, new_value), "Should be True")



if __name__ == "__main__":
    unittest.main()