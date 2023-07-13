import unittest
import numpy as np
from utils.metrics import *
class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        self.y_true = np.array([0,0,0,0,1,0,0,1,1,1])
        self.y_pred= np.array([0,0,0,0,0,1,1,1,1,1])

    def test_accuracy(self):
        results = accuracy(self.y_true, self.y_pred)
        self.assertEqual(results, 0.7)


if __name__ == "__main__":
    unittest.main()