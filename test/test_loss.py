import unittest
import numpy as np
from utils.loss import *
import torch

class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        self.y_pred = np.array([[3,2], [4, 12], [3, 33]])
        self.y_test = np.array([1, 1, 1])

    def test_cross_entropy_loss(self):
        criterion = torch.nn.CrossEntropyLoss()
        expected_result = np.array(cross_entropy_loss(self.y_pred, self.y_test), dtype=np.float32)
        result = criterion(torch.tensor(self.y_pred, dtype=torch.float), 
                        torch.tensor(self.y_test, dtype=torch.long)).numpy()
        np.testing.assert_allclose(expected_result, result, rtol=1e-03)
    
    def test_softmax(self):
        criterion = torch.nn.Softmax(dim=1)
        expected_result = criterion(torch.tensor(self.y_pred, dtype=torch.float)).numpy()
        result = np.array(softmax(self.y_pred), dtype=np.float32)
        np.testing.assert_array_equal(expected_result, result)

if __name__ == "__main__":
    unittest.main()