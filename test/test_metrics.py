import unittest
import numpy as np
from utils.metrics import *
class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        self.cat_y_true = np.array([0,0,0,0,1,0,0,1,1,1])
        self.cat_y_pred= np.array([0,0,0,0,0,1,1,1,1,1])
        self.lin_y_true = np.array([100, -100, 0, 200])
        self.lin_y_pred = np.array([-100, -100, -100, -100])

    def test_accuracy(self):
        results = accuracy(self.cat_y_true, self.cat_y_pred)

        self.assertAlmostEqual(results, 0.7, places=7)

    def test_mean_square_error(self):
        mse = mean_square_error(self.lin_y_true, self.lin_y_pred)
        rmse = mean_square_error(self.lin_y_true, self.lin_y_pred, root=True)
        self.assertAlmostEqual(mse, 35000.0, places=7)
        self.assertAlmostEqual(rmse, 187.08286933869707, places=7)
        
    def test_mean_absolute_error(self):
        result = mean_absolute_error(self.lin_y_true, self.lin_y_pred)
        self.assertAlmostEqual(result, 150.0, places=7)

    def test_recall(self):
        result = recall(self.cat_y_true, self.cat_y_pred)
        np.testing.assert_allclose(result, np.array([0.66666667, 0.75]), rtol=1e-03)

    def test_precision(self):
        result = precision(self.cat_y_true, self.cat_y_pred)
        np.testing.assert_allclose(result, np.array([0.8, 0.6]), rtol=1e-03)

    def test_f1_score(self):
        result = f1_score(self.cat_y_true, self.cat_y_pred)
        np.testing.assert_allclose(result, np.array([0.72727272, 0.66666666]), rtol=1e-03)

if __name__ == "__main__":
    unittest.main()