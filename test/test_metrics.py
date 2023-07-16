import unittest
import numpy as np
from utils.metrics import *
from sklearn import metrics
class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        self.cat_y_true = np.array([0,0,0,0,1,0,0,1,1,1])
        self.cat_y_pred= np.array([0,0,0,0,0,1,1,1,1,1])
        self.lin_y_true = np.array([100, -100, 0, 200])
        self.lin_y_pred = np.array([-100, -100, -100, -100])

    def test_accuracy(self):
        results = accuracy(self.cat_y_true, self.cat_y_pred)
        expected_result = 0.7
        self.assertAlmostEqual(results, expected_result, places=7)

    def test_mean_squared_error(self):
        mse = mean_squared_error(self.lin_y_true, self.lin_y_pred)
        expected_mse = metrics.mean_squared_error(self.lin_y_true, self.lin_y_pred)
        rmse = mean_squared_error(self.lin_y_true, self.lin_y_pred, root=True)
        expected_rmse = metrics.mean_squared_error(self.lin_y_true, self.lin_y_pred, squared=False)
        self.assertAlmostEqual(mse, expected_mse, places=7)
        self.assertAlmostEqual(rmse, expected_rmse, places=7)
        
    def test_mean_absolute_error(self):
        result = mean_absolute_error(self.lin_y_true, self.lin_y_pred)
        expected_result =  metrics.mean_absolute_error(self.lin_y_true, self.lin_y_pred)
        self.assertAlmostEqual(result, expected_result, places=7)

    def test_recall(self):
        result = recall(self.cat_y_true, self.cat_y_pred)
        expected_result = metrics.recall_score(self.cat_y_true, self.cat_y_pred, average=None)
        np.testing.assert_allclose(result, expected_result, rtol=1e-03)

    def test_precision(self):
        result = precision(self.cat_y_true, self.cat_y_pred)
        expected_result = metrics.precision_score(self.cat_y_true, self.cat_y_pred, average=None)
        np.testing.assert_allclose(result, expected_result, rtol=1e-03)

    def test_f1_score(self):
        result = f1_score(self.cat_y_true, self.cat_y_pred)
        expected_result = metrics.f1_score(self.cat_y_true, self.cat_y_pred, average=None)
        np.testing.assert_allclose(result, expected_result, rtol=1e-03)
    def test_confusion_matrix(self):
        cat_y_true = [1, 2, 3, 1, 2, 3]
        cat_y_pred= [1, 2, 1, 1, 2, 3]
        result = metrics.confusion_matrix(cat_y_true, cat_y_pred)
        expected_result = metrics.confusion_matrix(cat_y_true, cat_y_pred)
        np.testing.assert_allclose(result, expected_result)

if __name__ == "__main__":
    unittest.main()