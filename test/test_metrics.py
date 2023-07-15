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

        self.assertAlmostEqual(results, 0.7, places=7)

    def test_mean_squared_error(self):
        mse = mean_squared_error(self.lin_y_true, self.lin_y_pred)
        rmse = mean_squared_error(self.lin_y_true, self.lin_y_pred, root=True)
        self.assertAlmostEqual(mse, metrics.mean_squared_error(self.lin_y_true, self.lin_y_pred), places=7)
        self.assertAlmostEqual(rmse, metrics.mean_squared_error(self.lin_y_true, self.lin_y_pred, squared=False), places=7)
        
    def test_mean_absolute_error(self):
        result = mean_absolute_error(self.lin_y_true, self.lin_y_pred)
        result = mean_absolute_error(self.lin_y_true, self.lin_y_pred)
        self.assertAlmostEqual(result, metrics.mean_absolute_error(self.lin_y_true, self.lin_y_pred), places=7)

    def test_recall(self):
        result = recall(self.cat_y_true, self.cat_y_pred)
        np.testing.assert_allclose(result, metrics.recall_score(self.cat_y_true, self.cat_y_pred, average=None), rtol=1e-03)

    def test_precision(self):
        result = precision(self.cat_y_true, self.cat_y_pred)
        np.testing.assert_allclose(result, metrics.precision_score(self.cat_y_true, self.cat_y_pred, average=None), rtol=1e-03)

    def test_f1_score(self):
        result = f1_score(self.cat_y_true, self.cat_y_pred)
        np.testing.assert_allclose(result, metrics.f1_score(self.cat_y_true, self.cat_y_pred, average=None), rtol=1e-03)
    def test_confusion_matrix(self):
        cat_y_true = [1, 2, 3, 1, 2, 3]
        cat_y_pred= [1, 2, 1, 1, 2, 3]
        result = metrics.confusion_matrix(cat_y_true, cat_y_pred)
        np.testing.assert_allclose(result, metrics.confusion_matrix(cat_y_true, cat_y_pred))

if __name__ == "__main__":
    unittest.main()