# Utils

This project contains some useful machine learning/statistics concepts which I implemented to strengthen my understanding of ceratain topics.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The project uses Python and the following libraries:

- numpy
- sklearn
- matplotlib

Here is an example of working with few of implemented functions:

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from utils import metrics
from models import models
from utils import plots

np.set_printoptions(suppress=True)

# Load the iris dataset
iris = datasets.load_iris()
X = iris['data']
X = X[:, [0, 2]]
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = models.K_Nearest_Neighbors(3)
model.fit(X_train, y_train)
preds, y_true = model.predict(X_test, y_test)

# Plot decision regions and return computed metrics
plots.plot_decision_regions(X_test, y_test, model, print_metrics=True)
```
Outputs are visible inside example.ipynb file
