from typing import List
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from joblibspark import register_spark
from sklearn.linear_model import LogisticRegression
from sklearn.utils import parallel_backend
from sklearn.metrics import (
    precision_score, recall_score,
    accuracy_score, confusion_matrix
)

from pyspark.sql.dataframe import DataFrame
from torchvision.utils import make_grid
from torch import tensor

register_spark()

class LogReg:
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        # L2-regularized logistic regression
        self.model = LogisticRegression(
            C=C,
            penalty='l2',
            solver='saga',
            max_iter=max_iter,
            n_jobs=8,
            random_state=0
        )

    def train(self, df: DataFrame) -> List:
        # collect into numpy arrays
        X = np.array(df.select("image").collect()).reshape(-1, 3072)
        y = np.array(df.select("label").collect()).reshape(-1)
        print("Training data shape:", X.shape)

        # fit under spark backend
        with parallel_backend("spark"):
            self.model.fit(X, y)

        # evaluate on training set
        y_pred = self.model.predict(X)
        acc   = accuracy_score(y, y_pred)
        prec  = precision_score(y, y_pred, average="macro")
        rec   = recall_score(y, y_pred, average="macro")
        f1    = 2 * prec * rec / (prec + rec)

        return [self.model, acc, prec, rec, f1]

    def predict(self, df: DataFrame) -> List:
        X = np.array(df.select("image").collect()).reshape(-1, 3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        # direct predictions
        y_pred = self.model.predict(X)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="macro")
        rec = recall_score(y, y_pred, average="macro")
        f1 = 2 * prec * rec / (prec + rec)
        cm = confusion_matrix(y, y_pred)

        # optionally visualize a few examples per class
        self.visualize(X, y_pred)

        return [y_pred, acc, prec, rec, f1, cm]

    def visualize(self, X_flat: np.ndarray, y_pred: np.ndarray, samples_per_class: int = 25):
        # un-flatten and de-normalize if needed
        X = X_flat.reshape(-1, 32, 32, 3)
        classes = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
        for cls in range(10):
            idxs = np.where(y_pred == cls)[0][:samples_per_class]
            if len(idxs) == 0:
                continue
            imgs = X[idxs].transpose(0,3,1,2)  # to C×H×W
            grid = make_grid(tensor(imgs), nrow=5)
            plt.figure(figsize=(6,6))
            plt.title(f"Predicted: {classes[cls]}")
            plt.axis('off')
            plt.imshow(grid.permute(1,2,0))
            plt.show()
