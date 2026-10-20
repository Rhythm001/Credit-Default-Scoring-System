import numpy as np
from sklearn.linear_model import LogisticRegression

def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)

class PlattCalibrator:
    def fit(self, raw_prob, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_prob), y)
        return self

    def predict(self, raw_prob):
        return self.lr.predict_proba(_logit(raw_prob))[:, 1]
