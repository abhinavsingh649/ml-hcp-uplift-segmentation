import pandas as pd
from xgboost import XGBRegressor
from src.config import TREEMETHOD, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE

class XLearnerUplift:
    def __init__(self):
        self.model_t = XGBRegressor(
            n_estimators=N_ESTIMATORS, 
            max_depth=MAX_DEPTH, 
            learning_rate=LEARNING_RATE,
            tree_method=TREEMETHOD
        )
        self.model_c = XGBRegressor(
            n_estimators=N_ESTIMATORS, 
            max_depth=MAX_DEPTH, 
            learning_rate=LEARNING_RATE,
            tree_method=TREEMETHOD
        )

    def fit(self, X, treatment, y):
        X_t = X[treatment == 1]
        y_t = y[treatment == 1]

        X_c = X[treatment == 0]
        y_c = y[treatment == 0]

        self.model_t.fit(X_t, y_t)
        self.model_c.fit(X_c, y_c)

    def predict_uplift(self, X):
        mu1 = self.model_t.predict(X)
        mu0 = self.model_c.predict(X)
        uplift = mu1 - mu0
        return uplift

    def save_model(self, filepath):
        import joblib
        joblib.dump(self, filepath)
        print(f"Model successfully saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        import joblib
        model = joblib.load(filepath)
        print(f"Model successfully loaded from {filepath}")
        return model