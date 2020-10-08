import logging
import time
logging.basicConfig(format="%(asctime)s - %(message)s",level=logging.INFO)

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureCreator(BaseEstimator,TransformerMixin):
    """
        Create Features here
    """
    def __init__(self, power = True):
        self.start_time = time.time()
        self.name = "FeatureCreator"
        logging.info(f"{self.name} Processing ...")

        self.columns = []
        self.power = power

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        X_feat = self.feature_create_power(X)
        self.columns = X_feat.columns

        logging.info(f"{self.name} Finished Processing, total time taken: --- {round((time.time() - self.start_time),6)} seconds ---")
        return X_feat


    
    def feature_create_power(self, X):
        X_feat_create = X.copy()
        if self.power:
            X_feat_create["power"] = X_feat_create["horsepower"]/X_feat_create["weight"]
        return X_feat_create