import logging
import time
logging.basicConfig(format="%(asctime)s - %(message)s",level=logging.INFO)

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureCreator(BaseEstimator,TransformerMixin):
    """
        Create Features here
    """
    def __init__(self, power = True, disp_weight_ratio = True):
        self.start_time = time.time()
        self.name = "FeatureCreator"
        logging.info(f"{self.name} Processing ...")

        self.columns = []
        self.power = power
        self.disp_weight_ratio = disp_weight_ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_feat = X.copy()
        X_feat = self.feature_create_power(X_feat)
        X_feat = self.feature_create_disp_weight_ratio(X_feat)
        self.columns = X_feat.columns

        logging.info(f"{self.name} Finished Processing, total time taken: --- {round((time.time() - self.start_time),6)} seconds ---")
        return X_feat

    
    def feature_create_power(self, X):
        if self.power:
            X["power"] = X["horsepower"]/X["weight"]
        return X
    
    def feature_create_disp_weight_ratio(self, X):
        if self.disp_weight_ratio:
            X["disp_to_weight_rat"] = X["displacement"]/X["weight"]
        return X
        
    
    def get_feature_names(self):
        return self.columns