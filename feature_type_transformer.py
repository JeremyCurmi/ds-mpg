import pandas as pd
import numpy as np
import warnings

import logging
import time
logging.basicConfig(format="%(asctime)s - %(message)s",level=logging.INFO)

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureTypeTransformer(BaseEstimator,TransformerMixin):
    """
        Input a list of features that needs to change type.
        If num_to_str is TRUE then input features are expected to be numeric and needs to be changed to str
        Otherwise if num_to_str is FALSE then input features are excpected to be str and needs to be changed to numeric
    """
    def __init__(self, feature_list=[], num_to_str = True):
        self.start_time = time.time()
        self.name = "FeatureTypeTransformer"
        logging.info(f"{self.name} Processing ...")
        
        self.feature_list = feature_list
        self.num_to_str = num_to_str
        self.columns = []
    
    def fit(self, X,y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = self.type_transformer(X)
        logging.info(f"{self.name} Finished Processing, total time taken: --- {round((time.time() - self.start_time),6)} seconds ---")
        
        self.columns = X_transformed.columns
        return X_transformed
    
    def type_transformer(self, X):
        if self.num_to_str:
            for feature in self.feature_list:
                X[feature] = X[feature].astype(str)
        else:
            for feature in self.feature_list:
                X[feature] = X[feature].astype('int32')            
        return X
    
    def get_feature_names(self):
        return self.columns