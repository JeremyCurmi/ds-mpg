import pandas as pd
import numpy as np
import warnings

import logging
import time
logging.basicConfig(format="%(asctime)s - %(message)s",level=logging.INFO)


from sklearn.base import BaseEstimator, TransformerMixin

class Cleaner(BaseEstimator, TransformerMixin):
    """
        Data cleaning before preprocessing
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.name = "Cleaner"
        logging.info(f"{self.name} Processing ...")
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_clean = X.copy()
        X_clean = self.feature_clean_horsepower(X_clean)
        
        logging.info(f"{self.name} Finished Processing, total time taken: --- {round((time.time() - self.start_time),6)} seconds ---")
        return X_clean

    def feature_clean_horsepower(self, X):
        X["horsepower"] = pd.to_numeric(X["horsepower"], errors = "coerce")
        return X
