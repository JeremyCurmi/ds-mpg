import warnings
import logging
import time
logging.basicConfig(format="%(asctime)s - %(message)s",level=logging.INFO)

from sklearn.base import BaseEstimator, TransformerMixin


class DfTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.name = "ClassName"
        self.log_start( self.name)
        self.columns = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        self.log_end(self.name)

    def log_start(self,name):
        self.start_time = time.time()
        logging.info(f"{name} Processing ...")

    def log_end(self,name):
        logging.info(f"{name} Finished Processing, total time taken: --- {round((time.time() - self.start_time),6)} seconds ---")
     
    def get_feature_names(self):
        return self.columns