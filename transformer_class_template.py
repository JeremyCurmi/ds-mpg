import warnings
import logging
import time
logging.basicConfig(format="%(asctime)s - %(message)s",level=logging.INFO)

from sklearn.base import BaseEstimator, TransformerMixin


class ClassName(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.start_time = time.time()
        self.name = "CategoryEncoder"
        logging.info(f"{self.name} Processing ...")
        
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X, y=None):
        logging.info(f"{self.name} Finished Processing, total time taken: --- {round((time.time() - self.start_time),6)} seconds ---")

        