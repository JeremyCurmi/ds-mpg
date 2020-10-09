import warnings
import logging
import time
logging.basicConfig(format="%(asctime)s - %(message)s",level=logging.INFO)

from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalOrderer(BaseEstimator, TransformerMixin):
    """
        This Transformer Requires a feature_mapper_dict which is a dictionary in which the keys 
        are the features and the values are dictionaries with the required mapping of each feature 
        respectively. It also requires (optional) a value that will replace any of the current
        values which are not assigned in the feature mapping dictionary.
        
        Its IMPORTANT to note that any missing values found in a feature that is being ordered, will
        REMAIN as so, i.e. no imputation occurs to missing values, that is the job of an Imputer Transformer.
        
        Its IMPORTANT that none of the new values defined in the feature_mapper_dict should be equal
        to the replace_remaining_non_nas_values_by (by default 0), so the new values should start
        from 1 onwards (the higher the value the more important that value is, example 1 -good, 5 the best).
    """
    
    def __init__(self, feature_mapper_dict, replace_remaining_non_nas_values_by=0):
        self.start_time = time.time()
        self.name = "CategoricalOrderer"
        logging.info(f"{self.name} Processing ...")
        
        self.feature_mapper = feature_mapper_dict
        self.fill_value = replace_remaining_non_nas_values_by
        self.feature_with_null_values_dict = {}
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        self.feature_values_validator(X)
        X_ordered = self.orderer(X)
        
        logging.info(f"{self.name} Finished Processing, total time taken: --- {round((time.time() - self.start_time),6)} seconds ---")
        return X_ordered

    def orderer(self, X):
        
        for feature in self.feature_mapper:
            X.loc[X[feature].notnull(),feature] = X[feature].map(self.feature_mapper[feature]).fillna(self.fill_value)
            # X.loc[:,feature] = X[feature].map(self.feature_mapper[feature]).fillna(self.fill_value) # use this if you want to impute null values as well

        return X
    
    def feature_values_validator(self, X):
        
        for feature in self.feature_mapper:
            num_missing_values = X[feature].isnull().sum()
            self.feature_with_null_values_dict[feature] = num_missing_values
                    

        if len(list(self.feature_with_null_values_dict))>0:
            warnings.warn("The following features had missing values and will not be imputed {}".format(self.feature_with_null_values_dict),UserWarning)

        