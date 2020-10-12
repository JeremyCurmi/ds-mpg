import warnings

from df_transformer import DfTransformer

class CategoricalOrderer(DfTransformer):
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
        self.name = "CategoricalOrderer"
        super().log_start(self.name)
        
        self.feature_mapper = feature_mapper_dict
        self.fill_value = replace_remaining_non_nas_values_by
        self.features_with_null_values_dict = {}
        self.features_with_missing_mapping_values_dict = {}
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        self.null_values_validator(X)
        self.missing_value_mapping_validator(X)
        X_ordered = self.orderer(X)
        
        super().log_end(self.name)
        return X_ordered

    def orderer(self, X):
        
        for feature in self.feature_mapper:
            X.loc[X[feature].notnull(),feature] = X[feature].map(self.feature_mapper[feature]).fillna(self.fill_value)
            # X.loc[:,feature] = X[feature].map(self.feature_mapper[feature]).fillna(self.fill_value) # use this if you want to impute null values as well

        return X
    
    def null_values_validator(self, X):
        
        for feature in self.feature_mapper:
            num_missing_values = X[feature].isnull().sum()
            self.features_with_null_values_dict[feature] = num_missing_values
                    
        if len(list(self.features_with_null_values_dict))>0:
            warnings.warn("The following features has missing values and will not be imputed {}".format(self.features_with_null_values_dict),UserWarning)

    def missing_value_mapping_validator(self,X):
        for feature in self.feature_mapper:
            n_unique_feature_values = X[feature].nunique()
            n_unique_value_mappings = len(self.feature_mapper[feature])
            
            if n_unique_feature_values != n_unique_value_mappings:
                self.features_with_missing_mapping_values_dict[feature] = {"total_feature_unique_values":n_unique_feature_values,"provided_mapping_unique_values":n_unique_value_mappings}

        if len(list(self.features_with_missing_mapping_values_dict))>0:
            warnings.warn("The specified order mapping(s) for the following feature(s) have less (or more) unique values then the feature(s) have {}".format(self.features_with_missing_mapping_values_dict),UserWarning)
