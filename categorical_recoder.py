import warnings

from df_transformer import DfTransformer


class CategoricalRecoder(DfTransformer):
    """
        Recode categorical feature values
    """
    
    def __init__(self, feature_mapper):
        self.name = "CategoricalRecoder"
        super().log_start(self.name)  

        self.feature_mapper = feature_mapper
        self.features_with_null_values_dict = {}
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        self.null_values_validator(X)
        X_recode = self.recoder(X)
        
        super().log_end(self.name)
        return X

    def recoder(self, X):

        for feature in self.feature_mapper:
            X.loc[X[feature].notnull(),feature] = X[feature].map(self.feature_mapper[feature]).fillna(X[feature])

        return X
        
    def null_values_validator(self, X):
        
        for feature in self.feature_mapper:
            num_missing_values = X[feature].isnull().sum()
            if num_missing_values > 0:
                self.features_with_null_values_dict[feature] = num_missing_values
                    
        if len(list(self.features_with_null_values_dict))>0:
            warnings.warn("The following features has missing values and will not be imputed {}".format(self.features_with_null_values_dict),UserWarning)
