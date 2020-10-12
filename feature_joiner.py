import pandas as pd

from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer

from df_transformer import DfTransformer


class FeatureUnioner(DfTransformer):
    """
        Joins all dataframes coming from multiple transformers into one dataframe
    """
    def __init__(self, transformer_list, n_jobs=-1):
        self.name = "FeatureUnioner"
        super().log_start(self.name)

        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.feature_union = FeatureUnion(self.transformer_list, self.n_jobs)
        self.columns = []
        
    def fit(self, X, y=None):
        self.feature_union.fit(X)
        return self

    def transform(self, X, y=None):
        
        X_transform = self.feature_union.transform(X)
        self.concat_df_columns()
        X_transform = pd.DataFrame(X_transform, index=X_transform.index, columns = self.columns)

        super().log_end(self.name)
        return X_transform
    
    def concat_df_columns(self):
        
        for transformer in self.transformer_list:
            columns = transformer[1].steps[-1][1].get_feature_names()
            self.columns += columns
            
    def get_params(self, deep = True):
        """
            used for gridsearch
        """
        return self.feature_union.get_params(deep=deep)
            
            
class FeatureColumnTransformer(DfTransformer):
    
    def __init__(self, transformers, remainder="passthrough", n_jobs=-1):
        self.name = "FeatureColumnTransformer"
        super().log_start(self.name)
        
        self.transformers = transformers
        self.remainder = remainder
        self.n_jobs = n_jobs
        self.column_transfomer = ColumnTransformer(
            transformers = self.transformers,
            remainder = self.remainder,
            n_jobs=self.n_jobs
            )
        self.columns = None
        self.column_types = None
    
    def fit(self, X, y=None):
        self.column_transfomer.fit(X)
        return self
    
    def transform(self, X, y=None):
        X_concat = self.column_transfomer.transform(X)
        self.columns = self.column_transfomer.get_feature_names()
        self.rename_df_columns()
                
        X_concat = pd.DataFrame(X_concat, index = X.index, columns = self.columns)
        X_concat_df = self.redefine_column_types(X,X_concat)

        super().log_end(self.name)
        return X_concat_df
    
    def rename_df_columns(self):
        for i,col in enumerate(self.columns):
            self.columns[i] = col.split(sep="__")[-1]
            
    def redefine_column_types(self, X_input, X_output):
        for feature in X_input.columns:
            if feature in X_output.columns:
                X_output[feature]=X_output[feature].astype(X_input[feature].dtypes.name)
        return X_output