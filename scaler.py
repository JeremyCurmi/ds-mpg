import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from df_transformer import DfTransformer

class Scaler(DfTransformer):
    """
        Preprocessor for Numerical Features, which scales the values, choices are:
        1.StandardScaler -> Scales the feature to standard normal distribution
        2.RobustScaler -> Scales the feature to IQR to reduce outlier skewness of the feature
        3.MinMaxScaler -> Scales the feature in the range [0,1]

    """
    def __init__(self, method = "standard"):
        self.name = "Scaler"
        super().log_start(self.name)

        self.scaler = None
        self.scale_ = None
        self.method = method
        self.columns = None

        if method == "standard":
            self.mean_ = None
        elif method == "robust":
            self.center_ = None
        elif method == "minmax":
            self.min_ = None
            

    def fit(self, X, y=None):
        if self.method == "standard":
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            self.mean_ = pd.Series(self.scaler.mean_, index = X.columns)
        elif self.method == "robust":
            self.scaler = RobustScaler()
            self.scaler.fit(X)
            self.center_ = pd.Series(self.scaler.center_, index = X.columns)
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
            self.scaler.fit(X)
            self.min_ = pd.Series(self.scaler.min_, index = X.columns)
            self.max_ = pd.Series(self.scaler.min_, index = X.columns)            
        else:
            "add more different scaler methods"
            print("add more different scaler methods")
        self.scale_ = pd.Series(self.scaler.scale_, index=X.columns)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index = X.index, columns = X.columns)
        
        super().log_end(self.name)
        return X_scaled_df

    def return_feature_names(self):
        return list(self.columns)