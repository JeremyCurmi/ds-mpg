
from df_transformer import DfTransformer

class FeatureCreator(DfTransformer):
    """
        Create Features here
    """
    def __init__(self, power = True, disp_weight_ratio = True, model = True, brand = True
                ,annum = True):

        self.name = "FeatureCreator"
        super().log_start(self.name)

        self.columns = []
        self.power = power
        self.disp_weight_ratio = disp_weight_ratio
        self.model = model
        self.brand = brand
        self.annum = annum

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_feat = X.copy()
        X_feat = self.create_feature(X_feat)
        self.columns = X_feat.columns

        super().log_end(self.name)
        return X_feat

    
    def create_feature(self, X):
        if self.power:
            X["power"] = X["horsepower"]/X["weight"]
        if self.disp_weight_ratio:
            X["disp_to_weight_rat"] = X["displacement"]/X["weight"]
        if self.brand:
            X["brand"] = X["car name"].str.split().str[0]
        if self.model:
            X["model"] = X["car name"].str.split().str[1]
        if self.annum:
            X["annum"] = "70's"
            X.loc[X["model year"]>= 80, "annum"] = "80's"
        return X


    def get_feature_names(self):
        return self.columns