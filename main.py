from dataset import DataSet
from scaler import Scaler
from imputer import Imputer
from categorical_encoder import CategoryEncoder
from feature_selector import FeatureTypeSelector, FeatureSelector

import pandas as pd

def main():
    data = DataSet()
    data.fetch_data()
    data.split_X_y()
    data.split_X_y_train_test()

    print(data.X_train.head())

    scaler = Scaler()
    tmp = scaler.fit_transform(data.X_train[["cylinders","displacement"]])
    print(tmp.head())
    scaler = Scaler(method="robust")
    tmp = scaler.fit_transform(data.X_train[["cylinders","displacement"]])
    print(tmp.head())
    scaler = Scaler(method="minmax")
    tmp = scaler.fit_transform(data.X_train[["cylinders","displacement"]])
    print(scaler.min_)

    print()
    print()
    tmp = data.X_train
    tmp["horsepower"] = pd.to_numeric(tmp["horsepower"], errors = "coerce")
    imputer = Imputer()
    check = imputer.fit_transform(tmp[["horsepower"]])
    print(check)

    dummies = CategoryEncoder()
    check = dummies.fit_transform(tmp)
    print(check.head())
    check1 = dummies.transform(data.X_test)
    print()
    print()
    print(check1)
    print()
    print()
    feat_sel = FeatureSelector()
    check = feat_sel.fit_transform(tmp)
    print(check.head())

    feat_sel = FeatureSelector(feature_list=["cylinders","car name"])
    check = feat_sel.fit_transform(tmp)
    print(check.head())
if __name__ == "__main__":
    main()