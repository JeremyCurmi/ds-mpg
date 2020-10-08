from dataset import DataSet
from scaler import Scaler
from imputer import Imputer
from categorical_encoder import CategoryEncoder
from feature_selector import FeatureTypeSelector, FeatureSelector
from feature_creator import FeatureCreator
from pipelines import numeric_preprocessor_pipeline, cleaner_numeric_preprocessor_pipeline
from cleaner import Cleaner

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

    feat_creator = FeatureCreator()
    check = feat_creator.fit_transform(tmp)
    print(check)
    
    tmp_train = data.X_train
    
    cleaner1 = Cleaner()
    tmp1 = cleaner1.fit_transform(tmp_train)
    print("cleaned: \n",tmp1)
    print(tmp1.isnull().sum())
    
    num_pipe_df = numeric_preprocessor_pipeline.fit_transform(tmp_train)
    print(num_pipe_df)
    
    clean_num_pipe_df = cleaner_numeric_preprocessor_pipeline.fit_transform(tmp_train)
    print("\n clean_numerical pipeline df: \n",clean_num_pipe_df)
    print(clean_num_pipe_df.isnull().sum())

if __name__ == "__main__":
    main()