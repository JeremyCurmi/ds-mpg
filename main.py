from dataset import DataSet
from scaler import Scaler
from imputer import Imputer
from categorical_encoder import CategoryEncoder
from feature_selector import FeatureTypeSelector, FeatureSelector
from feature_creator import FeatureCreator
from pipelines import numeric_preprocessor_pipeline, cleaner_numeric_preprocessor_pipeline, scaler_preprocessor_pipeline, full_pipeline, feat_type_transformer_pipeline
from cleaner import Cleaner
from feature_type_transformer import FeatureTypeTransformer
from categorical_orderer import CategoricalOrderer

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
    print()
    print()
    print()
    print(clean_num_pipe_df.head())
    print(clean_num_pipe_df.shape)
    print()
    print()
    print()
    scaler_pipe_df = scaler_preprocessor_pipeline.fit_transform(tmp_train)
    print(scaler_pipe_df)
    print()
    print()
    print()
    full_pipe_df = full_pipeline.fit_transform(tmp_train)
    print(full_pipe_df)
    print()
    print(full_pipeline[0].get_params())
    print("\n\n\n\n\n")
    print()
    print(full_pipeline.get_params())
    print(clean_num_pipe_df.info())
    print("\n\n\n\n\n")
    print()
    feat_type_transf_df = feat_type_transformer_pipeline.fit_transform(tmp_train)
    print(feat_type_transf_df.head())
    print(feat_type_transf_df.info())
    print()
    print()
    print(feat_type_transf_df['cylinders'])
    print("\n\n\n\n\n")
    print()
    print("\n\n\n\n\n")
    print()
    print("Testing Categorical feature ordere \n\n\n\n")
    tmp_train = tmp_train.append(pd.Series(), ignore_index=True)
    print(tmp_train["model year"].value_counts(dropna=False))
    
    feat_orderer = CategoricalOrderer(feature_mapper_dict={"model year":{733:70},"cylinders":{8:111}})
    feat_orderer_df = feat_orderer.fit_transform(tmp_train)
    print(feat_orderer_df)
    
    
    
    
    
    
if __name__ == "__main__":
    main()