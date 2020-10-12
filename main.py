from dataset import DataSet
from scaler import Scaler
from imputer import Imputer
from categorical_encoder import CategoricalEncoder
from feature_selector import FeatureSelector
from feature_creator import FeatureCreator

from feature_binner import FeatureKBinner

from cleaner import Cleaner
from feature_type_transformer import FeatureTypeTransformer
from categorical_orderer import CategoricalOrderer

from pipelines import preprocessor_pipe, lasso_pipe, xgboost_pipe, lin_reg_pipe
from ml_training import grid_search
import pandas as pd

def main():
    data = DataSet()
    data.fetch_data()
    data.split_X_y()
    data.split_X_y_train_test()
    
    tmp_train_X = data.X_train
    tmp_train_y = data.y_train
    
    # full_pipe_df = preprocessor_pipe.fit_transform(tmp_train_X)
    # print(full_pipe_df)
        
    # res, bp, _ = grid_search(tmp_train_X, tmp_train_y, lasso_pipe, 
    #         param_grid={'lasso__alpha': [1, 0.1, 0.01,0.001]},
    #         cv=5, scoring='neg_mean_squared_error')
    # print(res)

    res, bp, _ = grid_search(tmp_train_X, tmp_train_y, xgboost_pipe, 
            param_grid={"preprocessor__creator__annum":[True,False],
                        "preprocessor__creator__disp_weight_ratio":[True,False],
                        "preprocessor__creator__power":[True,False],
                        "preprocessor__creator__brand":[True,False],
                        "xgboost__max_depth":[4,6,8,10,12,20],
                        "xgboost__gamma":[0.01,0.1,1],
                        "xgboost__n_estimators":[100,250,500,750,1000]},
            cv=5, scoring='neg_mean_squared_error')
    print(res)
    # res, bp, _ = grid_search(tmp_train_X, tmp_train_y, lasso_pipe, 
    #         param_grid={"preprocessor__creator__brand":[False]},
    #         cv=5, scoring='neg_mean_squared_error')
    # print(res)
    
if __name__ == "__main__":
    main()