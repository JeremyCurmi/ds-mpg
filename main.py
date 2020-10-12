import pandas as pd
from sklearn.metrics import mean_squared_error

from dataset import data_fetcher
from pipelines import lasso_pipe, xgboost_pipe
from ml_training import grid_search, save_ml_model, load_ml_model

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

RUN_MODEL = False

def main():
    X_train, X_test, y_train, y_test = data_fetcher()
    
    if RUN_MODEL:
        # res, bp, model = grid_search(X_train, y_train, lasso_pipe, 
        #         param_grid={'lasso__alpha': [1, 0.1, 0.01,0.001]},
        #         cv=5, scoring='neg_mean_squared_error')
        # print(res)


        res, bp, model = grid_search(X_train, y_train, xgboost_pipe, 
                param_grid={"preprocessor__creator__annum":[True],
                            "preprocessor__creator__disp_weight_ratio":[False],
                            "preprocessor__creator__power":[False],
                            "preprocessor__creator__brand":[False],
                            "xgboost__max_depth":[4,6,8],
                            "xgboost__gamma":[0.1,1],
                            "xgboost__n_estimators":[100]},
                cv=5, scoring='neg_mean_squared_error')
        print(res)


        save_ml_model(model, "algorithm")

    alg = load_ml_model("algorithm")
    y_pred = alg.predict(X_test)

    print(mean_squared_error(y_test,y_pred))
    
if __name__ == "__main__":
    main()