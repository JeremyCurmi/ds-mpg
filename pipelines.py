import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector
from sklearn.linear_model import Lasso, LinearRegression
from xgboost import XGBRegressor
from cleaner import Cleaner

from feature_selector import FeatureSelector
from feature_creator import FeatureCreator
from feature_type_transformer import FeatureTypeTransformer
from feature_joiner import FeatureUnioner, FeatureColumnTransformer
from feature_remover import FeatureRemover
from feature_binner import FeatureKBinner
from feature_ordering import FeatureOrderer

from imputer import Imputer
from scaler import Scaler
from categorical_recoder import CategoricalRecoder
from categorical_encoder import CategoricalEncoder


num_feature_mapper = {
    "brand":{
    "chevroelt":"chevrolet",
    "chevy":"chevrolet",
    "maxda":"mazda",
    "mercedes-benz":"mercedes",
    "vokswagen":"volkswagen",
    "vw":"volkswagen",
    "toyouta":"toyota"
    },
    "origin":{1:"US",2:"EU",3:"Japan"},
    "cylinders":{3:6,5:6}
}

impute_lvl1_list = ["horsepower"]

lvl1_imputer_col_transf_pipe = FeatureColumnTransformer(
    transformers=[    
        ("impute1",Imputer(method="median"),impute_lvl1_list)
    ],
    remainder="passthrough",
    n_jobs=-1
)

scaler_pipe = FeatureColumnTransformer(
    transformers=[    
        ("scaler",Scaler(method="minmax"),make_column_selector(dtype_include=np.number))
    ],
    remainder="passthrough",
    n_jobs=-1
)

dummifier_pipe = FeatureColumnTransformer(
    transformers=[    
        ("encoder",CategoricalEncoder(drop_first=False),make_column_selector(dtype_include=object))
    ],
    remainder="passthrough",
    n_jobs=-1
)

# binner_pipe = FeatureColumnTransformer(
#     transformers=[    
#         ("binner",FeatureKBinner(),["weight"])
#     ],
#     remainder="passthrough",
#     n_jobs=-1
# )

preprocessor_pipe = Pipeline(
    steps=[
        ("cleaner",Cleaner()),
        ("impute_lvl1_features", lvl1_imputer_col_transf_pipe),
        ("creator",FeatureCreator()),
        ("recoder",CategoricalRecoder(feature_mapper=num_feature_mapper)),
        ("type_transformer_num",FeatureTypeTransformer(feature_list=["cylinders"],num_to_str=True)),
        ("remover",FeatureRemover(feature_list=["car name","model year"])),
        ("scaler",scaler_pipe),
        ("dummifier",dummifier_pipe),
        ("order",FeatureOrderer())
    ]
)

lasso_pipe = Pipeline(
    steps=[
        ("preprocessor",preprocessor_pipe),
        ("lasso",Lasso())
    ]
)

lin_reg_pipe = Pipeline(
    steps=[
        ("preprocessor",preprocessor_pipe),
        ("lin_reg",LinearRegression())
    ]
)
    
xgboost_pipe = Pipeline(
    steps=[
        ("preprocessor",preprocessor_pipe),
        ("xgboost",XGBRegressor())
    ]
)