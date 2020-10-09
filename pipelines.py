from sklearn.pipeline import Pipeline

from cleaner import Cleaner
from feature_selector import FeatureSelector
from imputer import Imputer
from scaler import Scaler
from feature_type_transformer import FeatureTypeTransformer

numeric_preprocessor_pipeline = Pipeline(
    steps = [
        ("feature_selection",FeatureSelector(feature_type="numeric")),
        ("feature_impute",Imputer(method="median")),
    ]
)

cleaner_numeric_preprocessor_pipeline = Pipeline(
    steps=[
        ("clean_pipe",Cleaner()),
        ("numeric_pipe",numeric_preprocessor_pipeline)
    ]
)
scaler_preprocessor_pipeline = Pipeline(
    steps =[
        ("feat_sel", FeatureSelector(feature_type="numeric",feature_list=["cylinders"])),
        ("scale", Scaler(method="standard"))
    ]
)

full_pipeline = Pipeline(
    steps=[
        ("cleaner_num_pipe",cleaner_numeric_preprocessor_pipeline),
        ("scaler",scaler_preprocessor_pipeline)
    ]
)


feat_type_transformer_pipeline = Pipeline(
    steps=[
        ("feat_sel", FeatureSelector(feature_type="numeric",feature_list=["cylinders","origin"])),
        ("feat_type_transf",FeatureTypeTransformer(feature_list=["cylinders"])),
        # ("feat_sel1", FeatureSelector(feature_type="categorical",feature_list=["cylinders"])),        # for testing purposes
        # ("feat_type_transf1",FeatureTypeTransformer(feature_list=["cylinders"], num_to_str=False))    # for testing purposes
    ]
)