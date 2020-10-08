from sklearn.pipeline import Pipeline

from cleaner import Cleaner
from feature_selector import FeatureSelector
from imputer import Imputer

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

