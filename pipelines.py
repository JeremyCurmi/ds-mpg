from sklearn.pipeline import Pipeline

from cleaner import Cleaner

from feature_selector import FeatureSelector
from feature_creator import FeatureCreator
from feature_type_transformer import FeatureTypeTransformer

from imputer import Imputer
from scaler import Scaler
from categorical_recoder import CategoricalRecoder
from categorical_encoder import CategoricalEncoder

# numeric_preprocessor_pipeline = Pipeline(
#     steps = [
#         ("feature_selection",FeatureSelector(feature_type="numeric")),
#         ("feature_impute",Imputer(method="median")),
#     ]
# )

# cleaner_numeric_preprocessor_pipeline = Pipeline(
#     steps=[
#         ("clean_pipe",Cleaner()),
#         ("numeric_pipe",numeric_preprocessor_pipeline)
#     ]
# )
# scaler_preprocessor_pipeline = Pipeline(
#     steps =[
#         ("feat_sel", FeatureSelector(feature_type="numeric",feature_list=["cylinders"])),
#         ("scale", Scaler(method="standard"))
#     ]
# )

# full_pipeline = Pipeline(
#     steps=[
#         ("cleaner_num_pipe",cleaner_numeric_preprocessor_pipeline),
#         ("scaler",scaler_preprocessor_pipeline)
#     ]
# )


# feat_type_transformer_pipeline = Pipeline(
#     steps=[
#         ("feat_sel", FeatureSelector(feature_type="numeric",feature_list=["cylinders","origin"])),
#         ("feat_type_transf",FeatureTypeTransformer(feature_list=["cylinders"])),
#         # ("feat_sel1", FeatureSelector(feature_type="categorical",feature_list=["cylinders"])),        # for testing purposes
#         # ("feat_type_transf1",FeatureTypeTransformer(feature_list=["cylinders"], num_to_str=False))    # for testing purposes
#     ]
# )
cleaner_pipe = Pipeline(
    steps=[
        ("cleaner",Cleaner()),
    ]
)
impute_cleaner = Pipeline(
    steps=[
        ("feat_sel",FeatureSelector(feature_type="numeric",feature_list=["horsepower"])),
        ("impute",Imputer(method="median"))
    ]
)
creator_pipe = Pipeline(
    steps =[  
        ("creator",FeatureCreator()),
    ]
)
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
recode_pipe = Pipeline(
    steps=[
        ("recoder",CategoricalRecoder(feature_mapper=num_feature_mapper))
    ]
)

type_transformer_pipe = Pipeline(steps=[
    ("type_transformer",FeatureTypeTransformer(feature_list=["cylinders"],num_to_str=True))
])

num_feature_scale_pipe = Pipeline(
    steps=[
        ("feat_sel",FeatureSelector()),
        ("scaler", Scaler())
    ]
)

cat_feature_encoder_pipe = Pipeline(
    steps=[
        ("feat_sel",FeatureSelector(feature_type="categorical")),
        ("dummifier",CategoricalEncoder())
    ]
)

full_pipe = Pipeline(
    steps=[
        ("cleaner", cleaner_pipe),
        # ("imputer1",impute_cleaner)
        ("creator",creator_pipe),
        ("numeric_processor",recode_pipe),
        ("num_to_str_transformer",type_transformer_pipe),
        ("scaler",num_feature_scale_pipe)
    ]
)