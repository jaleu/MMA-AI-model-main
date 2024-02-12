import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was: 0.6837357014180304
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            make_union(
                FunctionTransformer(copy),
                FunctionTransformer(copy)
            ),
            RBFSampler(gamma=0.15000000000000002)
        ),
        make_union(
            FunctionTransformer(copy),
            FunctionTransformer(copy)
        )
    ),
    XGBClassifier(alpha=10, learning_rate=0.01, max_depth=3, min_child_weight=16, n_estimators=100, n_jobs=1, subsample=0.05, tree_method="gpu_hist", verbosity=0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
