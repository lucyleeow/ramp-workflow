import os
import pandas as pd

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


def _merge_external_data(X):
    filepath = os.path.join(os.path.dirname(__file__),
                            'external_data_mini.csv')
    X_weather = pd.read_csv(filepath)
    X_merged = pd.merge(X, X_weather, how='left',
                        on=['DateOfDeparture', 'Arrival'], sort=False)
    return X_merged


def get_component():
    merger = FunctionTransformer(func=_merge_external_data)

    categorical_cols = ['Arrival', 'Departure']
    numerical_cols = ['WeeksToDeparture', 'std_wtd', 'Max TemperatureC']
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        (FunctionTransformer(), numerical_cols),
    )

    return make_pipeline(merger, preprocessor)
