import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.base import is_classifier

from ..utils.importing import import_file


class SKLearnPipeline(object):
    def __init__(self, *filenames):
        self.filenames = filenames

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        # import files
        components = [import_file(module_path, filename)
                      for filename in self.filenames]
        pipeline = make_pipeline(
            *[item.get_component() for item in components]
        )
        if isinstance(X_df, pd.DataFrame):
            X_train = X_df.iloc[train_is]
        else:
            X_train = X_df[train_is]
        return pipeline.fit(X_train, y_array[train_is])

    def test_submission(self, trained_model, X_df):
        if is_classifier(trained_model):
            return trained_model.predict_proba(X_df)
        return trained_model.predict(X_df)
