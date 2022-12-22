from typing import List
import pandas as pd
from sklearn.base import ClassifierMixin
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score


class CustomEnsemble:
    def __init__(self, classifiers: list, final_classifier,
                 retain_features: bool = True,
                 verbose: int = 0,
                 weak_learners: bool = True):
        self.ensemble = final_classifier
        self.classifiers = classifiers
        self.retain_features = retain_features
        self.verbose = verbose
        self.weak_learners = weak_learners

    def sample(self, x, y, frac: float = 0.5):
        temp = x.copy()
        temp["y"] = y
        temp = temp.sample(frac=frac)
        y = temp["y"]
        temp = temp.drop("y", axis=1)
        return temp, y

    def _get_sample_weights(self, y):
        total_cases = len(y)
        positive_cases = sum(y)
        negative_weight = positive_cases/total_cases
        positive_weight = 1-negative_weight
        weights = [
            negative_weight, positive_weight
        ]
        return [weights[i] for i in y]

    def _append_features(self, x):
        features = pd.DataFrame()
        for i, classifier in enumerate(self.classifiers):
            if self.verbose:
                print("predicting for classifier classifier", i, end="\r")
            feature_name = "score_" + str(i)
            predictions = pd.DataFrame(classifier.predict_proba(x))
            features[feature_name] = predictions[predictions.columns[-1]]
        return features

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        new_features = pd.DataFrame()
        weights = self._get_sample_weights(y)
        x.reset_index(inplace=True)
        x.drop('index', inplace=True, axis=1)
        y = y.reset_index()
        y = y["ordered"]

        for i, classifier in enumerate(self.classifiers):
            if self.verbose:
                print("training classifier", i, end="\r")
            if self.weak_learners:
                temp_x, temp_y = self.sample(x, y)
            else:
                temp_x = x
                temp_y = y

            feature_name = "score_" + str(i)
            classifier.fit(temp_x, temp_y)
            predictions = pd.DataFrame(classifier.predict_proba(x))
            new_features[feature_name] = predictions[predictions.columns[-1]]

        # now train final models
        if self.retain_features:
            new_features = pd.concat([x.copy(), new_features], axis=1)
        if self.verbose:
            print("training output classifier", end="\r")
        self.ensemble.fit(new_features, y)

        return self

    def predict_proba(self, x: pd.DataFrame):
        x.reset_index(inplace=True)
        x.drop('index', inplace=True, axis=1)

        features = self._append_features(x)
        if self.retain_features:
            features = pd.concat([x.copy(), features], axis=1)
        if self.verbose:
            print("predicting final score", end="\r")

        print(features.columns)
        return self.ensemble.predict_proba(features)

    def predict(self, x):
        features = self._append_features(x)

        if self.retain_features:
            features = pd.concat([x.copy(), features], axis=1)
        if self.verbose:
            print("predicting final score", end="\r")
        return self.ensemble.predict(features)

    def score(self, x, y):
        predictions = self.predict(x)
        return recall_score(y_true=y, y_pred=predictions)

    def feature_importances_(self, only_internal: bool = False):
        if not isinstance(self.ensemble, lgbm.LGBMModel):
            print("output model does not support feature importances")
            return dict()
        names = self.ensemble.feature_name_
        importances = self.ensemble.feature_importances_
        if only_internal:
            count = len(self.classifiers)
            names = names[-count:]
            importances = importances[-count:]

        m = sum(importances)
        result = dict()
        for i in range(len(names)):
            result[names[i]] = (importances[i]/m)
        return result


def make_model(df: pd.DataFrame, retain_features: bool = True,
               verbose: int = 0) -> CustomEnsemble:
    output_model = lgbm.LGBMClassifier(n_estimators=50, n_jobs=-1)
    df_columns = df.columns
    # removed models that do not allow specifying jobs
    submodels = [
        lgbm.LGBMClassifier(n_estimators=10, n_jobs=-1),
        lgbm.LGBMClassifier(n_estimators=15, n_jobs=-1),
        lgbm.LGBMClassifier(n_estimators=20, n_jobs=-1),
        MLPClassifier(hidden_layer_sizes=[len(df_columns) for i in range(3)]),
        RandomForestClassifier(n_estimators=10, n_jobs=-1),
        GaussianNB()
    ]

    return CustomEnsemble(submodels, output_model,
                          retain_features=retain_features,
                          verbose=verbose)