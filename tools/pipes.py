from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DummyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, sparse=True):
        self.ohe = OneHotEncoder(sparse)

    def fit(self, X, y=None):
        return self

    def get_idx_to_keep(self, shape):
        idx_to_delete = list()
        curr_idx = -1
        for cat in self.ohe.categories_:
            curr_idx = curr_idx + len(cat)
            idx_to_delete.append(curr_idx)

        idx_to_keep = [i for i in range(shape[-1]) if i not in idx_to_delete]

        print("idx_to_delete: {}".format(idx_to_delete))
        print("idx_to_keep: {}".format(idx_to_keep))
        return idx_to_keep

    def transform(self, X):
        old_x = self.ohe.fit_transform(X)
        idx_to_keep = self.get_idx_to_keep(old_x.shape)
        new_x = old_x[:, idx_to_keep].astype('int')
        print("old_x shape: {}".format(old_x.shape))
        print("new_x shape: {}".format(new_x.shape))

        return old_x


class DataTypeTransform(BaseEstimator, TransformerMixin):
    def __init__(self, datatype):
        self.datatype = datatype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(self.datatype)


def type_transformer(numerical_features, categorical_features, binary_features):
    return make_column_transformer(
        (
            make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler()
            ),
            numerical_features
        ),
        (
            make_pipeline(
                DataTypeTransform('category'),
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(sparse=False)
            ),
            categorical_features
        ),
        (
            make_pipeline(
                DataTypeTransform('category'),
                SimpleImputer(strategy="most_frequent")
            ),
            binary_features
        )
    )
