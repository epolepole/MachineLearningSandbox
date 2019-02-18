# %% Import statements
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# %% Read input file

df = pd.read_csv('Churn_Modelling.csv')
df.head()
df.hist()
df.info()

# %% Set datatypes

binary_features = ["HasCrCard", 'IsActiveMember']
categorical_features = ["Geography", "Gender"]
numerical_features = ["CreditScore", 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

X = df.drop(["Exited"], axis=1)
X[binary_features] = X[binary_features].astype("category")

for f in categorical_features:
    X[f] = X[f].astype('category')

y = df['Exited'].astype('bool')

# %% Column transformer
preprocess_pipeline = make_column_transformer(
    (
        numerical_features, make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler())
    ),
    (
        categorical_features, make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(sparse=False))
    ),
    (
        binary_features, SimpleImputer(strategy="most_frequent")
    )
)
X_preproc = preprocess_pipeline.fit_transform(X)
# %% Process input

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
classifier_pipeline = make_pipeline(
    preprocess_pipeline,
    SVC(kernel="rbf", random_state=42)
)
# %%
param_grid = {
    "svc__gamma": [0.1 * x for x in range(1, 6)]
}
classifier_model = GridSearchCV(classifier_pipeline, param_grid, cv=10)

# %%
classifier_model.fit(X_train, y_train)

# %%
y_score = classifier_model.decision_function(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = roc_auc_score(y_test, y_score)

# %%
import matplotlib.pyplot as plt

# Plot ROC curve
plt.figure(figsize=(16, 12))
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)', size=16)
plt.ylabel('True Positive Rate (Sensitivity)', size=16)
plt.title('ROC Curve', size=20)
plt.legend(fontsize=14)
