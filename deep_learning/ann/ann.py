# %% Read input file
import os

import pandas as pd

df = pd.read_csv(os.path.join(os.getcwd(), 'deep_learning', 'ann', 'data', 'Churn_Modelling.csv'))
df.head()
df.hist()
df.info()

# %% Set datatypes

binary_features = ["HasCrCard", 'IsActiveMember']
categorical_features = ["Geography", "Gender"]
numerical_features = ["CreditScore", 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

X = df.drop(["Exited"], axis=1).drop(["RowNumber"], axis=1).drop(["CustomerId"], axis=1).drop(["Surname"], axis=1)
y = df['Exited'].astype('bool')

# %% Data preparation pipeline
from tools.pipes import type_transformer

preprocess_pipeline = type_transformer(numerical_features, categorical_features, binary_features)
X_transformed = preprocess_pipeline.fit_transform(X)

# %% Split in train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Create the svc model pipeline
from tools.models import create_svc_model

classifier_model = create_svc_model(preprocess_pipeline)

# %% Create the linear regression model pipeline
from tools.models import create_lr_model

classifier_model = create_lr_model(preprocess_pipeline)

# %% Create the ann model pipeline
from tools.models import create_ann_model

classifier_model = create_ann_model(preprocess_pipeline)

# %% Fit the model
classifier_model.fit(X_train, y_train)

# %% Show model accuracy
from tools.errors import plot_roc_curve

y_score = classifier_model.decision_function(X_test)
plot_roc_curve(y_score, y_test)
