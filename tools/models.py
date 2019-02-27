from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


def create_svc_model(preprocess_pipeline):
    classifier_pipeline = make_pipeline(
        preprocess_pipeline,
        SVC(kernel="rbf", random_state=42)
    )

    return GridSearchCV(
        classifier_pipeline,
        {"svc__gamma": [0.1 * x for x in range(1, 6)]},
        cv=100, verbose=10, n_jobs=-1
    )


def create_lr_model(preprocess_pipeline):
    classifier_pipeline = make_pipeline(
        preprocess_pipeline,
        LogisticRegression(n_jobs=-1, random_state=42, solver='newton-cg')
    )

    return GridSearchCV(
        classifier_pipeline,
        {"logisticregression__C": [0.5 * x for x in range(1, 6)]},
        cv=100, verbose=10, n_jobs=-1
    )


def create_ann_model(preprocess_pipeline):
    ann_classifier = Sequential()
    ann_classifier.add(Dense(units=6, init='uniform', activation='relu', imput_dim='12'))
    ann_classifier.add(Dense(units=6, init='uniform', activation='relu'))
    ann_classifier.add(Dense(units=1, init='uniform', activation='sigmoid'))

    ann_classifier.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'])

    return make_pipeline(
        preprocess_pipeline,
        ann_classifier
    )

    # return GridSearchCV(
    #     classifier_pipeline,
    #     {"logisticregression__C": [0.5 * x for x in range(1, 6)]},
    #     cv=100, verbose=10, n_jobs=-1
    # )
