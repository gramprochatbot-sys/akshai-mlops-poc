"""
prefect workflow
"""
from prefect import flow, task
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import Bunch


@task
def load_data():
    """
    data load flow
    """
    iris : Bunch = load_iris(as_frame=True)
    # pylint: disable=no-member
    df = iris.frame
    x = df.drop(columns=["target"]).values
    y = df["target"].values
    return x, y


@task
def train(x, y, n_estimators=50):
    """
    model train flow
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1
    )
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
    clf.fit(x_train, y_train)
    joblib.dump(clf, "prefect_rf.joblib")
    return clf, x_test, y_test


@task
def evaluate(clf, x_test, y_test):
    """
    data evaluation flow
    """
    preds = clf.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print("Prefect flow accuracy:", acc)
    return acc


@flow
def train_flow(n_estimators=50):
    """
    main work flow
    """
    x, y = load_data()
    clf, x_test, y_test = train(x, y, n_estimators=n_estimators)
    acc = evaluate(clf, x_test, y_test)
    return acc


# Run flow locally
train_flow(80)
