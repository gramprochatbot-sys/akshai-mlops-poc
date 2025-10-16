from prefect import flow, task
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@task
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    return X, y


@task
def train(X, y, n_estimators=50):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "prefect_rf.joblib")
    return clf, X_test, y_test


@task
def evaluate(clf, X_test, y_test):
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Prefect flow accuracy:", acc)
    return acc


@flow
def train_flow(n_estimators=50):
    X, y = load_data()
    clf, X_test, y_test = train(X, y, n_estimators=n_estimators)
    acc = evaluate(clf, X_test, y_test)
    return acc


# Run flow locally
train_flow(80)
