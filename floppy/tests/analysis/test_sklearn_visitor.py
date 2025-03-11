import ast

import pytest

from floppy.analysis.visitors.sklearn_visitor import SklearnCallVisitor


def test_basic_sklearn_imports():
    code = """
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
"""
    tree = ast.parse(code)
    visitor = SklearnCallVisitor()
    visitor.visit(tree)

    imports = visitor.sklearn_imports
    assert "sklearn" in imports
    assert "sklearn.datasets" in imports
    assert "sklearn.model_selection" in imports
    assert "sklearn.preprocessing" in imports
    assert "sklearn.ensemble" in imports
    assert "sklearn.metrics" in imports


def test_sklearn_preprocessing():
    code = """
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

X = np.random.randn(100, 4)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)
"""
    tree = ast.parse(code)
    visitor = SklearnCallVisitor()
    visitor.visit(tree)

    calls = visitor.sklearn_calls
    assert ("call", "StandardScaler", 5) in calls
    assert ("call", "MinMaxScaler", 8) in calls
    assert ("attribute", "fit_transform", 6) in visitor.sklearn_attributes
    assert ("attribute", "fit_transform", 9) in visitor.sklearn_attributes


def test_sklearn_model_selection():
    code = """
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.random.randn(100, 4)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
scores = cross_val_score(clf, X, y, cv=5)

param_grid = {'n_estimators': [10, 20, 30]}
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X, y)
"""
    tree = ast.parse(code)
    visitor = SklearnCallVisitor()
    visitor.visit(tree)

    calls = visitor.sklearn_calls
    assert ("call", "train_test_split", 8) in calls
    assert ("call", "RandomForestClassifier", 9) in calls
    assert ("call", "cross_val_score", 10) in calls
    assert ("call", "GridSearchCV", 13) in calls


def test_sklearn_pipeline():
    code = """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

pipe.fit(X, y)
predictions = pipe.predict(X_test)
"""
    tree = ast.parse(code)
    visitor = SklearnCallVisitor()
    visitor.visit(tree)

    calls = visitor.sklearn_calls
    assert ("call", "Pipeline", 5) in calls
    assert ("call", "StandardScaler", 6) in calls
    assert ("call", "SVC", 7) in calls
    assert ("attribute", "fit", 10) in visitor.sklearn_attributes
    assert ("attribute", "predict", 11) in visitor.sklearn_attributes


def test_sklearn_metrics():
    code = """
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
"""
    tree = ast.parse(code)
    visitor = SklearnCallVisitor()
    visitor.visit(tree)

    calls = visitor.sklearn_calls
    assert ("call", "accuracy_score", 5) in calls
    assert ("call", "precision_score", 6) in calls
    assert ("call", "recall_score", 7) in calls
    assert ("call", "confusion_matrix", 8) in calls
    assert ("call", "classification_report", 9) in calls


def test_sklearn_clustering():
    code = """
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)

dbscan = DBSCAN(eps=0.5)
db_labels = dbscan.fit_predict(X)

gmm = GaussianMixture(n_components=3)
gmm_labels = gmm.fit_predict(X)
"""
    tree = ast.parse(code)
    visitor = SklearnCallVisitor()
    visitor.visit(tree)

    calls = visitor.sklearn_calls
    assert ("call", "KMeans", 4) in calls
    assert ("call", "DBSCAN", 7) in calls
    assert ("call", "GaussianMixture", 10) in calls
