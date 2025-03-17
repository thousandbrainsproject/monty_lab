# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import ast

from floppy.analysis.visitors.sklearn_visitor import SklearnCallVisitor


def test_basic_sklearn_imports() -> None:
    """Test basic sklearn imports."""
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

    imports = set(visitor.imports.values())
    assert "sklearn" in imports
    assert "sklearn.datasets" in imports
    assert "sklearn.model_selection.train_test_split" in imports
    assert "sklearn.preprocessing.StandardScaler" in imports
    assert "sklearn.ensemble.RandomForestClassifier" in imports
    assert "sklearn.metrics.accuracy_score" in imports


def test_sklearn_preprocessing() -> None:
    """Test sklearn preprocessing."""
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

    calls = visitor.calls
    assert ("direct", "sklearn.preprocessing.StandardScaler", 6) in calls
    assert ("direct", "sklearn.preprocessing.MinMaxScaler", 9) in calls
    assert ("attribute", "sklearn.fit_transform", 7) in calls
    assert ("attribute", "sklearn.fit_transform", 10) in calls


def test_sklearn_model_selection() -> None:
    """Test sklearn model selection."""
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

    calls = visitor.calls
    assert ("direct", "sklearn.model_selection.train_test_split", 9) in calls
    assert ("direct", "sklearn.ensemble.RandomForestClassifier", 10) in calls
    assert ("direct", "sklearn.model_selection.cross_val_score", 11) in calls
    assert ("direct", "sklearn.model_selection.GridSearchCV", 14) in calls
    assert ("attribute", "sklearn.fit", 15) in calls


def test_sklearn_pipeline() -> None:
    """Test sklearn pipeline."""
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

    calls = visitor.calls
    assert ("direct", "sklearn.pipeline.Pipeline", 6) in calls
    assert ("direct", "sklearn.preprocessing.StandardScaler", 7) in calls
    assert ("direct", "sklearn.svm.SVC", 8) in calls
    assert ("attribute", "sklearn.fit", 11) in calls
    assert ("attribute", "sklearn.predict", 12) in calls


def test_sklearn_metrics() -> None:
    """Test sklearn metrics."""
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

    calls = visitor.calls
    assert ("direct", "sklearn.metrics.accuracy_score", 6) in calls
    assert ("direct", "sklearn.metrics.precision_score", 7) in calls
    assert ("direct", "sklearn.metrics.recall_score", 8) in calls
    assert ("direct", "sklearn.metrics.confusion_matrix", 9) in calls
    assert ("direct", "sklearn.metrics.classification_report", 10) in calls


def test_sklearn_clustering() -> None:
    """Test sklearn clustering."""
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

    calls = visitor.calls
    assert ("direct", "sklearn.cluster.KMeans", 5) in calls
    assert ("direct", "sklearn.cluster.DBSCAN", 8) in calls
    assert ("direct", "sklearn.mixture.GaussianMixture", 11) in calls
    assert ("attribute", "sklearn.fit_predict", 6) in calls
    assert ("attribute", "sklearn.fit_predict", 9) in calls
    assert ("attribute", "sklearn.fit_predict", 12) in calls
