import ast

import pytest

from floppy.analysis.visitors.scipy_visitor import ScipyCallVisitor


def test_basic_scipy_imports():
    code = """
import scipy as sp
from scipy import stats
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy import *
"""
    tree = ast.parse(code)
    visitor = ScipyCallVisitor()
    visitor.visit(tree)

    imports = visitor.scipy_imports
    assert "scipy" in imports
    assert "scipy.stats" in imports
    assert "scipy.linalg" in imports
    assert "scipy.optimize" in imports
    assert "scipy.sparse" in imports


def test_scipy_optimization():
    code = """
from scipy.optimize import minimize, fmin
import numpy as np

def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

x0 = np.array([0, 0])
res = minimize(objective, x0, method='Nelder-Mead')
res2 = fmin(objective, x0)
"""
    tree = ast.parse(code)
    visitor = ScipyCallVisitor()
    visitor.visit(tree)

    calls = visitor.scipy_calls
    assert ("call", "minimize", 8) in calls
    assert ("call", "fmin", 9) in calls


def test_scipy_stats():
    code = """
from scipy import stats
import numpy as np

data = np.random.randn(100)
ks_stat, p_value = stats.kstest(data, 'norm')
t_stat, t_p_value = stats.ttest_1samp(data, 0)
norm_test = stats.normaltest(data)
"""
    tree = ast.parse(code)
    visitor = ScipyCallVisitor()
    visitor.visit(tree)

    calls = visitor.scipy_calls
    assert ("call", "kstest", 5) in calls
    assert ("call", "ttest_1samp", 6) in calls
    assert ("call", "normaltest", 7) in calls


def test_scipy_linalg():
    code = """
import scipy.linalg as la
import numpy as np

A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

x = la.solve(A, b)
eigenvals = la.eigvals(A)
det = la.det(A)
inv = la.inv(A)
"""
    tree = ast.parse(code)
    visitor = ScipyCallVisitor()
    visitor.visit(tree)

    calls = visitor.scipy_calls
    assert ("call", "solve", 7) in calls
    assert ("call", "eigvals", 8) in calls
    assert ("call", "det", 9) in calls
    assert ("call", "inv", 10) in calls


def test_scipy_sparse():
    code = """
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np

data = np.array([1, 2, 3])
row = np.array([0, 0, 1])
col = np.array([0, 2, 1])
sparse_matrix = csr_matrix((data, (row, col)), shape=(2, 3))
lil = lil_matrix((4, 4))
"""
    tree = ast.parse(code)
    visitor = ScipyCallVisitor()
    visitor.visit(tree)

    calls = visitor.scipy_calls
    assert ("call", "csr_matrix", 7) in calls
    assert ("call", "lil_matrix", 8) in calls


def test_scipy_signal():
    code = """
from scipy import signal
import numpy as np

t = np.linspace(0, 1, 1000)
sig = np.sin(2 * np.pi * 10 * t)
filtered = signal.butter(4, 0.2)
windowed = signal.windows.hamming(100)
peaks = signal.find_peaks(sig)
"""
    tree = ast.parse(code)
    visitor = ScipyCallVisitor()
    visitor.visit(tree)

    calls = visitor.scipy_calls
    assert ("call", "butter", 6) in calls
    assert ("call", "hamming", 7) in calls
    assert ("call", "find_peaks", 8) in calls


def test_scipy_interpolate():
    code = """
from scipy.interpolate import interp1d, UnivariateSpline
import numpy as np

x = np.linspace(0, 10, 10)
y = np.sin(x)
f = interp1d(x, y)
spline = UnivariateSpline(x, y)
"""
    tree = ast.parse(code)
    visitor = ScipyCallVisitor()
    visitor.visit(tree)

    calls = visitor.scipy_calls
    assert ("call", "interp1d", 6) in calls
    assert ("call", "UnivariateSpline", 7) in calls
