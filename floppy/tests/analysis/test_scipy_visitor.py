# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import ast

from floppy.analysis.visitors.scipy_visitor import ScipyCallVisitor


def test_basic_scipy_imports() -> None:
    """Test basic scipy imports."""
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

    imports = set(visitor.imports.values())
    assert "scipy" in imports
    assert "scipy.stats" in imports
    assert "scipy.linalg" in imports
    assert "scipy.optimize.minimize" in imports
    assert "scipy.sparse.csr_matrix" in imports


def test_scipy_optimization() -> None:
    """Test scipy optimization."""
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

    calls = visitor.calls
    assert ("direct", "scipy.optimize.minimize", 9) in calls
    assert ("direct", "scipy.optimize.fmin", 10) in calls


def test_scipy_stats() -> None:
    """Test scipy stats."""
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

    calls = visitor.calls
    assert ("attribute", "scipy.stats.kstest", 6) in calls
    assert ("attribute", "scipy.stats.ttest_1samp", 7) in calls
    assert ("attribute", "scipy.stats.normaltest", 8) in calls


def test_scipy_attribute_access() -> None:
    """Test scipy attribute access."""
    code = """
from scipy import stats
import numpy as np

data = np.random.randn(100)
result = stats.norm.pdf(data)
params = stats.norm.fit(data)
entropy = stats.norm.entropy()
"""
    tree = ast.parse(code)
    visitor = ScipyCallVisitor()
    visitor.visit(tree)

    calls = visitor.calls
    assert ("attribute", "scipy.stats.norm.pdf", 6) in calls
    assert ("attribute", "scipy.stats.norm.fit", 7) in calls
    assert ("attribute", "scipy.stats.norm.entropy", 8) in calls


def test_scipy_with_error_handling() -> None:
    """Test scipy with error handling."""
    code = """
try:
    from scipy import special
    x = special.gamma(5)
    y = special.invalid_function()
except AttributeError:
    pass
"""
    tree = ast.parse(code)
    visitor = ScipyCallVisitor()
    visitor.visit(tree)

    calls = visitor.calls
    assert ("attribute", "scipy.special.gamma", 4) in calls
    assert ("attribute", "scipy.special.invalid_function", 5) in calls


def test_scipy_linalg() -> None:
    """Test scipy linalg."""
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

    calls = visitor.calls
    assert ("attribute", "scipy.linalg.solve", 8) in calls
    assert ("attribute", "scipy.linalg.eigvals", 9) in calls
    assert ("attribute", "scipy.linalg.det", 10) in calls
    assert ("attribute", "scipy.linalg.inv", 11) in calls


def test_scipy_sparse() -> None:
    """Test scipy sparse."""
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

    calls = visitor.calls
    assert ("direct", "scipy.sparse.csr_matrix", 8) in calls
    assert ("direct", "scipy.sparse.lil_matrix", 9) in calls


def test_scipy_signal() -> None:
    """Test scipy signal."""
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

    calls = visitor.calls
    assert ("attribute", "scipy.signal.butter", 7) in calls
    assert ("attribute", "scipy.signal.windows.hamming", 8) in calls
    assert ("attribute", "scipy.signal.find_peaks", 9) in calls


def test_scipy_interpolate() -> None:
    """Test scipy interpolate."""
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

    calls = visitor.calls
    assert ("direct", "scipy.interpolate.interp1d", 7) in calls
    assert ("direct", "scipy.interpolate.UnivariateSpline", 8) in calls


def test_scipy_rotation() -> None:
    """Test scipy rotation."""
    code = """
from scipy.spatial.transform import Rotation as R
import numpy as np

# Create rotation from various inputs
rot1 = R.from_euler('xyz', [90, 45, 30], degrees=True)
rot2 = R.from_quat([0, 0, 0, 1])
rot3 = R.from_matrix(np.eye(3))

# Method chaining and operations
angles = rot1.as_euler('xyz', degrees=True)
quat = rot1.as_quat()
matrix = rot1.as_matrix()

# Composition of rotations
combined = rot1 * rot2
inverse = rot1.inv()

# Apply rotation to vectors
vec = np.array([1, 0, 0])
rotated = rot1.apply(vec)
"""
    tree = ast.parse(code)
    visitor = ScipyCallVisitor()
    visitor.visit(tree)

    calls = visitor.calls
    # Class methods (keep full path)
    assert ("attribute", "scipy.spatial.transform.Rotation.from_euler", 6) in calls
    assert ("attribute", "scipy.spatial.transform.Rotation.from_quat", 7) in calls
    assert ("attribute", "scipy.spatial.transform.Rotation.from_matrix", 8) in calls

    # Instance methods (use simple scipy.method_name format)
    assert ("attribute", "scipy.as_euler", 11) in calls
    assert ("attribute", "scipy.as_quat", 12) in calls
    assert ("attribute", "scipy.as_matrix", 13) in calls
    assert ("attribute", "scipy.inv", 17) in calls
    assert ("attribute", "scipy.apply", 21) in calls


def test_scipy_rotation_interpolation() -> None:
    """Test scipy rotation interpolation."""
    code = """
from scipy.spatial.transform import Rotation as R
import numpy as np

# Create some rotations
key_rots = R.from_euler('xyz', [[0, 0, 0], [90, 0, 0]], degrees=True)
key_times = [0, 1]

# Interpolation
slerp = key_rots.slerp(key_times)

# Get the interpolated rotation in different formats
angles = key_rots.as_euler('xyz', degrees=True)
quat = key_rots.as_quat()
"""
    tree = ast.parse(code)
    visitor = ScipyCallVisitor()
    visitor.visit(tree)

    calls = visitor.calls
    # Class methods (keep full path)
    assert ("attribute", "scipy.spatial.transform.Rotation.from_euler", 6) in calls

    # Instance methods (use simple scipy.method_name format)
    assert ("attribute", "scipy.slerp", 10) in calls
    assert ("attribute", "scipy.as_euler", 13) in calls
    assert ("attribute", "scipy.as_quat", 14) in calls
