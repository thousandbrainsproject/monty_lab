# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import ast

from floppy.analysis.visitors.numpy_visitor import NumpyCallVisitor


def test_basic_numpy_imports() -> None:
    """Test basic numpy imports."""
    code = """
import numpy as np
from numpy import array, zeros
import numpy.linalg as la
from numpy.random import normal, rand
from numpy import *
"""
    tree = ast.parse(code)
    visitor = NumpyCallVisitor()
    visitor.visit(tree)

    imports = set(visitor.imports.values())
    assert "numpy" in imports
    assert "numpy.linalg" in imports
    assert "numpy.random.normal" in imports
    assert "numpy.random.rand" in imports
    assert any("array" in imp for imp in imports)
    assert any("zeros" in imp for imp in imports)


def test_basic_numpy_calls() -> None:
    """Test basic numpy calls."""
    code = """
import numpy as np
x = np.array([1, 2, 3])
y = np.zeros((2, 2))
"""
    tree = ast.parse(code)
    visitor = NumpyCallVisitor()
    visitor.visit(tree)

    calls = visitor.calls
    assert ("attribute", "numpy.array", 3) in calls
    assert ("attribute", "numpy.zeros", 4) in calls
    # Check variable tracking
    assert "x" in visitor.variables
    assert "y" in visitor.variables


def test_variable_tracking() -> None:
    """Test variable tracking."""
    code = """
import numpy as np
x = np.array([1, 2, 3])
y = x  # Assignment from numpy variable
z = y.transpose()  # Method call on numpy variable
w, v = np.zeros(2), np.ones(2)  # Multiple assignment
"""
    tree = ast.parse(code)
    visitor = NumpyCallVisitor()
    visitor.visit(tree)

    # Check variable tracking
    assert all(var in visitor.variables for var in ["x", "y", "z", "w", "v"])
    # Check method calls are tracked
    assert ("attribute", "numpy.transpose", 5) in visitor.calls
    assert ("attribute", "numpy.zeros", 6) in visitor.calls
    assert ("attribute", "numpy.ones", 6) in visitor.calls


def test_complex_numpy_usage() -> None:
    """Test complex numpy usage."""
    code = """
import numpy as np
from numpy import array, zeros
import numpy.linalg as la
from numpy.random import normal, rand as random_gen
import numpy.fft as fft
from numpy import *

class MyClass: # line 9
    def __init__(self):
        self.data = np.array([1,2,3])

    def process(self):
        return self.data.mean()

# Multiple chained calls
x = np.array([1,2,3]).reshape(3,1).transpose().sum()

# Nested with multiple numpy calls
y = np.dot(np.array([1,2]), np.zeros(2))

# Keyword arguments
z = np.full(shape=(3,3), fill_value=1)

# List comprehension with numpy
arrays = [np.zeros(i) for i in range(3)]

# Multiple assignments
a = b = np.ones(5)

# Unpacking
c, d = np.array([1,2]), np.array([3,4])
"""
    tree = ast.parse(code)
    visitor = NumpyCallVisitor()
    visitor.visit(tree)

    # Check imports
    imports = set(visitor.imports.values())
    assert "numpy" in imports
    assert "numpy.linalg" in imports
    assert "numpy.random.normal" in imports
    assert "numpy.fft" in imports

    # Check calls
    calls = visitor.calls
    # Check specific attribute calls
    assert ("attribute", "numpy.array", 11) in calls  # in __init__
    assert ("attribute", "numpy.array", 17) in calls  # in chained calls
    assert ("attribute", "numpy.array", 20) in calls  # first array in dot call
    assert ("attribute", "numpy.dot", 20) in calls
    assert ("attribute", "numpy.zeros", 20) in calls  # second arg in dot call
    assert ("attribute", "numpy.full", 23) in calls
    assert ("attribute", "numpy.ones", 29) in calls
    assert ("attribute", "numpy.array", 32) in calls  # first array in unpacking
    assert ("attribute", "numpy.array", 32) in calls  # second array in unpacking

    # Check method calls on numpy objects
    assert ("attribute", "numpy.reshape", 17) in calls  # Method call on array result
    assert (
        "attribute",
        "numpy.transpose",
        17,
    ) in calls  # Method call on reshape result
    assert ("attribute", "numpy.sum", 17) in calls  # Method call on transpose result

    # Check variable tracking for multiple assignments and unpacking
    assert all(var in visitor.variables for var in ["a", "b", "c", "d"])


def test_numpy_attribute_access() -> None:
    """Test numpy attribute access."""
    code = """
import numpy as np
x = np.array([1,2,3])
mean = x.mean()
std = x.std()
shape = x.shape
"""
    tree = ast.parse(code)
    visitor = NumpyCallVisitor()
    visitor.visit(tree)

    calls = visitor.calls
    assert ("attribute", "numpy.array", 3) in calls
    assert ("attribute", "numpy.mean", 4) in calls
    assert ("attribute", "numpy.std", 5) in calls
    assert ("attribute", "numpy.shape", 6) in calls  # Now consistent with numpy prefix
    # Check variable tracking
    assert "x" in visitor.variables
    assert "mean" in visitor.variables
    assert "std" in visitor.variables


def test_numpy_subscript_and_slice() -> None:
    """Test numpy subscript and slice."""
    code = """
import numpy as np
arr = np.array([[1,2,3], [4,5,6]])
slice1 = arr[0]
slice2 = arr[0:2]
slice3 = arr[0:2, 1:3]
"""
    tree = ast.parse(code)
    visitor = NumpyCallVisitor()
    visitor.visit(tree)
    calls = visitor.calls
    assert ("attribute", "numpy.array", 3) in calls
    assert (
        "attribute",
        "numpy.getitem",
        4,
    ) in calls  # Now consistent with numpy prefix
    assert (
        "attribute",
        "numpy.getitem",
        5,
    ) in calls  # Now consistent with numpy prefix
    assert (
        "attribute",
        "numpy.getitem",
        6,
    ) in calls  # Now consistent with numpy prefix
    # Check variable tracking
    assert all(
        var in visitor.variables for var in ["arr", "slice1", "slice2", "slice3"]
    )


def test_numpy_math_operations() -> None:
    """Test numpy math operations."""
    code = """
import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])
add = a + b
sub = a - b
mul = a * b
div = a / b
dot = np.dot(a, b)
matmul = a @ b
"""
    tree = ast.parse(code)
    visitor = NumpyCallVisitor()
    visitor.visit(tree)

    calls = visitor.calls
    assert ("attribute", "numpy.array", 3) in calls
    assert ("attribute", "numpy.array", 4) in calls
    assert ("attribute", "numpy.dot", 9) in calls
    # Check binary operations
    binary_ops = [
        call
        for call in calls
        if call[0] == "attribute" and "binary_operation" in call[1]
    ]
    assert len(binary_ops) == 5  # +, -, *, /, @ # noqa: PLR2004
    # Check variable tracking
    assert all(
        var in visitor.variables
        for var in ["a", "b", "add", "sub", "mul", "div", "dot", "matmul"]
    )


def test_numpy_with_error_handling() -> None:
    """Test numpy with error handling."""
    code = """
try:
    import numpy as np
    x = np.array([1,2,3])
    y = np.invalid_function()
except AttributeError:
    pass
"""
    tree = ast.parse(code)
    visitor = NumpyCallVisitor()
    visitor.visit(tree)

    calls = visitor.calls
    assert ("attribute", "numpy.array", 4) in calls
    assert ("attribute", "numpy.invalid_function", 5) in calls
    # Check variable tracking
    assert "x" in visitor.variables


def test_numpy_submodule_assignment() -> None:
    """Test numpy submodule assignment."""
    code = """
import numpy as np
linalg = np.linalg
fft = np.fft
x = np.array([1,2,3])
y = linalg.norm(x)
z = fft.fft(x)
"""
    tree = ast.parse(code)
    visitor = NumpyCallVisitor()
    visitor.visit(tree)

    # Check submodule imports are tracked
    assert "linalg" in visitor.imports
    assert "fft" in visitor.imports
    assert visitor.imports["linalg"] == "numpy.linalg"
    assert visitor.imports["fft"] == "numpy.fft"
    # Check calls through submodules
    assert ("attribute", "numpy.linalg.norm", 6) in visitor.calls
    assert ("attribute", "numpy.fft.fft", 7) in visitor.calls
    # Check variable tracking
    assert all(var in visitor.variables for var in ["x", "y", "z"])
