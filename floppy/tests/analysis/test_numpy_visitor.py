import ast

import pytest

from floppy.analysis.visitors.numpy_visitor import NumpyCallVisitor


def test_basic_numpy_imports():
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

    imports = visitor.numpy_imports
    assert "numpy" in imports
    assert "numpy.linalg" in imports
    assert "numpy.random" in imports
    assert any("array" in imp for imp in imports)
    assert any("zeros" in imp for imp in imports)


def test_basic_numpy_calls():
    code = """
import numpy as np
x = np.array([1, 2, 3])
y = np.zeros((2, 2))
"""
    tree = ast.parse(code)
    visitor = NumpyCallVisitor()
    visitor.visit(tree)

    calls = visitor.numpy_calls
    assert ("call", "array", 2) in calls
    assert ("call", "zeros", 3) in calls


def test_complex_numpy_usage():
    code = """
import numpy as np
from numpy import array, zeros
import numpy.linalg as la
from numpy.random import normal, rand as random_gen
import numpy.fft as fft
from numpy import *

class MyClass:
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

# Broadcasting operations
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
result = arr1 + arr2
"""
    tree = ast.parse(code)
    visitor = NumpyCallVisitor()
    visitor.visit(tree)

    # Check imports
    imports = visitor.numpy_imports
    assert "numpy" in imports
    assert "numpy.linalg" in imports
    assert "numpy.random" in imports
    assert "numpy.fft" in imports

    # Check calls
    calls = visitor.numpy_calls
    expected_functions = {
        "array",
        "reshape",
        "transpose",
        "sum",
        "dot",
        "zeros",
        "full",
        "ones",
    }

    found_functions = {name for _, name, _ in calls}
    for func in expected_functions:
        assert func in found_functions, f"Expected to find {func} in numpy calls"


def test_numpy_attribute_access():
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

    calls = visitor.numpy_calls
    attributes = visitor.numpy_attributes

    assert ("call", "array", 2) in calls
    assert ("attribute", "mean", 3) in attributes
    assert ("attribute", "std", 4) in attributes
    assert ("attribute", "shape", 5) in attributes


def test_numpy_subscript_and_slice():
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

    calls = visitor.numpy_calls
    assert ("call", "array", 2) in calls


def test_numpy_math_operations():
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

    calls = visitor.numpy_calls
    assert ("call", "array", 2) in calls
    assert ("call", "array", 3) in calls
    assert ("call", "dot", 7) in calls


def test_numpy_with_error_handling():
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

    calls = visitor.numpy_calls
    assert ("call", "array", 3) in calls
    assert ("call", "invalid_function", 4) in calls
