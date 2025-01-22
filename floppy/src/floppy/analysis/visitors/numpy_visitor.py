import ast
from typing import Set, Dict, Tuple


class NumpyCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.numpy_imports = {}  # Track what names refer to numpy
        self.numpy_calls = set()  # Using a set to avoid duplicates
        self.numpy_variables = set()  # Track variables holding numpy objects
        self.star_imported = False  # Track if numpy was star imported

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.startswith("numpy"):
                self.numpy_imports[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("numpy"):
            if len(node.names) == 1 and node.names[0].name == "*":
                self.star_imported = True
                print("Warning: '*' imports detected - some calls might be missed")
            else:
                for alias in node.names:
                    self.numpy_imports[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Handle multiple assignments (a = b = np.array([1,2,3]))
        if isinstance(node.value, ast.Call) and self._is_numpy_call(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.numpy_variables.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self.numpy_variables.add(elt.id)

        # Handle attribute assignments (self.data = np.array([1,2,3]))
        elif isinstance(node.value, ast.Name):
            if node.value.id in self.numpy_imports:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.numpy_imports[target.id] = self.numpy_imports[
                            node.value.id
                        ]

        # Handle submodule assignments (fft = numpy.fft)
        elif isinstance(node.value, ast.Attribute):
            attr_chain = self._get_attribute_chain(node.value)
            if attr_chain[0] in self.numpy_imports:
                base_import = self.numpy_imports[attr_chain[0]]
                full_import = f"{base_import}.{'.'.join(attr_chain[1:])}"
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.numpy_imports[target.id] = full_import

        self.generic_visit(node)

    def visit_ListComp(self, node):
        # Handle list comprehensions
        self.visit(node.elt)  # Visit the expression
        for generator in node.generators:
            self.visit(generator)

    def visit_SetComp(self, node):
        # Handle set comprehensions
        self.visit(node.elt)
        for generator in node.generators:
            self.visit(generator)

    def visit_DictComp(self, node):
        # Handle dictionary comprehensions
        self.visit(node.key)
        self.visit(node.value)
        for generator in node.generators:
            self.visit(generator)

    def visit_Lambda(self, node):
        # Handle lambda functions
        self.visit(node.body)

    def visit_BinOp(self, node):
        # Handle binary operations (like arr1 + arr2)
        if self._is_numpy_variable(node.left) or self._is_numpy_variable(node.right):
            self._add_call("attribute", "numpy.binary_operation", node.lineno)
        self.generic_visit(node)

    def _is_numpy_variable(self, node):
        if isinstance(node, ast.Name):
            return node.id in self.numpy_variables
        return False

    def _is_numpy_call(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id in self.numpy_imports or (
                self.star_imported and hasattr(node.func, "id")
            )
        elif isinstance(node.func, ast.Attribute):
            attr_chain = self._get_attribute_chain(node.func)
            return (
                attr_chain[0] in self.numpy_imports
                or attr_chain[0] in self.numpy_variables
            )
        return False

    def _get_attribute_chain(self, node):
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.insert(0, current.id)
        return parts

    def _add_call(self, call_type: str, name: str, lineno: int):
        self.numpy_calls.add((call_type, name, lineno))

    def visit_Call(self, node):
        # Handle direct calls
        if isinstance(node.func, ast.Name):
            if node.func.id in self.numpy_imports:
                self._add_call("direct", self.numpy_imports[node.func.id], node.lineno)
            elif self.star_imported:
                # If we have star import, assume it might be numpy
                self._add_call("direct", f"numpy.{node.func.id}", node.lineno)

        # Handle attribute calls and chains
        elif isinstance(node.func, ast.Attribute):
            current = node.func
            method_chain = []

            # Walk up the attribute chain
            while isinstance(current, ast.Attribute):
                method_chain.insert(0, current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                base_name = current.id
                if base_name in self.numpy_imports:
                    full_name = (
                        self.numpy_imports[base_name] + "." + ".".join(method_chain)
                    )
                    self._add_call("attribute", full_name, node.lineno)
                elif base_name in self.numpy_variables:
                    full_name = "numpy." + ".".join(method_chain)
                    self._add_call("attribute", full_name, node.lineno)
            elif isinstance(current, ast.Call):
                self.visit(current)
                if method_chain:
                    self._add_call(
                        "attribute", "numpy." + ".".join(method_chain), node.lineno
                    )

        # Visit arguments and keywords
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

        # Handle any chained calls on the result
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Call
        ):
            self.visit(node.func.value)


# Test it with the complex cases
test_code = """
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

def test_complex_cases():
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

tree = ast.parse(test_code)
visitor = NumpyCallVisitor()
visitor.visit(tree)

# Convert set to sorted list for display
numpy_calls = [{"type": t, "name": n, "line": l} for t, n, l in visitor.numpy_calls]
for call in sorted(numpy_calls, key=lambda x: (x["line"], x["name"])):
    print(f"Found numpy call: {call}")
