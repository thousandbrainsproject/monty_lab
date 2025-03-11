# numpy_visitor.py
import ast

from .base_visitor import BaseLibraryVisitor


class NumpyCallVisitor(BaseLibraryVisitor):
    """Visitor for tracking NumPy function calls and operations."""

    def __init__(self):
        super().__init__("numpy")

    def visit_BinOp(self, node):
        """Override binary operations to handle numpy-specific operations."""
        if self._is_library_variable(node.left) or self._is_library_variable(
            node.right
        ):
            # NumPy-specific handling of array operations
            self._add_call("attribute", "numpy.binary_operation", node.lineno)
        self.generic_visit(node)