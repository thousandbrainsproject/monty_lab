# numpy_visitor.py
# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT


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
