# scipy_visitor.py
import ast

from .base_visitor import BaseLibraryVisitor


class ScipyCallVisitor(BaseLibraryVisitor):
    """Visitor for tracking SciPy function calls and operations."""

    def __init__(self):
        super().__init__("scipy")