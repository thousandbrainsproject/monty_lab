# sklearn_visitor.py
from .base_visitor import BaseLibraryVisitor


class SklearnCallVisitor(BaseLibraryVisitor):
    def __init__(self):
        super().__init__("sklearn")