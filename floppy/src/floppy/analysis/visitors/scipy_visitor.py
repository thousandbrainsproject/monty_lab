# scipy_visitor.py
from .base_visitor import BaseLibraryVisitor

class ScipyCallVisitor(BaseLibraryVisitor):
    def __init__(self):
        super().__init__("scipy")