# numpy_visitor.py
from .base_visitor import BaseLibraryVisitor

class NumpyCallVisitor(BaseLibraryVisitor):
    def __init__(self):
        super().__init__("numpy")

    def visit_ListComp(self, node):
        self.visit(node.elt)
        for generator in node.generators:
            self.visit(generator)

    def visit_SetComp(self, node):
        self.visit(node.elt)
        for generator in node.generators:
            self.visit(generator)

    def visit_DictComp(self, node):
        self.visit(node.key)
        self.visit(node.value)
        for generator in node.generators:
            self.visit(generator)

    def visit_Lambda(self, node):
        self.visit(node.body)

    def visit_BinOp(self, node):
        if self._is_library_variable(node.left) or self._is_library_variable(
            node.right
        ):
            self._add_call("attribute", "numpy.binary_operation", node.lineno)
        self.generic_visit(node)