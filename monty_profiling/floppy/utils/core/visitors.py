import ast
from typing import Dict, List, Set, Any
from collections import defaultdict


class ASTVisitor(ast.NodeVisitor):
    """AST visitor for finding FLOP operations in Python code."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracking attributes."""
        self.numpy_calls: List[Dict] = []
        self.scipy_calls: List[Dict] = []
        self.sklearn_calls: List[Dict] = []
        self.operators: List[Dict] = []
        self.current_method: Optional[str] = None
        self.method_contexts: Dict[str, List] = defaultdict(list)
        self.imports: List[Dict] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track the current method context."""
        prev_method = self.current_method
        self.current_method = node.name
        self.generic_visit(node)
        self.current_method = prev_method

    def visit_Import(self, node: ast.Import) -> None:
        """Track import statements."""
        for alias in node.names:
            if any(pkg in alias.name for pkg in ["numpy", "scipy", "sklearn"]):
                self.imports.append(
                    {
                        "module": alias.name,
                        "name": alias.asname or alias.name,
                        "line": node.lineno,
                        "col": node.col_offset,
                    }
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from-import statements."""
        if node.module and any(
            pkg in node.module for pkg in ["numpy", "scipy", "sklearn"]
        ):
            for alias in node.names:
                self.imports.append(
                    {
                        "module": node.module,
                        "name": alias.name,
                        "line": node.lineno,
                        "col": node.col_offset,
                    }
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Track function/method calls with their context."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module = node.func.value.id
                func_name = node.func.attr

                call_info = {
                    "module": module,
                    "function": func_name,
                    "line": node.lineno,
                    "col": node.col_offset,
                    "method_context": self.current_method,
                }

                if module in {"np", "numpy"}:
                    self.numpy_calls.append(call_info)
                elif module in {"sklearn", "sk"}:
                    self.sklearn_calls.append(call_info)
                    self.method_contexts["sklearn"].append(call_info)
                elif module == "scipy":
                    self.scipy_calls.append(call_info)
                    self.method_contexts["scipy"].append(call_info)

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Track binary operations."""
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Pow: "**",
        }
        op_type = type(node.op)
        if op_type in op_map:
            self.operators.append(
                {"type": op_map[op_type], "line": node.lineno, "col": node.col_offset}
            )
        self.generic_visit(node)
