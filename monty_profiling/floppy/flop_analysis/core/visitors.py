import ast
from typing import Dict, List, Optional
from collections import defaultdict

from floppy.flop_analysis.core.operations import OperationRegistry


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
        # Track imported names and their sources
        self.import_map: Dict[str, Dict[str, str]] = {
            "numpy": {},
            "scipy": {},
            "sklearn": {},
        }

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track the current method context."""
        prev_method = self.current_method
        self.current_method = node.name
        self.generic_visit(node)
        self.current_method = prev_method

    def visit_Import(self, node: ast.Import) -> None:
        """Track import statements and map imported names."""
        for alias in node.names:
            imported_name = alias.asname or alias.name

            # Track full imports
            if any(pkg in alias.name for pkg in ["numpy", "scipy", "sklearn"]):
                self.imports.append(
                    {
                        "module": alias.name,
                        "name": imported_name,
                        "line": node.lineno,
                        "col": node.col_offset,
                    }
                )

                # Map the imported name to its source
                base_module = alias.name.split(".")[0]
                if base_module in self.import_map:
                    self.import_map[base_module][imported_name] = alias.name

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from-import statements and map imported names."""
        if node.module and any(
            pkg in node.module for pkg in ["numpy", "scipy", "sklearn"]
        ):
            base_module = node.module.split(".")[0]

            for alias in node.names:
                imported_name = alias.asname or alias.name
                self.imports.append(
                    {
                        "module": node.module,
                        "name": imported_name,
                        "line": node.lineno,
                        "col": node.col_offset,
                    }
                )

                # Map the imported name to its full module path
                if base_module in self.import_map:
                    self.import_map[base_module][imported_name] = (
                        f"{node.module}.{alias.name}"
                    )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Track function/method calls with their context."""
        if isinstance(node.func, ast.Attribute):
            # Handle method calls on imported objects (e.g., kdtree.query())
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr

                # Check if this is a method call on a tracked object
                call_info = self._process_method_call(obj_name, method_name, node)
                if call_info:
                    self._add_call_info(call_info)

            # Handle direct module attribute calls (e.g., numpy.add)
            module = (
                node.func.value.id if isinstance(node.func.value, ast.Name) else None
            )
            if module:
                call_info = {
                    "module": module,
                    "function": node.func.attr,
                    "line": node.lineno,
                    "col": node.col_offset,
                    "method_context": self.current_method,
                }
                self._add_call_info(call_info)

        elif isinstance(node.func, ast.Name):
            # Handle calls to imported functions (e.g., KDTree())
            func_name = node.func.id
            for base_module, imports in self.import_map.items():
                if func_name in imports:
                    call_info = {
                        "module": base_module,
                        "function": func_name,
                        "line": node.lineno,
                        "col": node.col_offset,
                        "method_context": self.current_method,
                    }
                    self._add_call_info(call_info)

        self.generic_visit(node)

    def _process_method_call(
        self, obj_name: str, method_name: str, node: ast.Call
    ) -> Optional[Dict]:
        """Process a method call and return call info if it's a tracked operation."""
        for base_module, registry in {
            "scipy": OperationRegistry.SCIPY_OPERATIONS,
            "sklearn": OperationRegistry.SKLEARN_OPERATIONS,
        }.items():
            for class_name, class_info in registry.items():
                if (
                    class_name in self.import_map[base_module]
                    and method_name in class_info["methods"]
                ):
                    return {
                        "module": base_module,
                        "function": f"{class_name}.{method_name}",
                        "line": node.lineno,
                        "col": node.col_offset,
                        "method_context": self.current_method,
                    }
        return None

    def _add_call_info(self, call_info: Dict) -> None:
        """Add call info to the appropriate list based on the module."""
        module = call_info["module"]
        if module in {"np", "numpy"}:
            self.numpy_calls.append(call_info)
        elif module in {"sklearn", "sk"}:
            self.sklearn_calls.append(call_info)
            self.method_contexts["sklearn"].append(call_info)
        elif module == "scipy":
            self.scipy_calls.append(call_info)
            self.method_contexts["scipy"].append(call_info)

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
