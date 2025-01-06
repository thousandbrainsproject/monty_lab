# flop_analysis/core/visitors.py

import ast
from typing import Dict, List, Set, Any, Optional
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
        self.object_types: Dict[str, str] = {}  # Track variable names and their types

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
        """Enhanced tracking of function/method calls with better object method handling."""
        if isinstance(node.func, ast.Attribute):
            # Handle method calls on objects (e.g., kdtree.query())
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr

                # Check if this is a tracked object method call
                if obj_name in self.object_types:
                    obj_info = self.object_types[obj_name]
                    if method_name in obj_info["registry"]["methods"]:
                        call_info = {
                            "module": obj_info["module"],
                            "function": f"{obj_info['class_name']}.{method_name}",
                            "line": node.lineno,
                            "col": node.col_offset,
                            "method_context": self.current_method,
                        }
                        self._add_call_info(call_info)

            # Handle regular attribute calls (e.g., np.array())
            else:
                attr_parts = []
                current = node.func
                while isinstance(current, ast.Attribute):
                    attr_parts.append(current.attr)
                    current = current.value

                if isinstance(current, ast.Name):
                    base_module = current.id
                    attr_parts.reverse()
                    function_path = ".".join(attr_parts)

                    call_info = {
                        "module": base_module,
                        "function": function_path,
                        "line": node.lineno,
                        "col": node.col_offset,
                        "method_context": self.current_method,
                    }
                    self._add_call_info(call_info)

        elif isinstance(node.func, ast.Name):
            # Handle direct function calls
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

    def _add_call_info(self, call_info: Dict) -> None:
        """Add call info to the appropriate tracking lists."""
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
        """Track binary operations and add them to operator calls."""
        op_type = type(node.op)
        if op_type in OperationRegistry.ARITHMETIC_OPERATIONS:
            op_info = OperationRegistry.ARITHMETIC_OPERATIONS[op_type]

            call_info = {
                "module": "arithmetic",
                "function": op_info["name"],
                "symbol": op_info["symbol"],
                "line": node.lineno,
                "col": node.col_offset,
                "method_context": self.current_method,
            }

            # Add to operators list for detailed tracking
            self.operators.append(call_info)

            # Add to numpy calls to ensure they appear in main results
            self.numpy_calls.append(call_info)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track object instantiations and their types with enhanced metadata."""
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                class_name = node.value.func.id

                # Determine the source module and registry for the class
                source_module = None
                registry = None

                for module, imports in self.import_map.items():
                    if class_name in imports:
                        source_module = module
                        if module == "scipy":
                            registry = OperationRegistry.SCIPY_OPERATIONS
                        elif module == "sklearn":
                            registry = OperationRegistry.SKLEARN_OPERATIONS
                        break

                # If this is a tracked class (like KDTree)
                if registry and class_name in registry:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # Store enhanced metadata about the object
                            self.object_types[target.id] = {
                                "class_name": class_name,
                                "module": source_module,
                                "registry": registry[class_name],
                                "lineno": node.lineno,
                                "col_offset": node.col_offset,
                            }

        self.generic_visit(node)
