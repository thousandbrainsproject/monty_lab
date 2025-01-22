import ast
from typing import Set, Dict, Tuple


class ScipyCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.scipy_imports = {}  # Track what names refer to scipy
        self.scipy_calls = set()  # Using a set to avoid duplicates
        self.scipy_variables = set()  # Track variables holding scipy objects
        self.star_imported = False  # Track if scipy was star imported

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.startswith("scipy"):
                self.scipy_imports[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("scipy"):
            if len(node.names) == 1 and node.names[0].name == "*":
                self.star_imported = True
                print("Warning: '*' imports detected - some calls might be missed")
            else:
                for alias in node.names:
                    self.scipy_imports[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Handle multiple assignments (a = b = scipy.sparse.csr_matrix(...))
        if isinstance(node.value, ast.Call) and self._is_scipy_call(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.scipy_variables.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self.scipy_variables.add(elt.id)

        # Handle submodule assignments (optimize = scipy.optimize)
        elif isinstance(node.value, ast.Name):
            if node.value.id in self.scipy_imports:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.scipy_imports[target.id] = self.scipy_imports[
                            node.value.id
                        ]

        # Handle submodule assignments (sparse = scipy.sparse)
        elif isinstance(node.value, ast.Attribute):
            attr_chain = self._get_attribute_chain(node.value)
            if attr_chain[0] in self.scipy_imports:
                base_import = self.scipy_imports[attr_chain[0]]
                full_import = f"{base_import}.{'.'.join(attr_chain[1:])}"
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.scipy_imports[target.id] = full_import

        self.generic_visit(node)

    def _is_scipy_variable(self, node):
        if isinstance(node, ast.Name):
            return node.id in self.scipy_variables
        return False

    def _is_scipy_call(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id in self.scipy_imports or (
                self.star_imported and hasattr(node.func, "id")
            )
        elif isinstance(node.func, ast.Attribute):
            attr_chain = self._get_attribute_chain(node.func)
            return (
                attr_chain[0] in self.scipy_imports
                or attr_chain[0] in self.scipy_variables
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
        self.scipy_calls.add((call_type, name, lineno))

    def visit_Call(self, node):
        # Handle direct calls
        if isinstance(node.func, ast.Name):
            if node.func.id in self.scipy_imports:
                self._add_call("direct", self.scipy_imports[node.func.id], node.lineno)
            elif self.star_imported:
                # If we have star import, assume it might be scipy
                self._add_call("direct", f"scipy.{node.func.id}", node.lineno)

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
                if base_name in self.scipy_imports:
                    full_name = (
                        self.scipy_imports[base_name] + "." + ".".join(method_chain)
                    )
                    self._add_call("attribute", full_name, node.lineno)
                elif base_name in self.scipy_variables:
                    full_name = "scipy." + ".".join(method_chain)
                    self._add_call("attribute", full_name, node.lineno)
            elif isinstance(current, ast.Call):
                self.visit(current)
                if method_chain:
                    self._add_call(
                        "attribute", "scipy." + ".".join(method_chain), node.lineno
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


# Example usage:
if __name__ == "__main__":
    test_code = """
    import scipy as sp
    from scipy import optimize
    import scipy.sparse as sparse
    from scipy.linalg import solve
    from scipy.interpolate import interp1d
    from scipy import *

    def test_scipy_usage():
        # Direct calls
        result = optimize.minimize(lambda x: x**2, 0)
        
        # Create sparse matrix
        matrix = sparse.csr_matrix((3, 3))
        
        # Chained calls
        interpolator = interp1d([1, 2, 3], [4, 5, 6])
        y = interpolator(2.5)
        
        # Multiple assignments
        a = b = sparse.eye(3)
        
        # Method calls on scipy objects
        matrix.tocsc()
        
        # Nested calls
        result = solve(sparse.eye(3), [1, 2, 3])
    """

    tree = ast.parse(test_code)
    visitor = ScipyCallVisitor()
    visitor.visit(tree)

    # Convert set to sorted list for display
    scipy_calls = [{"type": t, "name": n, "line": l} for t, n, l in visitor.scipy_calls]
    for call in sorted(scipy_calls, key=lambda x: (x["line"], x["name"])):
        print(f"Found scipy call: {call}")
