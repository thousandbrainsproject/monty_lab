import ast


class BaseLibraryVisitor(ast.NodeVisitor):
    """Base class for library-specific visitors."""

    def __init__(self, library_name: str):
        self.library_name = library_name
        self.imports = {}  # Track what names refer to the library
        self.calls = set()  # Using a set to avoid duplicates
        self.variables = set()  # Track variables holding library objects
        self.star_imported = False  # Track if library was star imported

    def visit_Import(self, node):
        """Handle import statements."""
        for alias in node.names:
            if alias.name.startswith(self.library_name):
                self.imports[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Handle from-import statements."""
        if node.module and node.module.startswith(self.library_name):
            if len(node.names) == 1 and node.names[0].name == "*":
                self.star_imported = True
                print(f"Warning: '*' imports detected - some calls might be missed")
            else:
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
        self.generic_visit(node)

    def visit_Assign(self, node):
        """Handle assignments."""
        # Handle multiple assignments
        if isinstance(node.value, ast.Call) and self._is_library_call(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.variables.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self.variables.add(elt.id)

        # Handle name assignments
        elif isinstance(node.value, ast.Name):
            if node.value.id in self.imports:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.imports[target.id] = self.imports[node.value.id]

        # Handle submodule assignments
        elif isinstance(node.value, ast.Attribute):
            attr_chain = self._get_attribute_chain(node.value)
            if attr_chain[0] in self.imports:
                base_import = self.imports[attr_chain[0]]
                full_import = f"{base_import}.{'.'.join(attr_chain[1:])}"
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.imports[target.id] = full_import

        self.generic_visit(node)

    def _is_library_variable(self, node):
        """Check if node represents a library variable."""
        if isinstance(node, ast.Name):
            return node.id in self.variables
        return False

    def _is_library_call(self, node):
        """Check if node represents a library function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id in self.imports or (
                self.star_imported and hasattr(node.func, "id")
            )
        elif isinstance(node.func, ast.Attribute):
            attr_chain = self._get_attribute_chain(node.func)
            return attr_chain[0] in self.imports or attr_chain[0] in self.variables
        return False

    def _get_attribute_chain(self, node):
        """Get the full chain of attributes."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.insert(0, current.id)
        return parts

    def _add_call(self, call_type: str, name: str, lineno: int):
        """Add a call to the tracked calls set."""
        self.calls.add((call_type, name, lineno))

    def visit_Call(self, node):
        """Handle function calls."""
        # Handle direct calls
        if isinstance(node.func, ast.Name):
            if node.func.id in self.imports:
                self._add_call("direct", self.imports[node.func.id], node.lineno)
            elif self.star_imported:
                self._add_call(
                    "direct", f"{self.library_name}.{node.func.id}", node.lineno
                )

        # Handle attribute calls and chains
        elif isinstance(node.func, ast.Attribute):
            current = node.func
            method_chain = []

            while isinstance(current, ast.Attribute):
                method_chain.insert(0, current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                base_name = current.id
                if base_name in self.imports:
                    full_name = self.imports[base_name] + "." + ".".join(method_chain)
                    self._add_call("attribute", full_name, node.lineno)
                elif base_name in self.variables:
                    full_name = f"{self.library_name}." + ".".join(method_chain)
                    self._add_call("attribute", full_name, node.lineno)
            elif isinstance(current, ast.Call):
                self.visit(current)
                if method_chain:
                    self._add_call(
                        "attribute",
                        f"{self.library_name}." + ".".join(method_chain),
                        node.lineno,
                    )

        # Visit arguments and keywords
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

        # Handle chained calls
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Call
        ):
            self.visit(node.func.value)
