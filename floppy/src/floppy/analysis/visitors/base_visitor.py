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
        """Handle assignments to track library variables."""
        # First visit the value to ensure we catch any nested calls
        self.generic_visit(node.value)

        # Handle tuple assignments with multiple values
        if isinstance(node.targets[0], ast.Tuple):
            if isinstance(node.value, ast.Tuple):
                # Handle explicit tuple assignments: a, b = (x, y)
                for target, value in zip(node.targets[0].elts, node.value.elts):
                    if isinstance(target, ast.Name):
                        if isinstance(value, ast.Call) and self._is_library_call(value):
                            self.variables.add(target.id)
                        elif isinstance(value, ast.Name) and value.id in self.variables:
                            self.variables.add(target.id)
            else:
                # Handle sequence unpacking: a, b = some_call()
                if isinstance(node.value, ast.Call) and self._is_library_call(
                    node.value
                ):
                    for target in node.targets[0].elts:
                        if isinstance(target, ast.Name):
                            self.variables.add(target.id)

        # Handle single assignments
        else:
            if isinstance(node.value, ast.Call):
                # Track assignments from library calls
                if self._is_library_call(node.value):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.variables.add(target.id)

            # Track assignments from existing library variables
            elif isinstance(node.value, ast.Name) and node.value.id in self.variables:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.variables.add(target.id)

            # Handle assignments from method calls on library variables
            elif isinstance(node.value, ast.Call) and isinstance(
                node.value.func, ast.Attribute
            ):
                if (
                    isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id in self.variables
                ):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.variables.add(target.id)

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

    def visit_Call(self, node):
        """Handle function calls."""
        # Handle direct calls
        if isinstance(node.func, ast.Name):
            if node.func.id in self.imports:
                self._add_call("call", node.func.id, node.lineno)
            elif self.star_imported:
                self._add_call(
                    "call", f"{self.library_name}.{node.func.id}", node.lineno
                )

        # Handle attribute calls
        elif isinstance(node.func, ast.Attribute):
            # Build the attribute chain
            method_chain = []
            current = node.func
            base_var = None

            while isinstance(current, ast.Attribute):
                method_chain.insert(0, current.attr)
                current = current.value
                if isinstance(current, ast.Name):
                    base_var = current.id

            if base_var:
                if base_var in self.variables:
                    # Method call on a tracked object
                    self._add_call("call", method_chain[-1], node.lineno)
                elif base_var in self.imports:
                    # Module attribute call
                    full_path = f"{self.imports[base_var]}.{'.'.join(method_chain)}"
                    self._add_call("attribute", full_path, node.lineno)
            elif isinstance(current, ast.Call):
                # Handle chained calls
                self.visit(current)
                if method_chain:
                    self._add_call("call", method_chain[-1], node.lineno)

        # Visit arguments and keywords
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_ListComp(self, node):
        """Handle list comprehensions."""
        self.visit(node.elt)
        for generator in node.generators:
            self.visit(generator)

    def visit_SetComp(self, node):
        """Handle set comprehensions."""
        self.visit(node.elt)
        for generator in node.generators:
            self.visit(generator)

    def visit_DictComp(self, node):
        """Handle dictionary comprehensions."""
        self.visit(node.key)
        self.visit(node.value)
        for generator in node.generators:
            self.visit(generator)

    def visit_Lambda(self, node):
        """Handle lambda expressions."""
        self.visit(node.body)

    def visit_BinOp(self, node):
        """Handle binary operations."""
        if self._is_library_variable(node.left) or self._is_library_variable(
            node.right
        ):
            self._add_call(
                "attribute", f"{self.library_name}.binary_operation", node.lineno
            )
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Handle attribute access."""
        # Build the attribute chain
        current = node
        method_chain = []

        while isinstance(current, ast.Attribute):
            method_chain.insert(0, current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            base_name = current.id
            if base_name in self.variables:
                # Add the attribute access
                self._add_call("call", method_chain[-1], node.lineno)

        self.generic_visit(node)

    def visit_Subscript(self, node):
        """Handle subscript operations (indexing and slicing)."""
        if isinstance(node.value, ast.Name) and node.value.id in self.variables:
            self._add_call("call", "getitem", node.lineno)
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
