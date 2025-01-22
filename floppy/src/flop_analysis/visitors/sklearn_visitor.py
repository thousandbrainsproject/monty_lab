import ast
from typing import Set, Dict, Tuple


class SklearnCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.sklearn_imports = {}  # Track what names refer to sklearn
        self.sklearn_calls = set()  # Using a set to avoid duplicates
        self.sklearn_variables = set()  # Track variables holding sklearn objects
        self.star_imported = False  # Track if sklearn was star imported

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.startswith("sklearn"):
                self.sklearn_imports[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("sklearn"):
            if len(node.names) == 1 and node.names[0].name == "*":
                self.star_imported = True
                print("Warning: '*' imports detected - some calls might be missed")
            else:
                for alias in node.names:
                    self.sklearn_imports[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Handle model instantiation assignments (clf = RandomForestClassifier())
        if isinstance(node.value, ast.Call) and self._is_sklearn_call(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.sklearn_variables.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self.sklearn_variables.add(elt.id)

        # Handle module assignments (preprocessing = sklearn.preprocessing)
        elif isinstance(node.value, ast.Name):
            if node.value.id in self.sklearn_imports:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.sklearn_imports[target.id] = self.sklearn_imports[
                            node.value.id
                        ]

        # Handle submodule assignments (ensemble = sklearn.ensemble)
        elif isinstance(node.value, ast.Attribute):
            attr_chain = self._get_attribute_chain(node.value)
            if attr_chain[0] in self.sklearn_imports:
                base_import = self.sklearn_imports[attr_chain[0]]
                full_import = f"{base_import}.{'.'.join(attr_chain[1:])}"
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.sklearn_imports[target.id] = full_import

        self.generic_visit(node)

    def _is_sklearn_variable(self, node):
        if isinstance(node, ast.Name):
            return node.id in self.sklearn_variables
        return False

    def _is_sklearn_call(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id in self.sklearn_imports or (
                self.star_imported and hasattr(node.func, "id")
            )
        elif isinstance(node.func, ast.Attribute):
            attr_chain = self._get_attribute_chain(node.func)
            return (
                attr_chain[0] in self.sklearn_imports
                or attr_chain[0] in self.sklearn_variables
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
        self.sklearn_calls.add((call_type, name, lineno))

    def visit_Call(self, node):
        # Handle direct calls
        if isinstance(node.func, ast.Name):
            if node.func.id in self.sklearn_imports:
                self._add_call(
                    "direct", self.sklearn_imports[node.func.id], node.lineno
                )
            elif self.star_imported:
                # If we have star import, assume it might be sklearn
                self._add_call("direct", f"sklearn.{node.func.id}", node.lineno)

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
                if base_name in self.sklearn_imports:
                    full_name = (
                        self.sklearn_imports[base_name] + "." + ".".join(method_chain)
                    )
                    self._add_call("attribute", full_name, node.lineno)
                elif base_name in self.sklearn_variables:
                    # Handle method calls on sklearn objects (e.g., clf.fit(), clf.predict())
                    if method_chain:
                        self._add_call(
                            "method", f"sklearn.{'.'.join(method_chain)}", node.lineno
                        )
            elif isinstance(current, ast.Call):
                self.visit(current)
                if method_chain:
                    self._add_call(
                        "attribute", "sklearn." + ".".join(method_chain), node.lineno
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
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    import sklearn.metrics as metrics
    from sklearn.pipeline import Pipeline
    
    def test_sklearn_usage():
        # Create classifier
        clf = RandomForestClassifier(n_estimators=100)
        
        # Preprocessing
        scaler = preprocessing.StandardScaler()
        
        # Pipeline creation
        pipe = Pipeline([
            ('scaler', preprocessing.StandardScaler()),
            ('clf', RandomForestClassifier())
        ])
        
        # Method calls
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        
        # Metrics
        score = metrics.accuracy_score(y_test, predictions)
        
        # Chained calls
        clf.fit(X_train, y_train).predict(X_test)
    """

    tree = ast.parse(test_code)
    visitor = SklearnCallVisitor()
    visitor.visit(tree)

    # Convert set to sorted list for display
    sklearn_calls = [
        {"type": t, "name": n, "line": l} for t, n, l in visitor.sklearn_calls
    ]
    for call in sorted(sklearn_calls, key=lambda x: (x["line"], x["name"])):
        print(f"Found sklearn call: {call}")
