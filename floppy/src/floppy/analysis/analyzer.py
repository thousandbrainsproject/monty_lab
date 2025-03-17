# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import ast
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .exceptions import FileAnalysisError
from .visitors.numpy_visitor import NumpyCallVisitor
from .visitors.scipy_visitor import ScipyCallVisitor
from .visitors.sklearn_visitor import SklearnCallVisitor


class FlopAnalyzer:
    """Main analyzer class for finding and analyzing FLOP operations in Python code."""

    def __init__(self):
        self.analysis_results: Dict[str, Dict] = {}
        self.numpy_visitor = NumpyCallVisitor()
        self.scipy_visitor = ScipyCallVisitor()
        self.sklearn_visitor = SklearnCallVisitor()

    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """Analyze a single Python file for FLOP operations.

        Args:
            filepath: Path to the Python file to analyze

        Returns:
            Dict containing analysis results with keys:
            - numpy_calls: List of NumPy function calls
            - scipy_calls: List of SciPy function calls
            - sklearn_calls: List of scikit-learn function calls

        Raises:
            FileAnalysisError: If there's an error reading or parsing the file
        """
        try:
            with open(filepath) as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError) as e:
            raise FileAnalysisError(f"Error reading/parsing {filepath}: {e!s}")

        # Reset visitors
        self.numpy_visitor = NumpyCallVisitor()
        self.scipy_visitor = ScipyCallVisitor()
        self.sklearn_visitor = SklearnCallVisitor()

        # Visit the AST with each visitor
        self.numpy_visitor.visit(tree)
        self.scipy_visitor.visit(tree)
        self.sklearn_visitor.visit(tree)

        # Format calls into a consistent structure
        numpy_calls = [
            {
                "module": "numpy",
                "type": t,
                "function": n,
                "line": l,
            }
            for t, n, l in self.numpy_visitor.calls
        ]

        scipy_calls = [
            {
                "module": "scipy",
                "type": t,
                "function": n,
                "line": l,
            }
            for t, n, l in self.scipy_visitor.calls
        ]

        sklearn_calls = [
            {
                "module": "sklearn",
                "type": t,
                "function": n,
                "line": l,
            }
            for t, n, l in self.sklearn_visitor.calls
        ]

        results = {
            "numpy_calls": numpy_calls,
            "scipy_calls": scipy_calls,
            "sklearn_calls": sklearn_calls,
            "imports": self._collect_imports(),
        }

        self.analysis_results[filepath] = results
        return results

    def _collect_imports(self) -> List[Dict]:
        """Collect all imports from the visitors."""
        imports = []

        # Collect NumPy imports
        for name, module in self.numpy_visitor.imports.items():
            imports.append({"module": "numpy", "name": name})

        # Collect SciPy imports
        for name, module in self.scipy_visitor.imports.items():
            imports.append({"module": "scipy", "name": name})

        # Collect scikit-learn imports
        for name, module in self.sklearn_visitor.imports.items():
            imports.append({"module": "sklearn", "name": name})

        return imports

    def analyze_directory(
        self, directory: str, recursive: bool = True
    ) -> Dict[str, Any]:
        """Analyze all Python files in a directory.

        Args:
            directory: Path to directory to analyze
            recursive: Whether to recursively analyze subdirectories

        Returns:
            Dict containing:
            - files: Dict mapping filenames to their analysis results
            - total_stats: Aggregated statistics across all files
        """
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")

        pattern = "**/*.py" if recursive else "*.py"
        results = {
            "files": {},
            "total_stats": {
                "numpy_calls": 0,
                "scipy_calls": 0,
                "sklearn_calls": 0,
                "total_files": 0,
            },
        }

        for py_file in path.glob(pattern):
            try:
                file_results = self.analyze_file(str(py_file))
                results["files"][str(py_file)] = file_results

                # Update totals
                results["total_stats"]["numpy_calls"] += len(
                    file_results["numpy_calls"]
                )
                results["total_stats"]["scipy_calls"] += len(
                    file_results["scipy_calls"]
                )
                results["total_stats"]["sklearn_calls"] += len(
                    file_results["sklearn_calls"]
                )
                results["total_stats"]["total_files"] += 1

            except FileAnalysisError as e:
                print(f"Warning: {e!s}")
                continue

        return results

    def save_results(self, results: Dict[str, Any], output_dir: str) -> str:
        """Save analysis results to CSV files.

        Args:
            results: Analysis results from analyze_directory()
            output_dir: Directory to save results in

        Returns:
            Path to the saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        rows = []
        for filename, analysis in results["files"].items():
            # Add operations
            self._add_operation_rows(rows, filename, analysis)
            # Add imports
            self._add_import_rows(rows, filename, analysis)

        if rows:
            df = pd.DataFrame(rows)
            output_file = output_path / f"flop_analysis_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            return str(output_file)

        return ""

    def _add_operation_rows(
        self, rows: List[Dict], filename: str, analysis: Dict
    ) -> None:
        """Add rows for operations to the results DataFrame."""
        # Add all operations including library calls, assignments, and binary operations
        for op_type in ["numpy_calls", "sklearn_calls", "scipy_calls"]:
            for op in analysis[op_type]:
                row = {
                    "filename": filename,
                    "operation_type": self._get_operation_type(op["type"]),
                    "module": op["module"],
                    "function": op["function"],
                    "line": op.get("line", ""),
                }
                rows.append(row)

    def _get_operation_type(self, type_str: str) -> str:
        """Convert the operation type to a consistent category."""
        type_mapping = {
            "direct": "function_call",
            "attribute": "method_call",
            "binary": "binary_operation",
            "assign": "assignment",
        }
        return type_mapping.get(type_str, type_str)

    def _add_import_rows(self, rows: List[Dict], filename: str, analysis: Dict) -> None:
        """Add rows for imports to the results DataFrame."""
        for lib_type in ["numpy_imports", "sklearn_imports", "scipy_imports"]:
            if lib_type in analysis:
                for import_info in analysis[lib_type]:
                    row = {
                        "filename": filename,
                        "import_type": import_info.get(
                            "type", "direct"
                        ),  # direct, from, star
                        "module": import_info.get("module", ""),
                        "name": import_info.get("name", ""),
                        "alias": import_info.get("alias", ""),
                        "line": import_info.get("line", ""),
                    }
                    rows.append(row)
