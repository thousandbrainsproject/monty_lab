import ast
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from ..visitors.numpy_visitor import NumpyCallVisitor
from ..visitors.scipy_visitor import ScipyCallVisitor
from ..visitors.sklearn_visitor import SklearnCallVisitor
from .exceptions import FileAnalysisError


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
            with open(filepath, "r") as f:
                tree = ast.parse(f.read())
        except (IOError, SyntaxError) as e:
            raise FileAnalysisError(f"Error reading/parsing {filepath}: {str(e)}")

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
                "col": 0,  # Column information not currently tracked
                "method_context": "",  # Method context not currently tracked
            }
            for t, n, l in self.numpy_visitor.numpy_calls
        ]

        scipy_calls = [
            {
                "module": "scipy",
                "type": t,
                "function": n,
                "line": l,
                "col": 0,
                "method_context": "",
            }
            for t, n, l in self.scipy_visitor.scipy_calls
        ]

        sklearn_calls = [
            {
                "module": "sklearn",
                "type": t,
                "function": n,
                "line": l,
                "col": 0,
                "method_context": "",
            }
            for t, n, l in self.sklearn_visitor.sklearn_calls
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
        for name, module in self.numpy_visitor.numpy_imports.items():
            imports.append(
                {
                    "module": "numpy",
                    "name": name,
                    "line": 0,  # Line numbers not currently tracked for imports
                    "col": 0,
                }
            )

        # Collect SciPy imports
        for name, module in self.scipy_visitor.scipy_imports.items():
            imports.append({"module": "scipy", "name": name, "line": 0, "col": 0})

        # Collect scikit-learn imports
        for name, module in self.sklearn_visitor.sklearn_imports.items():
            imports.append({"module": "sklearn", "name": name, "line": 0, "col": 0})

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
            - summary: High-level summary of findings
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
            "summary": {},
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
                print(f"Warning: {str(e)}")
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
        # Add all operations including both library calls and arithmetic operations
        for op_type in ["numpy_calls", "sklearn_calls", "scipy_calls"]:
            for op in analysis[op_type]:
                row = {
                    "filename": filename,
                    "operation_type": "arithmetic_ops"
                    if op["module"] == "arithmetic"
                    else op_type.replace("_calls", "_ops"),
                    "module": op["module"],
                    "function": op["function"],
                    "method_context": op["method_context"],
                    "line": op["line"],
                    "column": op["col"],
                }

                # Add symbol and flops for arithmetic operations
                if op["module"] == "arithmetic":
                    row["symbol"] = op.get("symbol", "")

                rows.append(row)

    def _add_import_rows(self, rows: List[Dict], filename: str, analysis: Dict) -> None:
        """Add rows for imports to the results DataFrame."""
        for imp in analysis["imports"]:
            rows.append(
                {
                    "filename": filename,
                    "operation_type": "import",
                    "module": imp["module"],
                    "function": imp.get("name", ""),
                    "method_context": "",
                    "line": imp["line"],
                    "column": imp["col"],
                }
            )