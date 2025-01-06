import ast
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import pandas as pd
from .visitors import ASTVisitor
from .operations import OPERATION_REGISTRY
from .exceptions import FileAnalysisError


class FlopAnalyzer:
    """Main analyzer class for finding and analyzing FLOP operations in Python code."""

    def __init__(self):
        self.analysis_results: Dict[str, Dict] = {}
        self._visitor = ASTVisitor()

    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """Analyze a single Python file for FLOP operations.

        Args:
            filepath: Path to the Python file to analyze

        Returns:
            Dict containing analysis results with keys:
            - numpy_calls: List of NumPy function calls
            - scipy_calls: List of SciPy function calls
            - sklearn_calls: List of scikit-learn function calls
            - method_contexts: Dict mapping functions to their contexts

        Raises:
            FileAnalysisError: If there's an error reading or parsing the file
        """
        try:
            with open(filepath, "r") as f:
                tree = ast.parse(f.read())
        except (IOError, SyntaxError) as e:
            raise FileAnalysisError(f"Error reading/parsing {filepath}: {str(e)}")

        self._visitor.reset()
        self._visitor.visit(tree)

        results = {
            "numpy_calls": self._visitor.numpy_calls,
            "scipy_calls": self._visitor.scipy_calls,
            "sklearn_calls": self._visitor.sklearn_calls,
            "method_contexts": self._visitor.method_contexts,
            "imports": self._visitor.imports,
            "operators": self._visitor.operators,
        }

        self.analysis_results[filepath] = results
        return results

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
                "total_flops": 0,
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

        self._generate_summary(results)
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

    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """Generate a high-level summary of the analysis results."""
        total_stats = results["total_stats"]
        results["summary"] = {
            "total_files_analyzed": total_stats["total_files"],
            "total_flop_operations": total_stats["total_flops"],
            "most_common_operations": self._get_most_common_operations(
                results["files"]
            ),
            "files_with_most_flops": self._get_files_with_most_flops(results["files"]),
        }

    def _get_most_common_operations(
        self, files: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """Get the most frequently used FLOP operations across all files."""
        operation_counts: Dict[str, int] = {}
        for file_results in files.values():
            for op in file_results["numpy_calls"]:
                operation_counts[op["function"]] = (
                    operation_counts.get(op["function"], 0) + 1
                )

        return sorted(
            [
                {"operation": op, "count": count}
                for op, count in operation_counts.items()
            ],
            key=lambda x: x["count"],
            reverse=True,
        )[:10]

    def _get_files_with_most_flops(
        self, files: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """Get the files with the highest number of FLOP operations."""
        file_counts = [
            {
                "file": filename,
                "flop_count": len(results["numpy_calls"])
                + len(results["scipy_calls"])
                + len(results["sklearn_calls"]),
            }
            for filename, results in files.items()
        ]
        return sorted(file_counts, key=lambda x: x["flop_count"], reverse=True)[:10]

    def _add_operation_rows(
        self, rows: List[Dict], filename: str, analysis: Dict
    ) -> None:
        """Add rows for operations to the results DataFrame."""
        for op_type in ["numpy_calls", "sklearn_calls", "scipy_calls"]:
            for op in analysis[op_type]:
                rows.append(
                    {
                        "filename": filename,
                        "operation_type": op_type.replace("_calls", "_ops"),
                        "module": op["module"],
                        "function": op["function"],
                        "method_context": op["method_context"],
                        "line": op["line"],
                        "column": op["col"],
                    }
                )

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
