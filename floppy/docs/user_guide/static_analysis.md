# Static Code Analysis

The static code analysis feature helps identify potential FLOP operations in your codebase before execution.

## Overview

The analyzer uses Python's `ast` module to parse source code and identify:

1. Function calls that could contribute to FLOP operations
2. Import statements related to numerical computing
3. Potential FLOP-heavy code regions

## What Gets Analyzed

### Function Calls

- **NumPy Operations**: All NumPy function calls, including:
  - ufuncs
  - Linear algebra operations
  - Array manipulations

- **SciPy Operations**: Particularly from:
  - spatial module
  - linear algebra module

- **scikit-learn Operations**: Machine learning operations involving numerical computations

### Import Analysis

Tracks imports from numerical computing libraries:

```python
import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
```

## Usage

### Basic Analysis

```python
from floppy.flop_analysis.core.analyzer import FlopAnalyzer

# Analyze a single file
analyzer = FlopAnalyzer()
analyzer.analyze_file('example_code.py')

# Analyze a directory
analyzer.analyze_directory('path/to/code')
```

### Command Line Interface

```bash
python run_static_analysis.py --dir path/to/analyze
```

Results are saved in `results/static_analysis/flop_analysis.csv`.

## Analysis Results

The analysis provides:

1. File-level breakdown of numerical operations
2. Location information (line numbers) for each operation
3. Import dependencies
4. Aggregated statistics across multiple files

## Integration with FLOP Counting

Static analysis complements runtime FLOP counting by:

1. Identifying potential FLOP-heavy code regions before execution
2. Helping optimize code by highlighting areas that might need attention
3. Providing insights for selective monitoring configuration
