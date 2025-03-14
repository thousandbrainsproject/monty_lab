# Floppy: FLOP Analysis and Counting Framework

Floppy is a framework for analyzing and counting floating-point operations (FLOPs) in Python code, with special focus on numerical computations using NumPy, SciPy, and scikit-learn.

## Features

- **Runtime FLOP Counting**: Track FLOPs in your code as it executes
  - Transparent array operation tracking
  - Support for high-level NumPy/SciPy operations
  - Thread-safe operation
  - Selective monitoring of code paths

- **Static Analysis**: Analyze potential FLOP operations in your codebase
  - Identify FLOP-heavy code regions
  - Track numerical computing dependencies
  - Get insights before execution

## Quick Start

```python
from floppy.counting.core import FlopCounter

with FlopCounter() as counter:
    result = np.matmul(a, b)
    print(f"FLOPs: {counter.flops}")
```

## Documentation

For detailed documentation, see the `docs/` directory:

- [Getting Started](docs/user_guide/getting_started.md)
- [FLOP Counting Guide](docs/user_guide/flop_counting.md)
- [Static Analysis](docs/user_guide/static_analysis.md)
- [API Reference](docs/api/index.md)

## Dependencies

Floppy requires the same dependencies as Monty because it is running Monty code. There are no additional dependencies for counting.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](docs/contributing.md) for details.

## License

[Add your chosen license here]

## Overview

This framework provides multiple approaches to FLOP counting in the `floppy.counting` module.

1. Operation Interception via TrackedArray wrapper
2. Function wrapping for high-level operations
3. Manual FLOP counting for complex operations

Multiple approaches are necessary because numerical operations in Python are implemented in different ways:

1. NumPy's low-level operations are implemented through the ufunc system, which we can intercept using TrackedArray's `__array_ufunc__` interface
2. Higher-level functions like `np.matmul` or `np.linalg.norm` don't use ufuncs, so we need explicit function wrapping to count their FLOPs
3. Complex operations from SciPy and scikit-learn (like KD-tree queries) are implemented by overriding methods in Monty directly, because these are harder to intercept.

In addition, it contains code for static code analysis in `floppy.analysis` to automatically identify operations that contribute to potential FLOP operations.

## Core Components

### FlopCounter

The `FlopCounter` is the central component that manages FLOP counting across all operations. It provides:

1. **Context Management**: Used as a context manager or decorator to track FLOPs within a scope:

   ```python
   with FlopCounter() as counter:
       result = np.dot(array1, array2)
       print(f"FLOPs: {counter.flops}")
   ```

2. **Thread-Safe Operation**: All FLOP counting is thread-safe, allowing use in multi-threaded environments.

3. **Selective Monitoring**:
   - `skip_paths`: Exclude specific code paths from FLOP counting
   - `include_paths`: Override skip_paths for specific paths
   - Useful for focusing on application code vs library code

4. **Operation Registry**:
   - Maintains a registry of operations and their FLOP counting rules
   - Each operation (add, multiply, exp, etc.) has its own counting logic
   - Easily extensible for new operations

5. **Monkey-Patching System**:
   - Temporarily patches NumPy functions during the counting context
   - Automatically wraps arrays in TrackedArray
   - Restores original functionality after context exit

### TrackedArray

`TrackedArray` is a NumPy array subclass that enables transparent FLOP counting. Key features:

1. **Transparent Operation Tracking**:
   - Inherits from `np.ndarray`
   - Preserves all NumPy array functionality
   - Automatically tracks operations without user intervention

   ```python
   with FlopCounter() as counter:
       a = np.array([1, 2, 3])  # Automatically wrapped as TrackedArray
       b = a + 1  # One FLOP per element is counted
   ```

2. **Operation Interception**:
   - Implements `__array_ufunc__` for universal function tracking
   - Handles all NumPy operations (arithmetic, math functions, etc.)
   - Maintains tracking through array operations (slicing, reshaping)

3. **Smart FLOP Counting**:
   - Counts FLOPs based on operation type and array size
   - Handles broadcasting and reduction operations correctly
   - Supports both scalar and array operations

4. **Zero Overhead When Inactive**:
   - No performance impact when counter is not active
   - Efficient unwrapping of nested TrackedArrays
   - Caches wrapped methods for better performance

5. **Comprehensive Operation Support**:
   - Basic arithmetic (+, -, *, /)
   - Mathematical functions (sqrt, exp, log)
   - Linear algebra operations (dot, matmul)
   - Reductions (sum, mean)
   - Universal functions (ufuncs)

## Static Code Analysis

The static code analysis is implemented in `floppy.analysis.analyzer.py`. It uses Python's `ast` module to parse source code and identify operations that could contribute to FLOP operations. The analyzer tracks:

### Function Calls

- **NumPy Operations**: Tracks all NumPy function calls, including ufuncs, linear algebra operations, and array manipulations
- **SciPy Operations**: Identifies SciPy function calls, particularly from spatial and linear algebra modules
- **scikit-learn Operations**: Captures machine learning operations that may involve significant numerical computations

### Import Analysis

Tracks all imports related to numerical computing libraries to understand dependencies and potential FLOP sources:

- NumPy imports (e.g., `import numpy as np`, `from numpy import array`)
- SciPy imports (e.g., `from scipy.spatial import KDTree`)
- scikit-learn imports (e.g., `from sklearn.neighbors import NearestNeighbors`)

### Usage Example

Analyze a file:

```python
from floppy.flop_analysis.core.analyzer import FlopAnalyzer

analyzer = FlopAnalyzer()
analyzer.analyze_file('example_code.py')
```

Analyze a directory:

```python
analyzer.analyze_directory('path/to/code')
```

The analysis results include:

- File-level breakdown of numerical operations
- Location information (line numbers) for each operation
- Import dependencies
- Aggregated statistics across multiple files

This static analysis complements the runtime FLOP counting by helping identify where FLOPs might occur in the codebase, even before execution.

## Counting Approaches

### TrackedArray Wrapper

Intercepts NumPy array operations through the `__array_ufunc__` interface to count FLOPs for:

- Basic arithmetic operations (+, -, *, /)
- Element-wise operations (np.add, np.multiply, etc.)
- Broadcasting operations
- Reduction operations (sum, mean)

Example:

```python
from floppy.counting.core import FlopCounter

with FlopCounter() as counter:
    a = np.array([[1, 2], [3, 4]])  # Automatically wrapped as TrackedArray
    b = a + 1  # Counts one FLOP per element
```

### Function Wrapping

Handles higher-level NumPy/SciPy operations through explicit wrappers:

- Matrix multiplication (np.matmul, @)
- Linear algebra operations (np.linalg.norm, inv, etc.)
- Statistical operations (mean, std, var)
- Trigonometric functions

Example:

```python
from floppy.counting.core import FlopCounter

with FlopCounter() as counter:
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result = np.matmul(a, b)  # Counts 2*M*N*P FLOPs
```

### Individually Wrapped Complex Operations

Manual FLOP counting for the following methods in Monty:

- `tbp.monty.frameworks.models.evidence_matching.EvidenceGraphLM._update_evidence_with_vote`
- `tbp.monty.frameworks.models.evidence_matching.EvidenceGraphLM._calculate_evidence_for_new_locations`
- `tbp.monty.frameworks.models.goal_state_generation.EvidenceGoalStateGenerator._compute_graph_mismatch`

The FLOP counting for these operations is done by inheriting from the above classes (e.g. `EvidenceGraphLM` as `FlopCounterEvidenceGraphLM`) and overriding the above methods to include FLOP counting.

Example:

```python
class FlopCounterEvidenceGraphLM(EvidenceGraphLM):

    def _update_evidence_with_vote(self, *args, **kwargs):
        # ... existing code ...
        ... = tree.query(...)

        # Custom code to count FLOPs for KDTree query
        num_search_points = ...
        num_reference_points = ...
        ...
        kdtree_query_flops = ...
        self.flop_counter.add_flops(kdtree_query_flops)

        # ... remainder ofexisting code ...
```

#### FLOPs for KDTree Construction and Query

KDTree operations are one of the key components we track in Monty's evidence matching system.

**KDTree Construction:**
The construction of a k-d tree has a complexity of $O(kn \log_2(n))$ FLOPs, where:

- $n$ is the number of points in the dataset
- $k$ is the number of dimensions
- $\log_2(n)$ represents the average depth of the tree

For each level of the tree ($\log_2(n)$ levels), we need to:

1. Find the median along the current dimension ($O(n)$ operations)
2. Partition the points ($O(kn)$ operations to compare k-dimensional points)

**KDTree Query:**
For querying nearest neighbors, our implementation breaks down FLOP counting into several components. Note that we assume a balanced tree structure, which is the default behavior in [SciPy's KDTree implementation (balanced_tree=True)](https://docs.scipy.org/doc/scipy-1.15.0/reference/generated/scipy.spatial.KDTree.html):

1. **Tree Traversal:**
   - FLOPs = num_search_points × dim × log₂(num_reference_points)
   - Represents operations needed to traverse the tree to the appropriate leaf nodes
   - This logarithmic complexity is guaranteed by the balanced tree structure

2. **Distance Calculations:**
   - FLOPs = num_search_points × num_examined_points × (3 $\times$ dim + dim + 1)
   - Where num_examined_points = log₂(num_reference_points) due to balanced tree property
   - 3 operations per dimension (subtract, square, add)
   - dim additions for summing
   - 1 square root operation

3. **Heap Operations:**
   - FLOPs = num_search_points × num_examined_points × log₂(k)
   - Where k is the number of nearest neighbors requested (vote_nn)
   - Maintains priority queue for k-nearest neighbors

4. **Bounding Box Checks:**
   - FLOPs = num_search_points × num_examined_points × dim
   - Represents comparisons against bounding box boundaries

Total query FLOPs = traversal_flops + distance_flops + heap_flops + bounding_box_flops

Where:

- num_search_points: number of query points
- num_reference_points: number of points in the KD-tree
- dim: dimensionality of the points
- num_examined_points: estimated as log₂(num_reference_points)

Note: These are theoretical approximations. Actual FLOP counts may vary based on:

- Data distribution
- Tree balance
- Search radius/nearest neighbor parameters
- Optimizations in the underlying SciPy implementation

## Usage

To count FLOPs in Monty:

```bash
cd ~/tbp/monty_labs/floppy
python run_flop_counter.py -e <experiment_name>
```

To count FLOPs in your own code:

```python
from floppy.counting.core import FlopCounter

# Basic usage
with FlopCounter() as counter:
    # Your numerical computations here
    result = np.matmul(a, b)
    print(f"FLOPs: {counter.flops}")

# With detailed logging
from logging import getLogger
logger = getLogger("flop_counter")
with FlopCounter(logger=logger) as counter:
    result = np.linalg.norm(vector)
    print(f"FLOPs: {counter.flops}")
```

To analyze FLOP operations in source code:

```bash
python run_static_analysis.py --dir path/to/analyze
```
