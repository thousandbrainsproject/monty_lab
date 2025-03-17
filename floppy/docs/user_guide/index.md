# User Guide

## Overview

Floppy provides multiple approaches to FLOP counting in the `floppy.counting` module:

1. Operation Interception via TrackedArray wrapper
2. Function wrapping for high-level operations
3. Manual FLOP counting for complex operations

Multiple approaches are necessary because numerical operations in Python are implemented in different ways:

1. NumPy's low-level operations are implemented through the ufunc system, which we can intercept using TrackedArray's `__array_ufunc__` interface
2. Higher-level functions like `np.matmul` or `np.linalg.norm` don't use ufuncs, so we need explicit function wrapping to count their FLOPs
3. Complex operations from SciPy and scikit-learn (like KD-tree queries) are implemented by overriding methods in Monty directly, because these are harder to intercept.

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

### Function Wrapping

Handles higher-level NumPy/SciPy operations through explicit wrappers:

- Matrix multiplication (np.matmul, @)
- Linear algebra operations (np.linalg.norm, inv, etc.)
- Statistical operations (mean, std, var)
- Trigonometric functions

Example:

```python
from floppy.counting.base import FlopCounter

with FlopCounter() as counter:
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result = np.matmul(a, b)  # Counts 2*M*N*P FLOPs
```

### Individually Wrapped Complex Operations (KDTree)

Manual FLOP counting for the following methods in Monty:

- `tbp.monty.frameworks.models.evidence_matching.EvidenceGraphLM._update_evidence_with_vote`
- `tbp.monty.frameworks.models.evidence_matching.EvidenceGraphLM._calculate_evidence_for_new_locations`
- `tbp.monty.frameworks.models.goal_state_generation.EvidenceGoalStateGenerator._compute_graph_mismatch`

The FLOP counting for these operations is done by inheriting from the above classes (e.g. `EvidenceGraphLM` as `FlopCounterEvidenceGraphLM`) and overriding the above methods to include FLOP counting.

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
   - FLOPs = num_search_points × num_examined_points × (3 × dim + dim + 1)
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

The analysis results include:

- File-level breakdown of numerical operations
- Location information (line numbers) for each operation
- Import dependencies
- Aggregated statistics across multiple files

This static analysis complements the runtime FLOP counting by helping identify where FLOPs might occur in the codebase, even before execution.
