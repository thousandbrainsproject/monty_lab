# FLOP Counting Guide

Floppy provides multiple approaches to FLOP counting, each suited for different types of operations.

## Counting Approaches

### 1. TrackedArray Wrapper

Intercepts NumPy array operations through the `__array_ufunc__` interface to count FLOPs for:

- Basic arithmetic operations (+, -, *, /)
- Element-wise operations (np.add, np.multiply, etc.)
- Broadcasting operations
- Reduction operations (sum, mean)

Example:

```python
from floppy.counting.base import FlopCounter

with FlopCounter() as counter:
    a = np.array([[1, 2], [3, 4]])  # Automatically wrapped as TrackedArray
    b = a + 1  # Counts one FLOP per element
```

### 2. Function Wrapping

Handles higher-level NumPy/SciPy operations through explicit wrappers:

- Matrix multiplication (np.matmul, @)
- Linear algebra operations (np.linalg.norm, inv, etc.)
- Statistical operations (mean, std, var)
- Trigonometric functions

Example:

```python
with FlopCounter() as counter:
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result = np.matmul(a, b)  # Counts 2*M*N*P FLOPs
```

### 3. Complex Operations

For complex operations like KD-tree queries, we provide detailed FLOP counting based on theoretical complexity:

#### KDTree Operations

**Construction:**

- Complexity: O(kn log₂(n)) FLOPs
- n: number of points
- k: number of dimensions

**Query:**
Total query FLOPs = traversal_flops + distance_flops + heap_flops + bounding_box_flops

Where:

- traversal_flops = num_search_points × dim × log₂(num_reference_points)
- distance_flops = num_search_points × num_examined_points × (3 × dim + dim + 1)
- heap_flops = num_search_points × num_examined_points × log₂(k)
- bounding_box_flops = num_search_points × num_examined_points × dim

## Advanced Usage

### Selective Monitoring

You can exclude or include specific code paths:

```python
with FlopCounter(skip_paths=['library_code'], include_paths=['my_app']) as counter:
    # Only counts FLOPs in 'my_app' paths
    result = compute_something()
```

### Thread Safety

All FLOP counting is thread-safe, allowing use in multi-threaded environments.

See [Known Issues](../advanced_topics/known_issues.md) for more details.
