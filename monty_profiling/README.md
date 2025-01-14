# Floppy: FLOP Analysis and Counting Framework

Floppy is a framework for analyzing and counting floating-point operations (FLOPs) in Python code, with special focus on numerical computations using NumPy, SciPy, and scikit-learn.

This repository was developed in part during for the Demonstrating Monty Capabilities project.

## Overview

This framework provides multiple approaches to FLOP counting in the `floppy.flop_counting` module.

1. Operation Interception via TrackedArray wrapper
2. Function wrapping for high-level operations
3. Manual FLOP counting for complex operations

Multiple approaches are necessary because numerical operations in Python are implemented in different ways:

1. NumPy's low-level operations are implemented through the ufunc system, which we can intercept using TrackedArray's `__array_ufunc__` interface
2. Higher-level functions like `np.matmul` or `np.linalg.norm` don't use ufuncs, so we need explicit function wrapping to count their FLOPs
3. Complex operations from SciPy and scikit-learn (like KD-tree queries) are implemented by overriding methods in Monty directly, because these are harder to intercept.

In addition, it contains code for static code analysis in `floppy.flop_analysis` to automatically identify operations that contribute to potential FLOP operations.

## Counting Approaches

### TrackedArray Wrapper

Intercepts NumPy array operations through the `__array_ufunc__` interface to count FLOPs for:

- Basic arithmetic operations (+, -, *, /)
- Element-wise operations (np.add, np.multiply, etc.)
- Broadcasting operations
- Reduction operations (sum, mean)

Example:

```python
from floppy.flop_counting.counter import FlopCounter

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
from floppy.flop_counting.counter import FlopCounter

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

## Dependencies

Floppy requires the same dependencies as Monty because it is running Monty code. There are no additional dependencies for counting.

## Usage

Execute Monty the same way in this repository. Floppy adjusted the run.py script to use the FlopCounter.

```
cd ~/tbp/monty_labs/monty_profiling
python run.py -e <experiment_name>
```

## Results

Results are saved in `~/tbp/monty_lab/monty_profiling/results/flop_traces.csv`.

## Running Tests

Add the directory to the Python path:

```bash
export PYTHONPATH=$PYTHONPATH:~/tbp/monty_labs/monty_profiling
```

To run the tests, use:

```bash
python tests/test_add.py
```

**Note:** The tests fail when using `pytest`. I think it is because pytest handles imports and module state differently from running the script directly, and cannot interfere with FlopCounter's monkey-patching.

## Operations Not Yet Supported

- [1/2] Method calls, e.g. `a.sum()` (partially supported for ufuncs like `arr.add(arr2)`, but not for methods)
- einsum
- linalg.solve
