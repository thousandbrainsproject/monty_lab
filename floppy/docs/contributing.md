# Contributing to Floppy

Thank you for your interest in contributing to Floppy! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Clone the repository
2. Install development dependencies (same as Monty's dependencies)
3. Set up your Python environment

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write descriptive docstrings in NumPy format
- Include unit tests for new functionality

## Adding Support for New Operations

One of the most common contributions is adding FLOP counting support for new NumPy/SciPy operations. This is done through the Protocol system.

### Understanding the Protocol System

The Protocol system defines how FLOP counting should be performed for different types of operations. Each Protocol implementation specifies:

1. What operations it handles
2. How to count FLOPs for those operations
3. How to wrap/unwrap arrays for tracking

### Example: Adding a New Protocol

Here's an example of adding support for a new NumPy function:

```python
from floppy.counting.protocols import BaseProtocol
from floppy.counting.utils import count_flops

class MyNewFunctionProtocol(BaseProtocol):
    """Protocol for handling my_new_function."""

    def __init__(self):
        super().__init__()
        self.handled_functions = {np.my_new_function}

    def count_operation_flops(self, *args, **kwargs):
        """Count FLOPs for my_new_function.
        
        Parameters
        ----------
        *args, **kwargs : 
            Arguments passed to my_new_function
            
        Returns
        -------
        int
            Number of FLOPs
        """
        # Example: if the function performs one operation per element
        input_array = args[0]
        return count_flops(input_array.size)

    def wrap_args(self, *args, **kwargs):
        """Wrap input arrays in TrackedArray."""
        # Typically you'll want to wrap numpy arrays
        wrapped_args = [
            self.wrap(arg) if isinstance(arg, np.ndarray) else arg
            for arg in args
        ]
        return wrapped_args, kwargs
```

### Steps to Add New Operation Support

1. **Identify the Operation Type**
   - Is it a ufunc?
   - Is it a high-level function?
   - What are its inputs and outputs?

2. **Create a Protocol Class**
   - Inherit from `BaseProtocol`
   - Define `handled_functions`
   - Implement `count_operation_flops`
   - Implement `wrap_args` if needed

3. **Calculate FLOPs**
   - Document your FLOP counting methodology
   - Consider all edge cases (broadcasting, dtype, etc.)
   - Use helper functions from `floppy.counting.utils`

4. **Add Tests**
   - Test basic functionality
   - Test edge cases
   - Test FLOP counting accuracy

### Example: Matrix Operation Protocol

Here's a real-world example for matrix operations:

```python
class MatrixOperationProtocol(BaseProtocol):
    """Protocol for matrix operations like matmul."""

    def __init__(self):
        super().__init__()
        self.handled_functions = {np.matmul, np.dot}

    def count_operation_flops(self, *args, **kwargs):
        """Count FLOPs for matrix multiplication.
        
        For matrix multiplication of (M,N) @ (N,P):
        - Each element requires N multiplications and N-1 additions
        - Total FLOPs = M*N*P*2
        """
        a, b = args
        if a.ndim == 1 and b.ndim == 1:
            # Dot product
            return a.size * 2
        else:
            # Matrix multiplication
            m, n = a.shape
            p = b.shape[1]
            return m * n * p * 2

    def wrap_args(self, *args, **kwargs):
        """Wrap input matrices."""
        wrapped_args = [
            self.wrap(arg) if isinstance(arg, np.ndarray) else arg
            for arg in args
        ]
        return wrapped_args, kwargs
```

## Testing Your Changes

1. Add unit tests in `tests/`
2. Run the test suite:

   ```bash
   python tests/test_your_feature.py
   ```

## Submitting Changes

1. Create a new branch for your changes
2. Write clear commit messages
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## Documentation

When adding new features:

1. Add docstrings to all new functions and classes
2. Update relevant documentation in `docs/`
3. Add examples if appropriate
4. Document any new dependencies

## Questions?

If you have questions about contributing, please:

1. Check existing documentation
2. Look through existing protocols for examples
3. Open an issue for discussion

Thank you for contributing to Floppy!
