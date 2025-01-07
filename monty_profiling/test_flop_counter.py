import numpy as np
import pytest
from floppy.flop_counting.counter import FlopCounter


def test_flop_counting():
    """Test FLOP counting for various operations."""
    counter = FlopCounter()

    # Create test arrays
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    v = np.array([1, 2])

    test_cases = [
        # Basic arithmetic (magic methods)
        {
            "name": "Matrix addition",
            "operation": lambda: np.add(a, b),
            "expected_flops": 4,  # One addition per element
            "shape_check": (2, 2),
        },
        {
            "name": "Matrix multiplication",
            "operation": lambda: np.matmul(a, b),
            "expected_flops": 12,  # 2*2*2 multiplications and 2*2 additions
            "shape_check": (2, 2),
        },
        {
            "name": "Element-wise multiplication",
            "operation": lambda: a * b,
            "expected_flops": 4,  # One multiplication per element
            "shape_check": (2, 2),
        },
        # Linear algebra operations
        {
            "name": "Matrix norm",
            "operation": lambda: np.linalg.norm(a),
            "expected_flops": 8,  # 4 squares + 3 additions + 1 sqrt
            "shape_check": (),
        },
        {
            "name": "Matrix inverse",
            "operation": lambda: np.linalg.inv(a),
            "expected_flops": 14,  # ~2/3 * n³ + 2n² for 2x2 matrix
            "shape_check": (2, 2),
        },
        # Trigonometric functions
        {
            "name": "Sine operation",
            "operation": lambda: np.sin(a),
            "expected_flops": 32,  # 8 FLOPs per element (4 elements)
            "shape_check": (2, 2),
        },
        # Statistical operations
        {
            "name": "Mean operation",
            "operation": lambda: np.mean(a),
            "expected_flops": 4,  # n-1 additions + 1 division
            "shape_check": (),
        },
        {
            "name": "Standard deviation",
            "operation": lambda: np.std(a),
            "expected_flops": 17,  # 4n + 1 for n=4
            "shape_check": (),
        },
        # Vector operations
        {
            "name": "Vector dot product",
            "operation": lambda: np.dot(v, v),
            "expected_flops": 3,  # 2 multiplications + 1 addition
            "shape_check": (),
        },
        # Mixed operations
        {
            "name": "Complex expression",
            "operation": lambda: np.sin(a @ b) + np.cos(a),
            "expected_flops": 76,  # matmul(12) + sin(32) + cos(32)
            "shape_check": (2, 2),
        },
    ]

    for case in test_cases:
        counter.flops = 0
        with counter:
            result = case["operation"]()
        flop_count = counter.flops

        # Verify result shape
        assert (
            result.shape == case["shape_check"]
        ), f"{case['name']}: Expected shape {case['shape_check']}, got {result.shape}"

        # Verify FLOP count using stored value
        assert (
            flop_count == case["expected_flops"]
        ), f"{case['name']}: Expected {case['expected_flops']} FLOPs, got {flop_count}"

        # Reset counter for next test
        counter.flops = 0


def test_broadcasting():
    """Test FLOP counting with broadcasting operations."""
    counter = FlopCounter()

    # Create test arrays with different shapes
    a = np.array([[1, 2], [3, 4]])  # (2, 2)
    v = np.array([1, 2])  # (2,)
    s = np.array(2)  # scalar

    broadcast_cases = [
        {
            "name": "Matrix + scalar",
            "operation": lambda: a + s,
            "expected_flops": 4,  # One addition per matrix element
        },
        {
            "name": "Matrix + vector (broadcasting)",
            "operation": lambda: a + v.reshape(2, 1),
            "expected_flops": 4,  # One addition per matrix element
        },
        {
            "name": "Matrix * vector",
            "operation": lambda: a * v,
            "expected_flops": 4,  # One multiplication per element after broadcasting
        },
    ]

    for case in broadcast_cases:
        with counter:
            _ = case["operation"]()
            assert (
                counter.flops == case["expected_flops"]
            ), f"{case['name']}: Expected {case['expected_flops']} FLOPs, got {counter.flops}"
        counter.flops = 0


def test_chained_operations():
    """Test FLOP counting with chained operations."""
    counter = FlopCounter()

    a = np.array([[1, 2], [3, 4]])

    with counter:
        # (A + A) * (A @ A)
        result = (a + a) * (a @ a)

        # Expected FLOPs:
        # - Matrix addition: 4
        # - Matrix multiplication: 12
        # - Final multiplication: 4
        expected_flops = 20

        assert (
            counter.flops == expected_flops
        ), f"Chained operations: Expected {expected_flops} FLOPs, got {counter.flops}"


def test_basic_operations():
    counter = FlopCounter()
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    with counter:
        c = a + b  # Should count 3 FLOPs
        d = a * b  # Should count 3 more FLOPs

    assert counter.flops == 6


if __name__ == "__main__":
    test_flop_counting()
