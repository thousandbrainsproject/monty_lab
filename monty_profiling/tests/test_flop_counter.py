# import numpy as np
# from monty_profiling.floppy.flop_counting.counter import FlopCounter, TrackedArray

# def test_basic_arithmetic():
#     """Test basic arithmetic operations with explicit FLOP counting."""
#     counter = FlopCounter()

#     with counter:
#         a = np.array([[1, 2], [3, 4]])
#         b = np.array([[5, 6], [7, 8]])
#         # Test division
#         c = a / b
#         print()
#         # Test addition
#         c = a + b
#         assert isinstance(c, TrackedArray)
#         assert counter.flops == 4  # One FLOP per element
#         assert np.array_equal(c.array, a.array + b.array)
#         counter.flops = 0

#         # Test multiplication
#         d = a * b
#         assert isinstance(d, TrackedArray)
#         assert counter.flops == 4
#         assert np.array_equal(d.array, a.array * b.array)
#         counter.flops = 0

#         # Test mixed scalar operations
#         e = 2 * a
#         assert isinstance(e, TrackedArray)
#         assert counter.flops == 4
#         assert np.array_equal(e.array, 2 * a.array)


# def test_numpy_functions():
#     """Test NumPy function integration with TrackedArray."""
#     counter = FlopCounter()

#     with counter:
#         a = np.array([[1, 2], [3, 4]])
#         b = np.array([[5, 6], [7, 8]])

#         # Test np.add
#         c = np.add(a, b)
#         assert isinstance(c, TrackedArray)
#         assert counter.flops == 4
#         assert np.array_equal(c.array, np.add(a.array, b.array))
#         counter.flops = 0

#         # Test np.multiply
#         d = np.multiply(a, b)
#         assert isinstance(d, TrackedArray)
#         assert counter.flops == 4
#         assert np.array_equal(d.array, np.multiply(a.array, b.array))


# def test_matrix_operations():
#     """Test matrix operations with proper FLOP counting."""
#     counter = FlopCounter()

#     with counter:
#         a = np.array([[1, 2], [3, 4]])
#         b = np.array([[5, 6], [7, 8]])

#         # Test matrix multiplication
#         c = a @ b
#         assert isinstance(c, TrackedArray)
#         # 2 * 2 * 2 multiplications and 2 * 2 additions = 12 FLOPs
#         assert counter.flops == 12
#         assert np.array_equal(c.array, a.array @ b.array)
#         counter.flops = 0

#         # Test np.matmul
#         d = np.matmul(a, b)
#         assert isinstance(d, TrackedArray)
#         assert counter.flops == 12
#         assert np.array_equal(d.array, np.matmul(a.array, b.array))

# def test_broadcasting():
#     """Test operations with broadcasting."""
#     counter = FlopCounter()

#     with counter:
#         a = np.array([[1, 2], [3, 4]])
#         v = np.array([1, 2])

#         # Test broadcasting with addition
#         c = a + v
#         assert isinstance(c, TrackedArray)
#         assert counter.flops == 4  # Still one FLOP per output element
#         expected = a.array + v.array
#         assert np.array_equal(c.array, expected)
#         counter.flops = 0

#         # Test broadcasting with multiplication
#         d = a * v
#         assert isinstance(d, TrackedArray)
#         assert counter.flops == 4
#         assert np.array_equal(d.array, a.array * v.array)

# def test_reduction_operations():
#     """Test reduction operations like sum and mean."""
#     counter = FlopCounter()

#     with counter:
#         a = np.array([[1, 2], [3, 4]])

#         # Test sum
#         s = np.sum(a)
#         assert isinstance(s, (TrackedArray, np.number))
#         assert counter.flops == 3  # n-1 additions for n elements
#         assert s.array == np.sum(a.array)
#         counter.flops = 0

#         # Test mean
#         m = np.mean(a)
#         assert isinstance(m, (TrackedArray, np.number))
#         assert counter.flops == 4  # n-1 additions + 1 division
#         assert m.array == np.mean(a.array)

# def test_chained_operations():
#     """Test chained operations maintain proper tracking."""
#     counter = FlopCounter()

#     with counter:
#         a = np.array([[1, 2], [3, 4]])
#         b = np.array([[5, 6], [7, 8]])

#         # Test multiple operations
#         c = (a + b) * (a @ b)
#         assert isinstance(c, TrackedArray)
#         # 4 (add) + 12 (matmul) + 4 (multiply) = 20 FLOPs
#         assert counter.flops == 20
#         assert np.array_equal(c.array, (a.array + b.array) * (a.array @ b.array))

# def test_unsupported_operations():
#     """Test unsupported operations."""
#     counter = FlopCounter()

#     with counter:
#         a = np.array([[1, 2], [3, 4]])
#         b = np.array([[5, 6], [7, 8]])
#         tranpose = a.T
#         assert isinstance(tranpose, TrackedArray)
#         assert counter.flops == 0
#         assert np.array_equal(tranpose.array, a.array.T)

# if __name__ == "__main__":
#     test_basic_arithmetic()
#     test_numpy_functions()
#     test_matrix_operations()
#     test_broadcasting()
#     test_reduction_operations()
#     test_chained_operations()
