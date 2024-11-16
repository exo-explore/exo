import mlx.core as mx
import numpy as np

# Create some test arrays
a = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.float32)
b = mx.array([[7, 8, 9], [10, 11, 12]], dtype=mx.float32)

# Test basic operations
print("Array a:")
print(a)
print("\nArray b:")
print(b)

# Test multiplication
c = mx.multiply(a, b)
print("\nElement-wise multiplication (a * b):")
print(c)

# Test matrix multiplication
d = a @ b.transpose()
print("\nMatrix multiplication (a @ b.T):")
print(d)

# Test converting to numpy
print("\nConvert to numpy array:")
numpy_array = np.array(c)
print(numpy_array)
