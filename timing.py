import numpy as np
import time

def matrix_benchmark():
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    start = time.time()
    for _ in range(50):
        np.dot(a, b)
    print(f"Elapsed: {time.time() - start:.2f} seconds")

matrix_benchmark()