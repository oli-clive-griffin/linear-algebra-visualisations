import numpy as np
from plot import row_picture_3d

def gaussian_elimination(A, b):
    A = A.astype(float)  # Ensure A is of float type for operations
    b = b.astype(float)  # Ensure b is of float type for operations
    n = A.shape[0]  # Number of equations
    aug = np.hstack((A, np.expand_dims(b, axis=1)))

    stages = [(A.copy(), b.copy())]

    # Forward elimination to get an upper triangular matrix
    for i in range(n):
        for j in range(i+1, n):
            ratio = aug[j][i] / aug[i][i]
            aug[j] -= ratio * aug[i]
            stages.append((aug[:, :-1].copy(), aug[:, -1].copy()))

    # Back substitution to get a diagonal matrix
    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            ratio = aug[j][i] / aug[i][i]
            aug[j] -= ratio * aug[i]
            stages.append((aug[:, :-1].copy(), aug[:, -1].copy()))

    # Normalize diagonal to 1
    for i in range(n):
        b[i] /= A[i][i]
        A[i][i] = 1

    return A, b, stages

# Example usage
A = np.array(
    [
        [4, 1, -1],
        [-1, -3, 2],
        [-0.2, 2, 3],
    ], dtype=float)
b = np.array([8, -11, -3], dtype=float)
sol = np.linalg.solve(A, b)

A_, b_, stages = gaussian_elimination(A, b)

for i, (A, b) in enumerate(stages):
    assert np.allclose(np.linalg.solve(A, b), sol), "Solution is not valid" + str(np.linalg.solve(A, b)) + str(i)

for A, b in stages:
    row_picture_3d(A, b).show()

# print("Diagonal A after Gaussian Elimination and Back Substitution:\n", A_)
# print("Vector b representing the solution x:\n", b_)

# # Example usage
# A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
# b = np.array([8, -11, -3], dtype=float)

# A_, b_ = gaussian_elimination(A, b)

# print('aug 1\n', np.hstack((A, np.expand_dims(b, axis=1))))
# print('aug 2\n', np.hstack((A_, np.expand_dims(b_, axis=1))))

# # validate the solution
# x = np.linalg.solve(A, b)
# x_ = np.linalg.solve(A_, b_)

# print("Solution (x):\n", x)
# print("Solution (x_):\n", x_)
# print("Are solutions equal?", np.allclose(x, x_))
