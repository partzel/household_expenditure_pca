import numpy as np


def jacobi_eigen(matrix, tol=1e-10, max_iterations=100):
    a, b = matrix.shape
    if a != b:
        raise ValueError("Matrix must be square")

    n = a
    eigenvectors = np.eye(n)
    A = matrix.astype(float)

    for _ in range(max_iterations):
        max_off_diag = 0
        p, q = 0, 1

        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_off_diag:
                    max_off_diag = abs(A[i, j])
                    p, q = i, j

        if max_off_diag < tol:
            break

        if A[p, p] == A[q, q]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan2(2 * A[p, q], A[p, p] - A[q, q])

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        J = np.eye(n)
        J[p, p] = cos_theta
        J[q, q] = cos_theta
        J[p, q] = -sin_theta
        J[q, p] = sin_theta

        A = J.T @ A @ J
        eigenvectors = eigenvectors @ J

    eigenvalues = np.diag(A)
    return eigenvalues, eigenvectors