from .__import__ import *
import numpy as onp


def random_normal(*args, **kwargs):
    return np.asarray(onp.random.normal(*args, **kwargs), dtype=compute_dtype)


def normalize(X):
    norm = np.linalg.norm(X, ord=2, axis=-1, keepdims=True)
    return X / (norm + (norm == 0))


def random_rot_mat(dim):
    RET = onp.eye(dim)
    for _ in range(1000):
        ROT = onp.eye(dim)
        ax1, ax2 = onp.random.randint(0, dim, [2])
        θ = onp.random.uniform(-np.pi, np.pi) * 0.001
        if ax1 != ax2:
            ROT[ax1, ax1] = onp.cos(θ)
            ROT[ax1, ax2] = onp.sin(θ)
            ROT[ax2, ax1] = -onp.sin(θ)
            ROT[ax2, ax2] = onp.cos(θ)
        RET = RET @ ROT
    return RET


def simple_rotation_matrix(A, B):
    dim = A.shape[-1]
    A_OTH = B - np.einsum("ij,ik,ik->ij", A, A, B)
    A, B, A_OTH = map(normalize, (A, B, A_OTH))

    prop = np.einsum("ij,ik->ijk", A, A) + np.einsum("ij,ik->ijk", A_OTH, A_OTH)
    ind = np.eye(dim, dtype=compute_dtype)[None, ...] - prop
    proj_size = np.sqrt(np.square(A_OTH) + np.square(A))
    proj = normalize(prop)

    δ = (
        np.sign(A_OTH) * np.arccos(np.clip(np.einsum("ijk,ik->ij", proj, A), -1, 1))
        + np.arccos(np.clip(np.einsum("ij,ij->i", A, B), -1, 1))[:, None]
    )

    return ind + proj_size[..., None] * (
        np.einsum("ij,ik->ijk", np.cos(δ), A)
        + np.einsum("ij,ik->ijk", np.sin(δ), A_OTH)
    )
