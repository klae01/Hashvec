from .__import__ import *


def random_normal(offset=0, scale=1, shape=[]):
    if use_jax:
        return (
            jax.random.normal(
                jax.random.PRNGKey(123), tuple(shape), dtype=compute_dtype
            )
            * scale
            + offset
        )
    return np.asarray(onp.random.normal(offset, scale, shape), dtype=compute_dtype)


def normalize(X):
    norm = np.linalg.norm(X, ord=2, axis=-1, keepdims=True)
    return X / (norm + (norm == 0))


def random_rot_mat(dim, angle_sample=1000, angle_max=0.001 * np.pi):
    ROT = onp.eye(dim)
    AX = onp.random.randint(0, dim, [angle_sample, 2])
    θ = onp.random.uniform(-angle_max, angle_max, angle_sample)

    rot_mat = onp.transpose(
        [
            [onp.cos(θ), onp.sin(θ)],
            [-onp.sin(θ), onp.cos(θ)],
        ],
        [2, 0, 1],
    )

    for ax, rot in zip(AX, rot_mat):
        if ax[0] != ax[1]:
            ROT[ax] = rot @ ROT[ax]
    return ROT


@jax.jit
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
