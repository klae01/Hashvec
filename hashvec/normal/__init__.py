import jax
from .__import__ import *
from .utils import random_normal, normalize, simple_rotation_matrix
from .optimizer import optimizer


@jax.jit
def rotate(P, V):
    P /= np.linalg.norm(P, ord=2, axis=-1, keepdims=True)
    θ = np.linalg.norm(V, ord=2, axis=-1, keepdims=True)
    pos = P * np.cos(θ) + V * np.sinc(θ / np.pi)
    return pos


def cost(normal):
    X = np.square(np.einsum("im,jm->ij", normal, normal))
    X = X.sum() - np.einsum("ii", X)
    return X


grad = jax.jit(jax.value_and_grad(cost))


@partial(jax.jit, static_argnums=(0,))
def __uniform_normal_vector(opt, normal, n, learning_rate):
    # Gradient
    cost_v, grad_v = map(lambda x: x / (n * (n - 1)), grad(normal))
    eff = opt.update(normal, grad_v)

    step = learning_rate * eff

    new_normal = rotate(normal, step)
    ROT = simple_rotation_matrix(normal, new_normal)

    normal = new_normal
    opt.M = np.einsum("ij,ijk->ik", opt.M, ROT)
    opt.S = ((opt.S[:, None] * abs(ROT)) ** 2).sum(axis=1) ** 0.5

    return cost_v, step, new_normal


def uniform_normal_vector(
    n: int, m: int, steps: int = 500, learning_rate=3000.0, verbose=False
):
    # n dimension
    # m normal vector
    normal = normalize(random_normal(0, 1, (n, m)))
    opt = optimizer()

    for iters in range(steps):
        # scheduler
        step_size = (
            (iters + 0.1) / (steps / 3)
            if iters < steps / 3
            else (steps - iters) / (2 * steps / 3) * 0.9 + 0.1
        )
        step_size *= -learning_rate

        cost_v, step, normal = __uniform_normal_vector(opt, normal, n, step_size)

        if verbose and iters % 10 == 0:
            print(
                cost_v,
                np.linalg.norm(normal, ord=2, axis=-1).mean(),
                abs(step).min(),
                abs(step).max(),
            )
    return normal


# uniform_normal_vector(100, 3, 200, 0.007)
# uniform_normal_vector(1000, 100, 2000, 0.5) # 0.009009007
# uniform_normal_vector(102, 100, 2000, 0.04) # 0.00019802
