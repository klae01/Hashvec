import numpy as np

def normalize(X):
    return X/np.linalg.norm(X, ord=2, axis = -1, keepdims=True)
def rotate(P, V):
    θ = np.linalg.norm(V, ord=2, axis = -1, keepdims=True)
    limit = 0.01
    V[θ[..., 0] > limit] *= limit / θ[θ > limit][..., None]
    θ = np.minimum(θ, limit)
    return P*np.cos(θ) + V/θ*np.sin(θ)

def uniform_normal_vector(n: int, m: int, steps: int = 2000, learning_rate = 0.01):
    normal = normalize(np.random.normal(size=(n, m)))
    momentum = 0
    for iters in range(steps):
        ploss = normal@normal.T/n
        deriv = ploss@normal
        ortho = deriv - normal * np.einsum("im,im->i",normal, deriv)[..., None]

        momentum = momentum * 0.9 + ortho * 0.1

        step = -momentum*learning_rate

        normal = rotate(normal, step)
        if iters % 10 == 0:
            normal = normalize(normal)
            print(np.abs(ploss).sum()-1, np.linalg.norm(normal, ord=2, axis = -1, keepdims=True).mean())