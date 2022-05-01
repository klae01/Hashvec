import jax
import jax.numpy as np
@jax.jit
def normalize(X):
    return X/np.linalg.norm(X, ord=2, axis = -1, keepdims=True)
@jax.jit
def rotate(P, V):
    θ = np.linalg.norm(V, ord=2, axis = -1, keepdims=True)
    # limit = 0.01
    # V[θ[..., 0] > limit] *= limit / θ[θ > limit][..., None]
    # θ = np.minimum(θ, limit)
    return P*np.cos(θ) + V/θ*np.sin(θ)
    
# @jax.jit
# def cost(normal):
#     X=np.triu(np.einsum("im,jm->ij", normal, normal), k=1)
#     X=np.einsum("ij,ij", X, X) # np.square(X).sum()
#     return X / (normal.shape[0] * (normal.shape[0]-1) / 2)

@jax.jit
def cost(normal):
    X=np.square(np.einsum("im,jm->ij", normal, normal))
    X=(X.sum()-np.einsum("ii", X))
    return X / (normal.shape[0] * (normal.shape[0]-1))



grad = jax.grad(cost)

def uniform_normal_vector(n: int, m: int, steps: int = 200, learning_rate = 0.01):
    # n dimension 
    # m normal vector
    normal = normalize(jax.random.normal(jax.random.PRNGKey(123), (n, m)))
    momentum = 0
    for iters in range(steps):
        gradient = grad(normal)
        ortho = gradient - normal * (normal@gradient.T).sum(axis = -1, keepdims = True)

        momentum += (ortho - momentum) * 0.1

        step = (iters + 0.1) / (steps / 3) if iters < steps / 3 else (steps - iters) / (2 * steps / 3) * 0.9 + 0.1
        step *= -momentum * learning_rate
        
        normal = rotate(normal, step)
        normal = normalize(normal)
        if iters % 10 == 0:
            print(cost(normal), np.linalg.norm(normal, ord=2, axis = -1).mean(), abs(step).min(), abs(step).max())
