import jax
import jax.numpy as np
@jax.jit
def normalize(X):
    return X/np.linalg.norm(X, ord=2, axis = -1, keepdims=True)
    
@jax.jit
def rotate(P, V):
    P /= np.linalg.norm(P, ord=2, axis = -1, keepdims=True)
    θ = np.linalg.norm(V, ord=2, axis = -1, keepdims=True)
    return P*np.cos(θ) + V*np.sinc(θ/np.pi)

@jax.jit
def cost(normal):
    X=np.square(np.einsum("im,jm->ij", normal, normal))
    X=(X.sum()-np.einsum("ii", X))
    return X / (normal.shape[0] * (normal.shape[0]-1))

grad = jax.grad(cost)

def uniform_normal_vector(n: int, m: int, steps: int = 500, learning_rate = 0.25):
    # n dimension 
    # m normal vector
    normal = normalize(jax.random.normal(jax.random.PRNGKey(123), (n, m)))
    β1 = 0.75
    β2 = 0.9975
    Ɛ = 1e-6
    M = 0
    S = 0
    for iters in range(steps):
        gradient = grad(normal)

        M += (1-β1) * (gradient - M)
        S += (1-β2) * (np.square(gradient-S) - S) + Ɛ

        eff = M / (1-β1**(1 + iters))
        eff /= (S / (1-β2**(1 + iters)))**0.5 + Ɛ

        eff -= np.einsum("ij,ik,ik->ij", normal, normal, eff)
        debug = abs(np.einsum("ij,ij->i", normal, eff)).sum()

        step = (iters + 0.1) / (steps / 3) if iters < steps / 3 else (steps - iters) / (2 * steps / 3) * 0.9 + 0.1
        step *= -eff * learning_rate
        
        normal = rotate(normal, step)
        # normal = normalize(normal)
        if iters % 10 == 0:
            print(cost(normal), np.linalg.norm(normal, ord=2, axis = -1).mean(), abs(step).min(), abs(step).max(), debug)

# uniform_normal_vector(100, 3, 200, 0.007)
# uniform_normal_vector(1000, 100, 2000, 0.5)