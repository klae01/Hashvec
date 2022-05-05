import jax
import jax.numpy as np
from jax.config import config
config.update("jax_debug_nans", True) 

@jax.jit
def normalize(X):
    return X/np.linalg.norm(X, ord=2, axis = -1, keepdims=True)

@jax.jit
def rotate(P, V):
    P /= np.linalg.norm(P, ord=2, axis = -1, keepdims=True)
    θ = np.linalg.norm(V, ord=2, axis = -1, keepdims=True)
    pos = P*np.cos(θ) + V*np.sinc(θ/np.pi)
    return pos

@jax.jit
def cost(normal):
    X=np.square(np.einsum("im,jm->ij", normal, normal))
    X=(X.sum()-np.einsum("ii", X))
    return X

grad = jax.value_and_grad(cost)

def uniform_normal_vector(n: int, m: int, steps: int = 500, learning_rate = 0.25):
    # n dimension 
    # m normal vector
    normal = normalize(jax.random.normal(jax.random.PRNGKey(123), (n, m)))
    β1 = 0.9
    β2 = 0.9975
    Ɛ = 1e-10
    M = 0
    S = 0
    for iters in range(steps):
        # Gradient
        cost_v, grad_v = map(lambda x: x/(n * (n-1)), grad(normal) )
        gradient = np.einsum("ij,ik->ijk", normal, grad_v)
        
        # Adabelief optimizer
        M += (1-β1) * (gradient - M)
        S += (1-β2) * (np.square(gradient-M) - S) + Ɛ
        S = np.maximum(S, 0)

        eff = M / (1-β1**(1 + iters))
        eff /= ( S / (1-β2**(1 + iters)) )**0.5 + Ɛ
        
        eff = np.einsum("ij,ijk->ik", 1 / normal / m, eff)
        eff -= np.einsum("ij,ik,ik->ij", normal, normal, eff)

        # # clip norm 
        # norm = np.linalg.norm(eff, ord = 2, axis = -1, keepdims = True)
        # eff = np.where(norm > 0.1, eff / norm * 0.1, eff)

        # scheduler
        step = (iters + 0.1) / (steps / 3) if iters < steps / 3 else (steps - iters) / (2 * steps / 3) * 0.9 + 0.1
        step *= -learning_rate
        step *= eff
        
        normal = rotate(normal, step)
        # normal = normalize(normal)
        if iters % 10 == 0:
            print(cost_v, np.linalg.norm(normal, ord=2, axis = -1).mean(), abs(step).min(), abs(step).max())

# uniform_normal_vector(100, 3, 200, 0.007)
# uniform_normal_vector(1000, 100, 2000, 0.5) # 0.009009007
# uniform_normal_vector(102, 100, 2000, 0.04) # 0.00019802