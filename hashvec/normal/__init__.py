import jax
import jax.numpy as np
from jax.config import config
config.update("jax_debug_nans", True) 

@jax.jit
def normalize(X):
    return X/np.linalg.norm(X, ord=2, axis = -1, keepdims=True)

@jax.jit
def cost(EYE_MAT, normal):
    convert = np.einsum("ijk,ik->ij", EYE_MAT, normal)
    cost=np.square(np.einsum("im,jm->ij", convert, convert))
    cost=(cost.sum()-np.einsum("ii", cost))
    return cost / (normal.shape[0] * (normal.shape[0]-1))

grad = jax.value_and_grad(cost, argnums=0)

def uniform_normal_vector(n: int, m: int, steps: int = 500, learning_rate = 0.25):
    # n dimension 
    # m normal vector
    EYE_MAT = np.identity(m)[None, :].repeat(n, axis = 0)
    normal = normalize(jax.random.normal(jax.random.PRNGKey(123), (n, m)))
    β1 = 0.5
    β2 = 0.999
    Ɛ = 1e-12
    M = 0
    S = 0
    for iters in range(steps):
        # Gradient
        cost_v, grad_v = grad(EYE_MAT, normal)
        gradient = -grad_v

        # Adabelief optimizer
        M += (1-β1) * (gradient - M)
        S += (1-β2) * (np.square(gradient-M) - S) + Ɛ
        S = np.maximum(S, 0)

        eff = M / (1-β1**(1 + iters))
        eff /= ( S / (1-β2**(1 + iters)) )**0.5 + Ɛ


        # scheduler
        step = (iters + 0.1) / (steps / 3) if iters < steps / 3 else (steps - iters) / (2 * steps / 3) * 0.9 + 0.1
        step *= learning_rate
        step *= eff

        scale = abs(np.linalg.det(EYE_MAT+step))
        step /= scale[..., None, None]
        
        delta = np.einsum("ijk,ik->ij", step, normal)

        # clip norm 
        norm = np.linalg.norm(delta, ord = 2, axis = -1, keepdims = True)
        delta = np.where(norm > 0.01, delta / norm * 0.01, delta)
        
        normal = normal / scale[..., None] + delta

        if iters % 10 == 0:
            print(cost_v, np.linalg.norm(normal, ord=2, axis = -1).mean(), (step).min(), (step).max(), abs(step).min(), abs(step).max())
        normal = normalize(normal)

# uniform_normal_vector(100, 3, 200, 0.007)
# uniform_normal_vector(1000, 100, 2000, 0.5) # 0.009009007
# uniform_normal_vector(102, 100, 2000, 0.04) # 0.00019802