try:
    import jax.numpy as np

    use_jax = True
    compute_dtype = np.float32
    # from jax.config import config
    # config.update("jax_debug_nans", True)
except:
    import numpy as np

    use_jax = False
