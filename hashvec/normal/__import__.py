try:
    import jax
    import jax.numpy as np
    from functools import partial

    use_jax = True
    # from jax.config import config
    # config.update("jax_debug_nans", True)
except:
    import numpy as np

    use_jax = False

compute_dtype = np.float32
