from functools import partial

try:

    import jax
    import jax.numpy as np

    use_jax = True
    # from jax.config import config
    # config.update("jax_debug_nans", True)
except:
    import numpy as np

    use_jax = False

import numpy as onp

compute_dtype = np.float32
