import os
import warnings

from .normal import uniform_normal_vector
from .normal.__import__ import compute_dtype, np, onp


def build_table(num_plain, dimansions, save_cache=True, verbose=False):
    assert num_plain > dimansions
    result = {}

    if verbose:
        import tqdm

        RANGE = tqdm.trange(dimansions, num_plain)
    else:
        RANGE = range(dimansions, num_plain)

    for vector_cnt in RANGE:
        file_name = f"/tmp/hashvec/{vector_cnt}_{dimansions}.npy"
        normal = None
        if os.path.exists(file_name):
            try:
                normal = np.asarray(onp.load(file_name), dtype=compute_dtype)
                normal /= np.linalg.norm(normal, ord=2, axis=-1, keepdims=True)
            except:
                pass
        if normal is None:
            normal = uniform_normal_vector(vector_cnt, dimansions)
            if save_cache:
                try:
                    os.makedirs("/tmp/hashvec", exist_ok=True)
                    onp.save(file_name, onp.asarray(normal, dtype=onp.float32))
                except Exception as e:
                    warnings.warn(e)

        result[(vector_cnt, dimansions)] = normal
    return result
