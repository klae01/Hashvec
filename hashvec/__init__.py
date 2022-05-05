import os
import json
import numpy as np

from hashvec.normal import uniform_normal_vector


def build_table(num_plain, dimansions, verbose=False):
    assert num_plain < dimansions
    result = {}

    if verbose:
        import tqdm

        RANGE = tqdm.trange(dimansions, num_plain)
    else:
        RANGE = range(dimansions, num_plain)

    for vector_cnt in RANGE:
        file_name = f"/tmp/hashvec/{vector_cnt}_{dimansions}.json"
        if os.path.exists(file_name):
            with open(file_name, "r") as fp:
                normal = json.load(fp)
        else:
            normal = uniform_normal_vector(vector_cnt, dimansions)

            os.makedirs("/tmp/hashvec", exist_ok=True)
            with open(file_name, "w") as fp:
                json.dump(np.asarray(normal).tolist(), fp)

        result[(vector_cnt, dimansions)] = normal
    return result
