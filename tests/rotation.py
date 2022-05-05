import unittest
from hashvec.normal.utils import normalize, random_rot_mat, simple_rotation_matrix
from hashvec.normal.__import__ import *
import numpy as np
import random

close_opt = {"rtol": 1e-5, "atol": 1e-6}


def almostGreater(X, Y):
    return (np.isclose(X, Y, **close_opt) | (X >= Y)).all()


class Tests(unittest.TestCase):
    def test_random(self):
        CASE = 10
        for dim in [2, 3, 4, 8, 16, 32, 64]:
            A = normalize(np.random.normal(0, 1, [CASE, dim]))
            ROT = np.stack([random_rot_mat(dim) for _ in range(CASE)], axis=0)
            B = normalize(np.einsum("ij,ijk->ik", A, ROT))

            IND = simple_rotation_matrix(A, B)
            # print(abs(B - np.einsum("ij,ijk->ik", A, IND)).sum())
            # self.assertTrue(all(np.trace(ind) >= np.trace(rot) for ind, rot in zip(IND, ROT)), f"{CASE} {dim} : it is not optimal")
            self.assertTrue(
                all(
                    almostGreater(np.trace(ind), np.trace(rot))
                    for ind, rot in zip(IND, ROT)
                ),
                f"{CASE} {dim} : it is not optimal",
            )

            infered_error = np.linalg.norm(
                B - np.einsum("ij,ijk->ik", A, IND), ord=2, axis=-1
            )
            no_move_error = np.linalg.norm(B - A, ord=2, axis=-1)
            self.assertTrue(
                almostGreater(no_move_error, infered_error),
                f"{CASE} {dim} : it is not correct rotation",
            )
            self.assertTrue(
                np.mean(no_move_error) > np.mean(infered_error) * 10,
                f"{CASE} {dim} : it is not correct rotation {np.mean(no_move_error)} >? {np.mean(infered_error)}",
            )

            if dim > 2:
                samples = normalize(np.random.normal(0, 1, [100, dim]))
                sample_transform = normalize(np.einsum("aj,ijk->iak", samples, IND))
                dist_sample = np.linalg.norm(
                    samples[None, ...] - sample_transform, ord=2, axis=-1
                )
                dist_max = np.linalg.norm(A - B, ord=2, axis=-1)[:, None]
                self.assertTrue(
                    almostGreater(dist_max, dist_sample),
                    f"{CASE} {dim} : it is not minimal movement ({dist_max} >? {dist_sample})",
                )

    def test_non_rotate(self):
        CASE = 10
        for dim in [2, 3, 4, 8, 16]:
            A = normalize(np.random.normal(0, 1, [CASE, dim]))
            IND = simple_rotation_matrix(A, A)
            SIM = normalize(IND.reshape((IND.shape[0], -1))) @ normalize(
                np.eye(dim).reshape(-1)
            )
            self.assertTrue(
                [sim >= 0.9999 for sim in SIM], "{dim} : fail to make identity matrix"
            )

    def test_oth(self):
        for dim in [2, 3, 4, 8, 16]:
            for tries in range(10):
                A, B = np.zeros([2, 1, dim])
                case = [random.randrange(0, dim), random.randrange(0, dim)]
                A[0][case[0]] = 1
                B[0][case[1]] = 1
                ROT = np.eye(dim)
                ROT[case[0]][case[0]] = 0
                ROT[case[0]][case[1]] = 1
                ROT[case[1]][case[0]] = 1
                ROT[case[1]][case[1]] = 0

                IND = simple_rotation_matrix(A, A)
                SIM = normalize(IND.reshape((IND.shape[0], -1))) @ normalize(
                    ROT.reshape(-1)
                )
                self.assertTrue(
                    [sim >= 0.9999 for sim in SIM],
                    f"{tries} {dim} : fail to make identity matrix",
                )

    def test_dtype(self):
        A = np.zeros([1, 103])
        A[0][13] = 1
        IND = simple_rotation_matrix(A, A)
        self.assertEqual(IND.dtype, compute_dtype)


# if __name__ == "__main__":
#     unittest.main()
