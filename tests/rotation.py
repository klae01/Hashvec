import random
import unittest

from hashvec.normal.__import__ import *
from hashvec.normal.utils import normalize, random_rot_mat, simple_rotation_matrix

close_opt = {"rtol": 1e-5, "atol": 1e-6}


def almostGreater(X, Y):
    return (onp.isclose(X, Y, **close_opt) | (X >= Y)).all()


class Tests(unittest.TestCase):
    def test_rotation(self):
        for dim in [2, 3, 4, 8, 16, 128]:
            for _ in range(10):
                rot = random_rot_mat(dim)
                tr = onp.trace(rot @ rot.T) / dim
                self.assertAlmostEqual(1, tr)

    def test_random(self):
        CASE = 10
        for dim in [2, 3, 4, 8, 16, 32, 64]:
            A = normalize(onp.random.normal(0, 1, [CASE, dim]))
            ROT = onp.stack([random_rot_mat(dim) for _ in range(CASE)], axis=0)
            B = normalize(onp.einsum("ij,ijk->ik", A, ROT))

            IND = simple_rotation_matrix(A, B)
            # print(abs(B - onp.einsum("ij,ijk->ik", A, IND)).sum())
            # self.assertTrue(all(onp.trace(ind) >= onp.trace(rot) for ind, rot in zip(IND, ROT)), f"{CASE} {dim} : it is not optimal")
            self.assertTrue(
                all(
                    almostGreater(onp.trace(ind), onp.trace(rot))
                    for ind, rot in zip(IND, ROT)
                ),
                f"{CASE} {dim} : it is not optimal",
            )

            infered_error = onp.linalg.norm(
                B - onp.einsum("ij,ijk->ik", A, IND), ord=2, axis=-1
            )
            no_move_error = onp.linalg.norm(B - A, ord=2, axis=-1)
            self.assertTrue(
                almostGreater(no_move_error, infered_error),
                f"{CASE} {dim} : it is not correct rotation",
            )
            self.assertTrue(
                onp.mean(no_move_error) > onp.mean(infered_error) * 10,
                f"{CASE} {dim} : it is not correct rotation {onp.mean(no_move_error)} >? {onp.mean(infered_error)}",
            )

            if dim > 2:
                samples = normalize(onp.random.normal(0, 1, [100, dim]))
                sample_transform = normalize(onp.einsum("aj,ijk->iak", samples, IND))
                dist_sample = onp.linalg.norm(
                    samples[None, ...] - sample_transform, ord=2, axis=-1
                )
                dist_max = onp.linalg.norm(A - B, ord=2, axis=-1)[:, None]
                self.assertTrue(
                    almostGreater(dist_max, dist_sample),
                    f"{CASE} {dim} : it is not minimal movement ({dist_max} >? {dist_sample})",
                )

    def test_non_rotate(self):
        CASE = 10
        for dim in [2, 3, 4, 8, 16]:
            A = normalize(onp.random.normal(0, 1, [CASE, dim]))
            IND = simple_rotation_matrix(A, A)
            SIM = normalize(IND.reshape((IND.shape[0], -1))) @ normalize(
                onp.eye(dim).reshape(-1)
            )
            self.assertTrue(
                [sim >= 0.9999 for sim in SIM], "{dim} : fail to make identity matrix"
            )

    def test_oth(self):
        for dim in [2, 3, 4, 8, 16]:
            for tries in range(10):
                A, B = onp.zeros([2, 1, dim])
                case = [random.randrange(0, dim), random.randrange(0, dim)]
                A[0][case[0]] = 1
                B[0][case[1]] = 1
                ROT = onp.eye(dim)
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
        A = onp.zeros([1, 103])
        A[0][13] = 1
        IND = simple_rotation_matrix(A, A)
        self.assertEqual(IND.dtype, compute_dtype)


# if __name__ == "__main__":
#     unittest.main()
