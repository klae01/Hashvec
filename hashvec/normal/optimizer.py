from .__import__ import *


class optimizer:
    def __init__(self, beta_1=0.9, beta_2=0.995, eps=1e-16, clip_norm=0.0001):
        self.M = 0
        self.S = 0

        self.β1 = beta_1
        self.β2 = beta_2
        self.Ɛ = eps
        self.clip = clip_norm

        self.iters = 0

    def update(self, weight, gradient):
        self.iters += 1
        gradient -= np.einsum("ij,ik,ik->ij", weight, weight, gradient)

        self.M += (1 - self.β1) * (gradient - self.M)
        self.S += (1 - self.β2) * (np.square(gradient - self.M) - self.S) + self.Ɛ
        self.S = np.maximum(self.S, 0)

        eff = self.M / (1 - self.β1**self.iters)
        eff /= (self.S / (1 - self.β2**self.iters)) ** 0.5 + self.Ɛ

        eff -= np.einsum("ij,ik,ik->ij", weight, weight, eff)

        # # clip norm
        # norm = np.linalg.norm(eff, ord = 2, axis = -1, keepdims = True)
        # eff = np.where(norm > self.clip, eff / norm * self.clip, eff)

        norm = np.linalg.norm(eff, ord=2, axis=-1, keepdims=True).max()
        eff *= self.clip / np.maximum(self.clip, norm)

        return eff
