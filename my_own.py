import numpy as np
from sklearn.datasets import make_moons, make_blobs


np.random.seed(42)

moons = make_moons(n_samples=100, noise=0.05)
moons_x, moons_y = moons
moons_y += 3
moons_dataset = np.hstack([moons_x, moons_y.reshape(-1, 1)])

blobs = make_blobs(n_samples=300, n_features=2, centers=[(-0.6, -0.5), (1.3, 1.4), (2, 0.9)], cluster_std=0.3)
blobs_x, blobs_y = blobs
blobs_dataset = np.hstack([blobs_x, blobs_y.reshape(-1, 1)])

dataset = np.vstack([moons_dataset, blobs_dataset])

np.savetxt("my_own.txt", dataset)
