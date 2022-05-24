import sys
import numpy as np
sys.path.append("./")

#from lib import common

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr

def save_distr(src, proj, name):
    plt.clf()
    p = np.arange(Vmin, Vmax+DELTA_Z, DELTA_Z)
    plt.subplot(2, 1, 1)
    plt.bar(p, src, width=0.5)
    plt.title("Source")
    plt.subplot(2, 1, 2)
    plt.bar(p, proj, width=0.5)
    plt.title("Projected")
    plt.savefig(name + ".png")


if __name__ == "__main__":
    np.random.seed(123)
    atoms = np.arange(Vmin, Vmax+DELTA_Z, DELTA_Z)

    # single peak distribution
    src_hist = np.zeros(shape=(1, N_ATOMS), dtype=np.float32)
    src_hist[0, N_ATOMS//2+1] = 1.0
    proj_hist = distr_projection(src_hist, np.array([2], dtype=np.float32), np.array([False]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    
    save_distr(src_hist[0], proj_hist[0], "peak-r=2")
    

    # normal distribution
    data = np.random.normal(size=1000, scale=3)
    hist = np.histogram(data, normed=True, bins=np.arange(Vmin - DELTA_Z/2, Vmax + DELTA_Z*3/2, DELTA_Z))

    src_hist = hist[0]
    proj_hist = distr_projection(np.array([src_hist]), np.array([2], dtype=np.float32), np.array([False]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(hist[0], proj_hist[0], "normal-r=2")
    #raise Exception("stop in the name of plod")

    # normal distribution, but done episode
    proj_hist = distr_projection(np.array([src_hist]), np.array([2], dtype=np.float32), np.array([True]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(hist[0], proj_hist[0], "normal-done-r=2")

    # clipping for out-of-range distribution
    proj_dist = distr_projection(np.array([src_hist]), np.array([10], dtype=np.float32), np.array([False]),
                                        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(hist[0], proj_dist[0], "normal-r=10")

    # test both done and not done, unclipped
    proj_hist = distr_projection(np.array([src_hist, src_hist]), np.array([2, 2], dtype=np.float32),
                                        np.array([False, True]), Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(src_hist, proj_hist[0], "both_not_clip-01-incomplete")
    save_distr(src_hist, proj_hist[1], "both_not_clip-02-complete")

    # test both done and not done, clipped right
    proj_hist = distr_projection(np.array([src_hist, src_hist]), np.array([10, 10], dtype=np.float32),
                                        np.array([False, True]), Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(src_hist, proj_hist[0], "both_clip-right-01-incomplete")
    save_distr(src_hist, proj_hist[1], "both_clip-right-02-complete")

    # test both done and not done, clipped left
    proj_hist = distr_projection(np.array([src_hist, src_hist]), np.array([-10, -10], dtype=np.float32),
                                        np.array([False, True]), Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(src_hist, proj_hist[0], "both_clip-left-01-incomplete")
    save_distr(src_hist, proj_hist[1], "both_clip-left-02-complete")

    pass
