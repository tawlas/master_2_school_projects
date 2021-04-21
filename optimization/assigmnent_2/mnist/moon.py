import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def generate_data(n_samples = 150):
    n_samples_per_class = n_samples // 3
    x, c = make_moons(2*n_samples_per_class, noise=0.08, random_state=20)
    x1, c1 = x[c==0], c[c==0]
    x2, c2 = x[c==1], c[c==1]
    x, c = make_moons(2*n_samples_per_class, noise=0.08, random_state=10)
    x3, c3 = x[c==0], c[c==0]
    x = np.vstack([x1, x2+(0.3,-0.2), x3+(2.6,0)] )
    c = np.hstack([c1, c2, c3+2] )
    x = x - x.mean(0)
    x = x / x.std(0)
    return x, c

def show_data(x, c, grid=None):
    colors = np.array([[ 0, 0.4, 1],[1,0,0.4],[0, 1, 0.5],[1, 0.7, 0.5]])
    plt.scatter(x[:,0], x[:,1], s=50, color=colors[c], edgecolor='k', linewidth=1.5)
    if grid:
        from matplotlib.colors import ListedColormap
        xcoords, ycoords, zcoords = grid
        extent = (xcoords.min(), xcoords.max(), ycoords.min(), ycoords.max())
        C = len(np.unique(zcoords))
        cmap = ListedColormap(colors[:C])
        plt.imshow(zcoords, origin='lower', extent=extent, cmap=cmap, alpha=0.2, aspect='auto')
    plt.show()