import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def gram_layout(D):
    D = (D + D.T) * 0.5
    r = D[0, :]
    c = D[:, 0][:, np.newaxis]
    M = 0.5 * (r ** 2 + c ** 2 - D ** 2)
    w, v = np.linalg.eig(M)
    x = np.real(v * np.sqrt(w))
    pos = {
        i: (x[i, 0], x[i, 1])
        for i in range(D.shape[0])
    }
    return pos


def plot(bikes, parks, cars, D, car_limits, capacity, paths=None, filename=None):
    G = nx.from_numpy_array(D)
    pos = gram_layout(D)
    nodes = D.shape[0]
    colors = ['black', ] + ['green', ] * bikes + ['blue', ] * parks

    if paths:
        rows, cols = {2: (1, 2), 3: (2, 2), 4: (2, 2), 5: (2, 3), 6: (2, 3)}[cars]
        fig, axs = plt.subplots(rows, cols)
        for i, path in enumerate(paths):
            ax = axs[i // cols, i % cols]
            e = [(0, path[0]), ] + list(zip(path, path[1:]))
            alpha = [1. if n == 0 or n in path else 0.1 for n in range(nodes)]
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=10, node_color=colors, alpha=alpha)
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=e)

        # remove empty axis
        if cars == 3 or cars == 5:
            axs[cars // cols, cars % cols].axis('off')
    else:
        nx.draw_networkx_nodes(G, pos, node_size=10, node_color=colors)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


if __name__ == "__main__":
    from utils import read, read_solution
    N = 1
    problem = read(N)
    sol = read_solution(N)
    plot(*problem, sol)
