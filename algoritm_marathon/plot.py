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


def plot(bikes, parks, cars, D, car_limits, paths=None, filename=None):
    G = nx.from_numpy_array(D)
    pos = gram_layout(D)
    colors = ['black', ] + ['green', ] * len(bikes) + ['blue', ] * len(parks)

    nx.draw_networkx_nodes(G, pos, node_size=10, node_color=colors)
    if paths:
        e = []
        for path in paths:
            e.extend(zip(path, path[1:]))

        nx.draw_networkx_edges(G, pos, edgelist=e)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_all():
    from utils import read

    for n in range(1, 31):
        problem = read(n)
        plot(*problem, filename=f'./img/{n}.png')
        plt.clf()
        print(n)


if __name__ == "__main__":
    # plot_all()
    from utils import read
    problem = read(n=1)
    plot(*problem)
