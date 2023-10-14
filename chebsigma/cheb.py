# @Author: Jiuzhou Tang
# @Time : 2023/10/14 14:11
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np


def gen_equal_spaced_nodes(n: int, interval: List)->np.ndarray:
    """
    create n number of equally spaced nodes within interval (boundary not included, nodes in the
    grid centers), i.e., the nodes of the input values are defined in the interval [a,b] as
    x_i = a + (i + 0.5)*dx, dx = (b-a)/N, i = 0, 1, ..., N-1
    :param n:
    :param interval:
    :return:
    """
    a, b = interval
    dx = (b - a) / n
    x = (b - a) * np.linspace(0, 1, n, endpoint=False) + 0.5 * dx + a
    return x

def gen_chebyshev_nodes(Ns: int, interval: List = [-1, 1]) -> np.ndarray:
    """
    create Ns +1 number of chebyshev nodes (chebyshev extreme points) within given interval,
    e.g., consider an interval of [-1,1], and let X_0, X_1, ..., X_{Ns} be the set of Ns points
    across the interval in reverse order
    1>= X_0 > X_1 > X_2 ... >X_{Ns} >=-1, and X_{j} = cos(j*PI/Ns), j = 0,1,...,Ns
    please refer to Prof Trefethen's textbook on chebyshev spectral methods for details.
    :param Ns:
    :param interval:
    :return:
    """
    a, b = interval
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((np.pi / Ns) * np.arange(Ns + 1))


def sample_from_callable_function(f: Callable, cheb_nodes: np.ndarray) -> np.ndarray:
    """
    sample function f on chebyshev interpolation nodes via simple for loop (not efficient)
    :param f: a scalar callable function
    :param cheb_nodes:
    :return:
    """
    r = np.zeros_like(cheb_nodes.shape, dtype=np.float32)
    for i in range(len(cheb_nodes)):
        r[i] = f(cheb_nodes[i])
    return r


def sample_from_values(values: np.ndarray, interval: list, Ns: int) -> np.ndarray:
    """
    Take N function values sampled on equal spaced nodes in given interval and use
    linear interpolation to convert into chebyshev interpotants of Ns +1 nodes.
    The nodes of the input values are defined in the interval [a,b] as
    x_i = a + (i + 0.5)*dx, dx = (b-a)/N, i = 0, 1, ..., N-1
    Note the equal spaced values should be fine enough for interpolate on chebyshev nodes.
    :param values:
    :param interval:
    :return:
    """
    n = values.size
    x = gen_equal_spaced_nodes(n, interval)
    chebyshev_nodes = gen_chebyshev_nodes(Ns, interval)
    f = np.interp(chebyshev_nodes, x, values)
    return f


if __name__ == "__main__":
    Ns = 12
    interval = [0, 1]
    chebNodes = gen_chebyshev_nodes(Ns, interval)
    import matplotlib.pyplot as plt
    plt.plot(chebNodes, "r+")
    plt.show()
    x = gen_equal_spaced_nodes(Ns, interval)
    f = x + x**2
    fc = sample_from_values(f, interval, Ns)
    plt.plot(x, f, "r+")
    plt.plot(gen_chebyshev_nodes(Ns, interval), fc, "b+")
    plt.show()




