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

def gen_chebyshev_nodes(Ns: int, interval: List = [-1, 1], kind:str = "second") -> np.ndarray:
    """
    create chebyshev nodes (chebyshev zeros or extreme points) within given interval,
    e.g., if second kind chebyshev nodes, aka, Chebyshev–Gauss–Lobatto, consider an interval of [-1,1], and let X_0, X_1, ..., X_{Ns} be the set of Ns points
    across the interval in reverse order
    1>= X_0 > X_1 > X_2 ... >X_{Ns} >=-1, and X_{j} = cos(j*PI/Ns), j = 0,1,...,Ns
    please refer to Prof Trefethen's textbook on chebyshev spectral methods for details.
    :param Ns:
    :param interval:
    :return:
    """
    a, b = interval
    if kind == "first":
        x = np.cos(np.pi*(np.arange(Ns) + 0.5)/Ns)
    elif kind == "second":
        x = np.cos(np.pi*(np.arange(Ns))/(Ns - 1))
    else:
        raise ValueError(f"kind must be either first or second, got {kind}")
    return 0.5*(a + b) + 0.5*(b - a)*x

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


def sample_from_values(values: np.ndarray, interval: list, Ns: int, kind="second") -> Tuple[np.ndarray,np.ndarray]:
    """
    Take N function values sampled on equal spaced nodes in given interval and use
    linear interpolation to convert into chebyshev interpotants of Ns nodes.
    Note the equal spaced values should be fine enough for interpolate on chebyshev nodes.
    :param values:
    :param interval:
    :param Ns:
    :param kind:
    :return:
    """
    n = values.size
    x = gen_equal_spaced_nodes(n, interval)
    chebyshev_nodes = gen_chebyshev_nodes(Ns, interval, kind=kind)
    f = np.interp(chebyshev_nodes, x, values)
    return chebyshev_nodes, f


def even_data(data:np.ndarray)->np.ndarray:
    """
    Construct Extended Data Vector (equivalent to creating an
    even extension of the original function)
    Return: array of length 2(N-1)
    For instance, [0,1,2,3,4] --> [0,1,2,3,4,3,2,1]
    """
    return np.concatenate([data, data[-2:0:-1]],)


def dct(data:np.ndarray)->np.ndarray:
    """
    Compute DCT using FFT
    """
    N = len(data)//2
    fftdata = np.fft.fft(data, axis=0)[:N+1]
    fftdata /= N
    fftdata[0] /= 2.
    fftdata[-1] /= 2.
    if np.isrealobj(data):
        data = np.real(fftdata)
    else:
        data = fftdata
    return data


def cheb_fit(sampled:np.ndarray):
    """
    Compute Chebyshev coefficients for values located on Chebyshev points.
    sampled: array; first dimension is number of Chebyshev points
    """
    asampled = np.asarray(sampled)
    if len(asampled) == 1:
        return asampled
    evened = even_data(asampled)
    coeffs = dct(evened)
    return coeffs

def evaluate_chebyshev(x, coeffs:np.ndarray)->np.ndarray:
    n = len(coeffs)
    result = np.zeros_like(x, dtype=float)
    for k in range(n):
        result += coeffs[k] * np.cos(k * np.arccos(x))
    return result



if __name__ == "__main__":
    interval = [0, 1]
    import matplotlib.pyplot as plt
    N = 40
    x = gen_equal_spaced_nodes(N, interval)
    f = x + x**2 + np.sin(np.pi*2*x)
    Ns = 5
    cheb_nodes, fc = sample_from_values(f, interval, Ns)
    plt.plot(x, f, "r+")
    plt.plot(cheb_nodes, fc, "b+")
    plt.show()
    coeffs = cheb_fit(fc)
    print(f"coefficients {coeffs}")





