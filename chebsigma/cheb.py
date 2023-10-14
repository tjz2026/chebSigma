# @Author: Jiuzhou Tang
# @Time : 2023/10/14 14:11
from typing import List,Tuple,Dict,Optional,Callable
import numpy as np

def genChebyshevNodes(Ns: int, interval:List = [-1,1])->np.ndarray:
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
    return 0.5*(a + b) + 0.5*(b - a) *np.cos((np.pi/Ns)*np.arange(Ns +1))





if __name__ == "__main__":
    chebNodes = genChebyshevNodes(12,[0,1])
    import matplotlib.pyplot as plt
    plt.plot(chebNodes,"r+")
    plt.show()
