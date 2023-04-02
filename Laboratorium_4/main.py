##
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

from typing import Union, List, Tuple

def chebyshev_nodes(n:int=10)-> np.ndarray:
    if type(n) != int or n<0:
        return None
    else:
        i:int
        x = np.cos([k * np.pi/n for k in range(0,n+1)])
        return x

"""Funkcja tworząca wektor zawierający węzły czybyszewa w postaci wektora (n+1,)

Parameters:
n(int): numer ostaniego węzła Czebyszewa. Wartość musi być większa od 0.
                                                                                                                                                                                        
Results:
np.ndarray: wektor węzłów Czybyszewa o rozmiarze (n+1,). 
            Jeżeli dane wejściowe niepoprawne funkcja zwraca None
"""

    
def bar_czeb_weights(n:int=10)-> np.ndarray:
    if type(n) == int:
        wagi_bar = np.zeros(n+1)
        for j in range(n + 1):
            if j == 0 or j == n:
                wagi_bar[j] = np.power(-1,j) * 0.5

            else:
                wagi_bar[j] = np.power(-1,j) * 1

        return wagi_bar
    else:
        return None


"""Funkcja tworząca wektor wag dla węzłów czybyszewa w postaci (n+1,)

Parameters:
n(int): numer ostaniej wagi dla węzłów Czebyszewa. Wartość musi być większa od 0.
 
Results:
np.ndarray: wektor wag dla węzłów Czybyszewa o rozmiarze (n+1,). 
            Jeżeli dane wejściowe niepoprawne funkcja zwraca None
"""

    
def  barycentric_inte(xi:np.ndarray,yi:np.ndarray,wi:np.ndarray,x:np.ndarray)-> np.ndarray:
    if type(xi) != np.ndarray or type(yi) != np.ndarray or type(wi) != np.ndarray or type(x) != np.ndarray\
            or xi.shape != yi.shape or xi.shape != wi.shape:
        return None
    else:
        Y = []
        for i in np.nditer(x):
            if i in xi:
                # omijamy dzielenie przez 0
                Y.append(yi[np.where(xi == i)[0][0]])
            else:
                # wzór w drugiej formie
                L = wi / (i - xi)
                Y.append(yi @ L / sum(L))
        return np.array(Y)








                    # """Funkcja przprowadza interpolację metodą barycentryczną dla zadanych węzłów xi
                    #     i wartości funkcji interpolowanej yi używając wag wi. Zwraca wyliczone wartości
                    #     funkcji interpolującej dla argumentów x w postaci wektora (n,) gdzie n to dłógość
                    #     wektora n.
                    #
                    # Parameters:
                    # xi(np.ndarray): węzły interpolacji w postaci wektora (m,), gdzie m > 0
                    # yi(np.ndarray): wartości funkcji interpolowanej w węzłach w postaci wektora (m,), gdzie m>0
                    # wi(np.ndarray): wagi interpolacji w postaci wektora (m,), gdzie m>0
                    # x(np.ndarray): argumenty dla funkcji interpolującej (n,), gdzie n>0
                    #
                    # Results:
                    # np.ndarray: wektor wartości funkcji interpolujący o rozmiarze (n,).
                    #             Jeżeli dane wejściowe niepoprawne funkcja zwraca None
                    # """

    
def L_inf(xr:Union[int, float, List, np.ndarray],x:Union[int, float, List, np.ndarray])-> float:
    if type(xr) == type(x) and xr is not None and x is not None:
        if type(x) == int or type(x) == float:
            return np.abs(xr - x)
        elif type(x) == list and len(xr) == len(x):
            return max([np.abs(xr[i]-x[i]) for i in range(len(x))])
        elif type(x) ==np.ndarray and xr.shape == x.shape:
            return max([np.abs(xr[i]-x[i])for i in range(xr.shape[0])])
        else:
            return np.NaN
    else:
        return np.NaN



    """Obliczenie normy  L nieskończonośćg. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.
    
    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)
    
    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """

def f1(x: float) -> float:
    return np.sign(x)*x + np.power(x, 2)
def f2(x: float) -> float:
    return np.sign(x)*np.power(x, 2)
def f3(x: float) -> float:
    return np.power(np.abs(np.sin(5*x)), 3)
def f4_1(x: float) -> float:
    return 1/(1 + x**2)
def f4_25(x: float) -> float:
    return 1/(1 + 25*(x**2))
def f4_100(x: float) -> float:
    return 1/(1 + 100*(x**2))
def f5(x: float) -> float:
    return np.sign(x)
