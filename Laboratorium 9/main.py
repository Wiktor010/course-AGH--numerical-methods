import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle 
from inspect import isfunction


from typing import Union, List, Tuple

def fun(x):
    return np.exp(-2*x)+x**2-1

def dfun(x):
    return -2*np.exp(-2*x) + 2*x

def ddfun(x):
    return 4*np.exp(-2*x) + 2


def bisection(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    if (type(a) == int or type(a) == float) and (type(b) == int or type(b) == float)\
            and type(epsilon) == float and type(iteration) == int:
        if f(a) * f(b) < 0:
            it: int = 0
            c: float = (a+b)/2
            if f(c) == 0:
                return c, it
            else:
                while epsilon < abs(a-b) and it < iteration:

                    c = (a+b)/2
                    if abs(f(c)) <= epsilon:
                        break
                    elif f(a) * f(c) < 0:
                        b = c
                        it += 1
                    elif f(c) * f(b) < 0:
                        a = c
                        it += 1

                return (a+b)/2, it
        else:
            return None
    else:
        return None


    # '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.
    #
    # Parametry:
    # a - początek przedziału
    # b - koniec przedziału
    # f - funkcja dla której jest poszukiwane rozwiązanie
    # epsilon - tolerancja zera maszynowego (warunek stopu)
    # iteration - ilość iteracji
    #
    # Return:
    # float: aproksymowane rozwiązanie
    # int: ilość iteracji
    # '''



def secant(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    if (type(a) == int or type(a) == float) and (type(b) == int or type(b) == float) \
            and type(epsilon) == float and type(iteration) == int:
        if epsilon > 0 and iteration > 0:

            if f(a) * f(b) < 0:
                it : int = 0
                for it in range(iteration):

                    x = (f(b) * a - f(a) * b) / (f(b) - f(a))

                    if f(a) * f(x) <= 0:
                        b = x
                    elif f(a) * f(x) > 0:
                        a = x

                    if abs(b - a) < epsilon or abs(f(x)) < epsilon:
                        return x, it

                return (f(b) * a - f(a) * b) / (f(b) - f(a)), iteration

    return None

    # '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą siecznych.
    #
    # Parametry:
    # a - początek przedziału
    # b - koniec przedziału
    # f - funkcja dla której jest poszukiwane rozwiązanie
    # epsilon - tolerancja zera maszynowego (warunek stopu)
    # iteration - ilość iteracji
    #
    # Return:
    # float: aproksymowane rozwiązanie
    # int: ilość iteracji
    # '''


def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float], ddf: typing.Callable[[float], float], a: Union[int,float], b: Union[int,float], epsilon: float, iteration: int) -> Tuple[float, int]:
    if (type(a) == int or type(a) == float) and (type(b) == int or type(b)==float) \
        and type(epsilon) ==float and type(iteration) ==int:
        if (df(a)*df(b) >= 0) and (ddf(a)*ddf(b) >= 0):
            if f(a) * f(b) > 0:
                return None
            elif f((a+b)/2) == 0:
                return (a+b)/2, 0
            else:
                it: int = 0
                c: float = 0
                c_new: float = 0
                if f(b) * ddf(a) > 0:
                    c_new = b
                else:
                    c_new = a
                while abs(a-b) > epsilon and it < iteration:
                    c = c_new
                    if abs(f(c)) <= epsilon:
                        break
                    else:
                        c_new = c - f(c) / df(c)
                        b = c_new
                        a = c
                        it += 1
                return b, it
        else:
            return None
    else:
        return None

    # ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    # Parametry:
    # f - funkcja dla której jest poszukiwane rozwiązanie
    # df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    # ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    # a - początek przedziału
    # b - koniec przedziału
    # epsilon - tolerancja zera maszynowego (warunek stopu)
    # Return:
    # float: aproksymowane rozwiązanie
    # int: ilość iteracji
    # '''


