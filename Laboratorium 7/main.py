import numpy as np
import scipy as sp
from scipy import linalg
from  datetime import datetime
import pickle

from typing import Union, List, Tuple


def spare_matrix_Abt(m: int,n: int):
    if type(m) != int or type(n) != int:
        return None
    else:
        if m < 1 or n < 1:
            return None
        else:
            t: np.array = np.linspace(0, 1, m)
            b: np.array = np.cos(4*t)
            A: np.array[np.array] = np.vander(t, n, True)
            return A, b

"""Funkcja tworząca zestaw składający się z macierzy A (m,n), wektora b (m,)  i pomocniczego wektora t (m,) zawierających losowe wartości
Parameters:
m(int): ilość wierszy macierzy A
n(int): ilość kolumn macierzy A
Results:
(np.ndarray, np.ndarray): macierz o rozmiarze (m,n) i wektorem (m,)
            Jeżeli dane wejściowe niepoprawne funkcja zwraca None
"""

def square_from_rectan(A: np.ndarray, b: np.ndarray):
    if isinstance(A, np.ndarray) and isinstance(b, np.ndarray):
        At = np.transpose(A)
        return At@A, At@b

    else:
        return None
"""Funkcja przekształcająca układ równań z prostokątną macierzą współczynników na kwadratowy układ równań. Funkcja ma zwrócić nową macierz współczynników  i nowy wektor współczynników
Parameters:
  A: macierz A (m,n) zawierająca współczynniki równania
  b: wektor b (m,) zawierający współczynniki po prawej stronie równania
Results:
(np.ndarray, np.ndarray): macierz o rozmiarze (n,n) i wektorem (n,)
         Jeżeli dane wejściowe niepoprawne funkcja zwraca None
 """




def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    if isinstance(A, np.ndarray) and isinstance(x, np.ndarray) and isinstance(b, np.ndarray):
        if A.shape[0] == b.shape[0]:
            return np.linalg.norm(b - A @ x)
        else:
            return None
    else:
        return None


# """Funkcja obliczająca normę residuum dla równania postaci:
# Ax = b
#
#   Parameters:
#   A: macierz A (m,n) zawierająca współczynniki równania
#   x: wektor x (n,) zawierający rozwiązania równania
#   b: wektor b (m,) zawierający współczynniki po prawej stronie równania
#
#   Results:
#   (float)- wartość normy residuom dla podanych parametrów
#   """

