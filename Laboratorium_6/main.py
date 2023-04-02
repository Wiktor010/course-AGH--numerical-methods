import numpy as np
import pickle
import numpy.random as npr
from typing import Union, List, Tuple
from matplotlib import pyplot as plt

def random_matrix_Ab(m: int):

    if isinstance(m,int) and m>0:
        A = np.random.randint(0, m , size=(m,m))
        b = np.random.randint(0, m, size=(m,))
        return A,b
    else:
        return None

    # """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    # Parameters:
    # m(int): rozmiar macierzy
    # Results:
    # (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
    #             Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    # """


def residual_norm(A:np.ndarray,x:np.ndarray, b:np.ndarray):
    if A[0].shape == x.shape and x.shape == b.shape:
        return np.linalg.norm(b - A@x)
    else:
        return None



    #"""Funkcja obliczająca normę residuum dla równania postaci:
    # Ax = b
    # 
    #   Parameters:
    #   A: macierz A (m,m) zawierająca współczynniki równania 
    #   x: wektor x (m.) zawierający rozwiązania równania 
    #   b: wektor b (m,) zawierający współczynniki po prawej stronie równania
    # 
    #   Results:
    #   (float)- wartość normy residuom dla podanych parametrów"""



def log_sing_value(n:int, min_order:Union[int,float], max_order:Union[int,float]):

    if n < 1 or type(min_order) == str or type(max_order) == str or n <= min_order:
        return None
    else:
        return np.logspace(min_order, max_order, n)

# """Funkcja generująca wektor wartości singularnych rozłożonych w skali logarytmiczne
#
#     Parameters:
#      n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
#      min_order(int,float): rząd najmniejszej wartości w wektorze wartości singularnych
#      max_order(int,float): rząd największej wartości w wektorze wartości singularnych
#     Results:
#      np.ndarray - wektor nierosnących wartości logarytmicznych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
#      """

    
def order_sing_value(n:int, order:Union[int,float] = 2, site:str = 'gre'):
    if n < 1 or type(order) == str or type(site) != str or (site != "gre" and site != "low"):
        return None
    else:
        sing = npr.rand(n)*10
        sorted_sing = sorted(sing)
        sing_value = sorted_sing[::-1]
        if site == "gre":
            sing_value[0] = sing_value[0] * 10 ** order
            return np.array(sing_value)
        elif site == "low":
            sing_value[-1] = sing_value[-1] * 10 ** order
            return np.array(sing_value)
        else:
            return None

# """Funkcja generująca wektor losowych wartości singularnych (n,) będących wartościami zmiennoprzecinkowymi losowanymi przy użyciu funkcji np.random.rand(n)*10.
#     A następnie ustawiająca wartość minimalną (site = 'low') albo maksymalną (site = 'gre') na wartość o  10**order razy mniejszą/większą.
#
#     Parameters:
#     n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
#     order(int,float): rząd przeskalowania wartości skrajnej
#     site(str): zmienna wskazująca stronnę zmiany:
#         - site = 'low' -> sing_value[-1] * 10**order
#         - site = 'gre' -> sing_value[0] * 10**order
#
#     Results:
#     np.ndarray - wektor wartości singularnych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
#     """



def create_matrix_from_A(A:np.ndarray, sing_value:np.ndarray):
    if type(A) == np.ndarray and type(sing_value) == np.ndarray:
        if A[0].shape == sing_value.shape:
            U, S, V = np.linalg.svd(A)
            a = U * sing_value
            if a.shape == V.shape:
                A2 = np.dot(a, V)
                return A2
    else:
        return None

# """Funkcja generująca rozkład SVD dla macierzy A i zwracająca otworzenie macierzy A z wykorzystaniem zdefiniowanego wektora warości singularnych
    #
    #         Parameters:
    #         A(np.ndarray): rozmiarz macierzy A (m,m)
    #         sing_value(np.ndarray): wektor wartości singularnych (m,)
    #
    #
    #         Results:
    #         np.ndarray: macierz (m,m) utworzoną na podstawie rozkładu SVD zadanej macierzy A z podmienionym wektorem wartości singularnych na wektor sing_valu """
    #

