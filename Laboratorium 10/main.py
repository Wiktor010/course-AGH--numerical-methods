import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from numpy.core._multiarray_umath import ndarray
from numpy.polynomial import polynomial as P
import pickle

# zad1
def polly_A(x: np.ndarray):

    if type(x) == np.ndarray:
        return P.polyfromroots(x)
    else:
        return None

    # """Funkcja wyznaczajaca współczynniki wielomianu przy znanym wektorze pierwiastków.
    # Parameters:
    # x: wektor pierwiastków
    # Results:
    # (np.ndarray): wektor współczynników
    #             Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    # """
    


def roots_20(a: np.ndarray):
    if type(a) == np.ndarray:
        w = a + 10 ** -10 * np.random.random_sample(a.shape)
        p = P.polyroots(w)
        return w, p
    else:
        return None
    # """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
    #     oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    # Parameters:
    # a: wektor współczynników
    # Results:
    # (np.ndarray, np. ndarray): wektor współczynników i miejsc zerowych w danej pętli
    #             Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    # """



# zad 2

def frob_a(wsp: np.ndarray):
    if type(wsp) == np.ndarray:
        F = np.zeros((wsp.size, wsp.size))
        for i in range(wsp.size - 1):
            F[i][i + 1] = 1
        for i in range(wsp.size):
            F[wsp.size - 1][i] = -wsp[i]
        W = np.linalg.eigvals(F)
        T, Z = scipy.linalg.schur(F)
        p = P.polyroots(wsp)
        return F, W, T, Z, p
    else:
        return None
    # """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
    #     oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    # Parameters:
    # a: wektor współczynników
    # Results:
    # (np.ndarray, np. ndarray, np.ndarray, np. ndarray,): macierz Frobenusa o rozmiarze nxn, gdzie n-1 stopień wielomianu,
    # wektor własności własnych, wektor wartości z rozkładu schura, wektor miejsc zerowych otrzymanych za pomocą funkcji polyroots
    #
    #             Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    # """
    
