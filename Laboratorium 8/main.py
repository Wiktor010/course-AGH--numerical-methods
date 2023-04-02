import numpy as np
import scipy as sp
import pickle
import numpy.random as npr

from typing import Union, List, Tuple, Optional


def diag_dominant_matrix_A_b(m: int) -> None:

    if isinstance(m, int) and m > 0:
        A = np.random.randint(0, 100, (m, m), dtype=int)
        b = np.random.randint(0, 100, (m,), dtype=int)

        max_in_rows_n_cols = np.sum(A, axis=0) + np.sum(A, axis=1)
        A = A + np.diag(max_in_rows_n_cols)

        return A, b
    else:

        return None
    # """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    # Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    # Parameters:
    # m int: wymiary macierzy i wektora
    #
    # Returns:
    # Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
    #                                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    # """



def is_diag_dominant(A: np.ndarray) -> Optional[bool]:
    if not isinstance(A, np.ndarray) or len(A.shape) != 2:
        return None
    elif A.shape[0] != A.shape[1]:
        return None
    else:
        m = A.shape[0]
        for i in range(m):
            row_sum: int = 0
            col_sum: int = 0
            for j in range(m):
                if j != i:
                    row_sum += A[i][j]
                    col_sum += A[j][i]
            if A[i][i] <= row_sum or A[i][i] <= col_sum:
                return False
        return True

    # """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    # Parameters:
    # A np.ndarray: macierz wejściowa
    #
    # Returns:
    # bool: sprawdzenie warunku
    #       Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    # """



def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(m, int) and m > 0:
        A = npr.randint(0, 9, (m, m), dtype=int)
        B = npr.randint(0, 9, (m,), dtype=int)
        A_sym = (A + A.T)/2
        return A_sym, B
    else:
        return None
    # """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    # Parameters:
    # m int: wymiary macierzy i wektora
    #
    # Returns:
    # Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
    #                                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    # """



def is_symmetric(A: np.ndarray) -> bool:
    if not isinstance(A,np.ndarray) or len(A.shape) !=2:
        return None
    elif A.shape[0] != A.shape[1]:
        return None
    else:
        m = A.shape[0]
        for i in range(m):
            for j in range(i ,m):
                if A[i][j] != A[i][j]:
                    return False
        return True

    # """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    # Parameters:
    # A np.ndarray: macierz wejściowa
    #
    # Returns:
    # bool: sprawdzenie warunku
    #       Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    # """


def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray) or not isinstance(x_init, np.ndarray) or \
        type(epsilon) !=float or type(maxiter) != int or maxiter < 0 or A.shape[1] !=A.shape[0] or b.shape[0] != x_init.shape[0] or \
        b.shape[0] != A.shape[0]:
        return None
    else:
        D = np.diag(np.diag(A))
        LU = A - D
        x = x_init
        D_inv = np.diag(1/np.diag(D))
        iterations:int = 0
        for i in range(maxiter):
            x_new = np.dot(D_inv, b - np.dot(LU,x))
            r_norm = np.linalg.norm(x_new - x)
            iterations +=1
            if r_norm < epsilon:
                return x_new, iterations
            x = x_new
        return x, iterations


    # """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    # Parameters:
    # A np.ndarray: macierz współczynników
    # b np.ndarray: wektor wartości prawej strony układu
    # x_init np.ndarray: rozwiązanie początkowe
    # epsilon Optional[float]: zadana dokładność
    # maxiter Optional[int]: ograniczenie iteracji
    #
    # Returns:
    # np.ndarray: przybliżone rozwiązanie (m,)
    #             Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    # int: iteracja
    # """


def random_matrix_Ab(m:int):
    # """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    # Parameters:
    # m(int): rozmiar macierzy
    # Results:
    # (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
    #             Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    # """
    if isinstance(m, int) and m > 0:
        A = npr.randint(0, m, size = (m,m), dtype = int)
        b = npr.randint(0, m, size=(m,), dtype=int)
        return A,b
    else:
        return None

def residual_norm(A:np.ndarray, x:np.ndarray, b:np.ndarray):

    if A[0].shape == x.shape and x.shape == b.shape:
        return np.linalg.norm(b - A@x)
    else:
        return None


