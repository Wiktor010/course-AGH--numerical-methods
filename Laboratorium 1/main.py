import numpy as np
import math


def cylinder_area(r:float,h:float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    r : float
    h : float
    if r > 0 and h > 0:
        return 2 * math.pi * r * r + 2 * math.pi * h * r
    else:
        return np.NaN

def fib(n:int):
    fibonacci = np.array([1, 1])
    if isinstance(n,int):
        if n < 0:
            return None
        elif n == 0:
            return None
        elif n == 1:
            return np.array([1])
        elif n == 2:
            return fibonacci
        else:
            for i in range(2, n):
                nastepny = fibonacci[i - 1] + fibonacci[i - 2]
                fibonacci = np.append(fibonacci, [nastepny])
            return np.array([fibonacci])
    else:
        return None

    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    return None

def matrix_calculations(a:float):

    a:float
    m= np.array(([[a,1,-a],[0,1,1],[-a,a,1]]))
    mdet = np.linalg.det(m)
    if mdet !=0:
        minv = np.linalg.inv(m)
    else:
        minv = float("NaN")

    mt = np.transpose(m)
    return minv,mt,mdet




def custom_matrix(m:int, n:int):

    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    m :int
    n :int


    if isinstance(m, int) and isinstance(n, int):


        if m < 0 or n < 0:
            return None
        else:
            z = np.zeros((m,n),dtype=int)
            for row in range(m):
                for col in range(n):
                    if row > col:
                        z[row][col] = row
                    else:
                        z[row][col] = col
            return z
    else:
        return None












