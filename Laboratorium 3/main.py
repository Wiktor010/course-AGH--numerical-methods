import numpy as np
import scipy
import pickle
import matplotlib
import matplotlib.pyplot as plt


from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych

    """
    if not isinstance(v,(int,float,List,np.ndarray)) or not isinstance(v_aprox,(int,float,List,np.ndarray)):
        return np.NaN

    if isinstance(v,(int,float)) and isinstance(v_aprox,(int,float)):
        return np.abs(v - v_aprox)

    elif isinstance(v, list) and isinstance(v_aprox, list): #oba typu lista
        if len(v) == len(v_aprox):
            odp = np.zeros(len(v), dtype=int)
            for i in range(len(v)):
                odp[i] = np.abs(v[i] - v_aprox[i])
            return odp
        else:
            return np.NaN

    elif isinstance(v,np.ndarray) and isinstance(v_aprox,(int,float)) or isinstance(v_aprox,np.ndarray) and isinstance(v,(int,float)):
        return np.abs(v-v_aprox)

    elif isinstance(v, np.ndarray) and isinstance(v_aprox, np.ndarray): #oba typu np.ndarray
        if all((m == n) or (m == 1) or (n == 1) for m, n in zip(v.shape[::-1], v_aprox.shape[::-1])):#all zwraca true jeśli warunek prawdziwy
            return np.abs(v - v_aprox)
        else:
            return np.NaN

    elif isinstance(v_aprox,list) and isinstance(v,np.ndarray):
        if len(v_aprox) ==v.shape[0]:#v.shape[0] - długość wierszy v
            lista = np.array([])
            for i in range(len(v_aprox)):
                lista = np.append(lista, abs(v[i] - v_aprox[i]))
            return lista
        else:
            return np.NaN

    elif isinstance(v,list) and isinstance(v_aprox,np.ndarray):
        if len(v) ==v_aprox.shape[0]:
            lista = np.array([])
            for i in range(len(v)):
                lista = np.append(lista, abs(v[i] - v_aprox[i]))
            return lista
        else:
            return np.NaN

    elif isinstance(v_aprox,(int,float)) and isinstance(v,list):
        lista = np.array([])
        for i in range(len(v)):
            lista = np.append(list,abs(v[i]- v_aprox))
        return lista
    elif isinstance(v,(int,float)) and isinstance(v_aprox,list):
        lista = np.array([])
        for i in range(len(v_aprox)):
            lista = np.append(lista,abs(v-v_aprox[i]))
        return lista






def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndar
    ray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    abso=absolut_error(v,v_aprox)
    if abso is np.NaN or isinstance(v,(int,float)) and v == 0 or isinstance(v,np.ndarray) and not v.any():
        return np.NaN
    elif isinstance(v, np.ndarray):
        return np.divide(abso,v)
    elif isinstance(abso,np.ndarray) and isinstance(v,list):
        results = np.zeros(len(v))
        for i in range(len(v)):
            if v[i] ==0:
                return np.NaN
            results[i] = abso[i]/v[i]
        return results
    else:
        return abso/v




def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(n ,int) and isinstance(c,(int,float)):
        b = 2**n
        p1 = b - b + c
        p2 = b + c - b
        return np.abs(p1-p2)
    else:
        return np.NaN



def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(x,(int,float)) or not isinstance(n,int) or n<0:
        return np.NaN
    else:
        e = 0
        for i in range(0,n):
            e=e+((1/np.math.factorial(i))*(x**i))
        exp_aprox = e
        return exp_aprox



def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych

    """
    if not isinstance(x, (int, float)) or not isinstance(k, int):
        return np.NaN
    else:
        if k < 0:
            return np.NaN
        elif k == 0:
            return 1
        elif k == 1:
            return np.cos(x)
        elif k > 1:
            coskx = 2 * np.cos(x) * coskx1(k - 1, x) - coskx1(k - 2, x)
            return coskx
        else:
            return np.NaN



def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 2.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx, sinkx float: aproksymowana wartość funkcji,
                        NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(x,(int,float)) or not isinstance(k,int):
        return  np.NaN
    else:
        if k < 0:
            return np.NaN
        elif k ==0:
            return 1,0
        elif k == 1:
            return np.cos(x),np.sin(x)
        elif k> 0:
            coskx = np.cos(x) * coskx2(k-1,x)[0] - np.sin(x) * coskx2(k-1,x)[1]
            sinkx = np.sin(x) * coskx2(k-1,x)[0] + np.cos(x) * coskx2(k-1,x)[1]
            return coskx,sinkx
        else:
            return np.NaN




def pi(n: int) -> float:
    """Funkcja znajdująca przybliżenie wartości stałej pi.
    Szczegóły w Zadaniu 5.
    
    Parameters:
    n Union[int, List[int], np.ndarray[int]]: liczba wyrazów w ciągu
    
    Returns:
    pi_aprox float: przybliżenie stałej pi,
                    NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(n,int) or n<1:
        return np.NaN
    else:
        s = 0
        for i in range(1,n+1):
            s = s + (1/i**2)
        pi_aprox = np.sqrt(6*s)
        return pi_aprox
