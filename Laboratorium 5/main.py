import numpy as np
import scipy
import pickle

from typing import Union, List, Tuple


def first_spline(x: np.ndarray, y: np.ndarray):
    if type(y) == np.ndarray and type(x) ==np.ndarray:
        if len(y) == len(x):
            lst = []
            lst1 = []
            for i in range(len(x)-1):
                a= (y[i+1]-y[i])/(x[i+1]-x[i])
                b = y[i] - a*x[i]
                lst.append(a)
                lst1.append(b)
            return lst,lst1
        else:
            return None
    else:
         return None



"""Funkcja wyznaczająca wartości współczynników spline pierwszego stopnia.

Parametrs:
x(float): argumenty, dla danych punktów
y(float): wartości funkcji dla danych argumentów

return (a,b) - krotka zawierająca współczynniki funkcji linowych"""




def cubic_spline(x: np.ndarray, y: np.ndarray, tol=1e-100):
        """
        Interpolacja splajnów cubicznych

        Returns:
        b współczynnik przy x stopnia 1
        c współczynnik przy x stopnia 2
        d współczynnik przy x stopnia 3
        """
        try:
            if x.shape != y.shape:
                return None
            x = np.array(x)
            y = np.array(y)
            ### check if sorted
            if np.any(np.diff(x) < 0):
                idx = np.argsort(x)
                x = x[idx]
                y = y[idx]

            size = len(x)
            delta_x = np.diff(x)
            delta_y = np.diff(y)

            ### Get matrix A
            A = np.zeros(shape=(size, size))
            b = np.zeros(shape=(size, 1))
            A[0, 0] = 1
            A[-1, -1] = 1

            for i in range(1, size - 1):
                A[i, i - 1] = delta_x[i - 1]
                A[i, i + 1] = delta_x[i]
                A[i, i] = 2 * (delta_x[i - 1] + delta_x[i])
                ### Get matrix b
                b[i, 0] = 3 * (delta_y[i] / delta_x[i] - delta_y[i - 1] / delta_x[i - 1])

            ### Solves for c in Ac = b
            print('Jacobi Method Output:')
            c = jacobi(A, b, np.zeros(len(A)), tol=tol, n_iterations=1000)

            ### Solves for d and b
            d = np.zeros(shape=(size - 1, 1))
            b = np.zeros(shape=(size - 1, 1))
            for i in range(0, len(d)):
                d[i] = (c[i + 1] - c[i]) / (3 * delta_x[i])
                b[i] = (delta_y[i] / delta_x[i]) - (delta_x[i] / 3) * (2 * c[i] + c[i + 1])

            return b.squeeze(), c.squeeze(), d.squeeze()
        except:
            return None


def jacobi(A, b, x0, tol, n_iterations=300):
    """
    Iteracyjne rozwiązanie równania Ax=b dla zadanego x0

    Returns:
    x - estymowane rozwiązanie
    """

    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    counter = 0
    x_diff = tol + 1

    while (x_diff > tol) and (counter < n_iterations):  # iteration level
        for i in range(0, n):  # element wise level for x
            s = 0
            for j in range(0, n):  # summation for i !=j
                if i != j:
                    s += A[i, j] * x_prev[j]

            x[i] = (b[i] - s) / A[i, i]
        # update values
        counter += 1
        x_diff = (np.sum((x - x_prev) ** 2)) ** 0.5
        x_prev = x.copy()  # use new x for next iteration

    print("Number of Iterations: ", counter)
    print("Norm of Difference: ", x_diff)
    return x


def L_inf(xr: Union[int, float, List, np.ndarray], x: Union[int, float, List, np.ndarray])-> float:
    if type(xr) == type(x) and xr is not None and x is not None:
        if type(x) == int or type(x) == float:
            return np.abs(xr - x)
        elif type(x) == list and len(xr) == len(x):
            return max([np.abs(xr[i] - x[i]) for i in range(len(x))])
        elif type(x) == np.ndarray and xr.shape == x.shape:
            return max([np.abs(xr[i] - x[i]) for i in range(xr.shape[0])])
        else:
            return np.nan
    else:
        return np.nan


def barycentric_inte(xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray) -> np.ndarray:
        if type(xi) != np.ndarray or type(yi) != np.ndarray or type(wi) != np.ndarray or type(x) != np.ndarray \
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


