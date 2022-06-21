import math
import matplotlib.pyplot as plt
import numpy as np


def morlet(a):

    psi_cos = math.cos(5*a)
    psi_exp = (-a**2)/2
    psi = psi_cos * math.exp(psi_exp)

    return psi


def plot_morlet():
    number = []
    x = []
    for i in range(-100, 100):
        number.append(morlet(i / 10))
        x.append(i / 10)

    plt.plot(x, number)
    plt.show()


def multi_morlet(z, d, t):
    """

    :rtype: object
    """
     # plot_morlet()
    d_diagonal = np.zeros((d.__len__(), d.__len__()), int)
    np.fill_diagonal(d_diagonal, d)
    f = (np.array(z) - np.array(t))
    a_matrix = np.dot(d_diagonal, f.T)
    aj = math.sqrt(np.dot(a_matrix.T, a_matrix))
    output = morlet(aj)
    return output
