import numpy as np
import matplotlib.pyplot as plt

def obtain_lucas_numbers_by_an_n(n):
    L = np.zeros(n)
    L[0], L[1] = 1, 3

    for i in range(2, n):
        L[i] = L[i - 1] + L[i - 2]

    return L


def main():
    n = 10**3
    lucas_numbers = obtain_lucas_numbers_by_an_n(n)
    plt.figure(figsize=(8, 5))
    plt.rc('font', size=15)
    plt.title('Lucas numbers ($n = 10^3$)')
    plt.xlabel('$n$')
    plt.ylabel('$L_n$')
    plt.plot(lucas_numbers)
    plt.show()


main()
