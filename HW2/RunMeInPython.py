import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def get_orbit(n, f):
    seed, X = np.random.rand(), np.zeros(n)
    X[0] = seed
    for i in range(1, 100):
        X[i] = f(X[i - 1])
    return X


def plot_orbits(f, m):
    plt.figure(figsize=(8, 5))
    plt.rc('font', size=15)
    plt.title(f'Orbits (Float, n = {m})', fontsize=20)
    plt.xlabel('$x_i$')
    plt.ylabel('$x_{i+1}$')

    legends = []

    for k in range(m):
        X = get_orbit(100, f)
        plt.plot([i for i in X][:-1], [i for i in X][1:])
        legends.append(f"$Orbit_{k}$")

    plt.legend(legends, loc=2)
    plt.show()


def get_rational_orbit(n, f):
    seed, Y = sp.Rational(1, 9), []
    Y.append(seed)

    for i in range(1, 100):
        Y.append(f(Y[i - 1]))
    return Y


def plot_rational_orbits(f, m):
    plt.figure(figsize=(8, 5))
    plt.rc('font', size=15)
    plt.title(f'Orbits (Rational, n = {m})', fontsize=20)
    plt.xlabel('$x_i$')
    plt.ylabel('$x_{i+1}$')

    legends = []

    for k in range(m):
        X = get_rational_orbit(100, f)
        plt.plot([i for i in X][:-1], [i for i in X][1:])
        legends.append(f"$Orbit_{k}$")

    plt.legend(legends, loc=2)
    plt.show()


def main():
    n = 100
    F = lambda x: 2.0 * x if 0.0 <= x < 0.5 else (2.0 * x) - 1.0
    F_ = lambda x: sp.Rational(2 * x) if 0.0 <= x < 0.5 else sp.Rational(2 * x - 1)

    plot_orbits(F, 2)
    plot_orbits(F, 10)
    plot_rational_orbits(F_, 2)


main()
