import numpy as np
from scipy.stats import t, norm
import matplotlib.pyplot as plt
from scipy.special import gamma

ns = [100, 1000, 10000]
alphas = [0.01, 0.1, 1, 10]
v = 3
mu = 1.5
sigma = 1
mean_, var_ = t.stats(v, mu, sigma, moments='mv')


def f(x, v, mu, std):
    return ((gamma((v + 1) / 2) /
            (gamma(v / 2) * np.sqrt(v * np.pi) * std)) *
            (1 + (1 / v) * ((x - mu) / std)**2)**(-(v + 1) / 2))


def plot_t_dist():
    x = np.linspace(-5, 5, 1000)
    plt.title('MCMC Student-t distribution')
    plt.plot(x, f(x, 3, 1.5, 1), color='#1d55a5')
    plt.axvline(x=mean_, color='red',
                label=f'mean={mean_}, std={round(np.sqrt(var_), 4)}')
    plt.axvspan(float(mean_) - np.sqrt(var_), float(mean_) +
                np.sqrt(var_), facecolor='#97d8ed')
    plt.grid()
    plt.show()


def MCMC(n, alpha, flag=False):
    x = np.zeros(n)
    accepted = 0
    if flag:
        print(f"MCMC: \nn={n}, alpha={alpha}\n")
    for i in range(n - 1):
        x_ = x[i] + norm.rvs(loc=x[0], scale=alpha)
        u = np.random.rand()
        if u < min(1, f(x_, v, mu, sigma) / f(x[i], v, mu, sigma)):
            accepted += 1
            x[i + 1] = x_
        else:
            x[i + 1] = x[i]
        if flag:
            print(f"Actual value={x[i]}, \t Proposed value={x_}")
            print(
                f"Hasting ratio={f(x_, v, mu, sigma) / f(x[i], v, mu, sigma)}")
            print(f"Uniform random number={u}")
            print(f"Next value={x[i+1]}\n")
    return x, round(accepted / n * 100, 2)


def plot_MCMC_values():
    print(f"Student-t: Mean={mean_}, Std={np.sqrt(var_)}")
    fig, ax = plt.subplots(3, 4,  # sharex=True, sharey=True,
                           gridspec_kw={'hspace': 0.12, 'wspace': 0.12},
                           figsize=(20, 10))
    fig.suptitle("MCMC Student-t distribution", fontsize=25)

    for i in range(len(ns)):
        for j in range(len(alphas)):
            X = MCMC(ns[i], alphas[j])
            ax[i, j].axhline(
                y=mean_, color='red', label=f'mean={round(np.mean(X[0]), 4)}, std={round(np.std(X[0]), 4)}')
            ax[i, j].axhspan(float(mean_) - np.sqrt(var_),
                             float(mean_) + np.sqrt(var_), facecolor='#97d8ed')
            ax[i, j].plot(
                X[0], label=f"alpha={alphas[j]}, n={ns[i]}, accept={X[1]}%", color='#1d55a5')
            ax[i, j].grid()
            ax[i, j].legend(loc=1)
    plt.show()


def plot_hist_MCMC():
    fig, ax = plt.subplots(3, 4,  # sharex=True, sharey=True,
                           gridspec_kw={'hspace': 0.12, 'wspace': 0.12},
                           figsize=(20, 10))
    fig.suptitle("MCMC Student-t distribution", fontsize=25)

    for i in range(len(ns)):
        for j in range(len(alphas)):
            X = MCMC(ns[i], alphas[j])
            ax[i, j].axvline(
                x=mean_, color='red', label=f'mean={round(np.mean(X[0]), 4)}, std={round(np.std(X[0]), 4)}')
            ax[i, j].axvspan(float(mean_) - np.sqrt(var_),
                             float(mean_) + np.sqrt(var_), facecolor='#97d8ed')
            ax[i, j].hist(X[0], density=True, bins=30,
                          label=f"alpha={alphas[j]}, n={ns[i]}, accept={X[1]}%")
            xs = np.linspace(-10, 10, 100)
            ax[i, j].plot(xs, f(xs, v, mu, sigma),
                          label="Student-t distribution")
            ax[i, j].grid()
            ax[i, j].legend(loc=1)
    plt.show()


def main():
    plot_t_dist()
    plot_MCMC_values()
    plot_hist_MCMC()


main()
