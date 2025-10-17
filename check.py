import numpy as np
import matplotlib.pyplot as plt


def show(name):
    data = np.loadtxt(name, delimiter=",")
    samples = data[:]
    plt.figure()
    plt.title(name)
    plt.hist(samples, bins=100, density=True, alpha=0.6)
    ll = 10
    a = 0
    b = 1
    # x = np.linspace(a, b, 100)
    # plt.plot(x, ll * np.exp(-ll * (x-a)))

    plt.legend()


show("samples_exp.csv")
show("samples_lognormal.csv")

plt.show()
