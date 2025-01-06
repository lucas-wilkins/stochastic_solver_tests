import numpy as np

def normal_generator(mu: float, sigma: float, x: np.ndarray, error: float=0.2):
    means = np.exp(-0.5*((x - mu)/sigma)**2)
    return means + np.random.randn(len(x))*error

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mu = 0
    sigma = 1

    x = np.linspace(-2, 2, 15)
    y_noisy = normal_generator(mu, sigma, x)
    y_exact = normal_generator(mu, sigma, x, 0)

    plt.plot(x, y_exact)
    plt.scatter(x, y_noisy)

    plt.show()
