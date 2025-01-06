import numpy as np

from normal_generator import normal_generator


def fitter(f, x, data, start_parameters, step_size=0.1, momentum=0.1, n_iters=1000):
    n_parameters = len(start_parameters)
    parameters = np.array(start_parameters, dtype=float)

    last_objective = np.sum((f(*parameters, x) - data) ** 2)

    points = [start_parameters]
    objective = [last_objective]

    last_step = np.zeros_like(parameters, dtype=float)

    for i in range(n_iters):
        print(i, parameters)

        # Random direction
        rel_probe = np.random.randn(n_parameters) * step_size

        probe = parameters + rel_probe

        new_objective = np.sum((f(*probe, x) - data) ** 2)
        delta = new_objective - last_objective

        momentumless_step = - rel_probe * np.sign(delta)*np.min([np.abs(delta), 1])

        step = momentum*last_step + (1-momentum)*momentumless_step

        parameters += step
        last_step = step

        last_objective = new_objective
        objective.append(last_objective)
        points.append(parameters.copy())

    return np.array(points), objective

if __name__ == "__main__":
    from normal_generator import normal_generator
    import matplotlib.pyplot as plt

    x = np.linspace(-3, 3, 15)
    data = normal_generator(0, 1, x, 0)

    points, objective = fitter(normal_generator, x, data, (2, 2), momentum=0)

    plt.figure("Path")
    plt.plot(points[:,0], points[:,1])

    plt.figure("objective")
    plt.plot(np.arange(len(objective)), objective)

    plt.show()