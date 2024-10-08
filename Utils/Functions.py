import numpy as np


# Not sure why presentation used class full of static methods, by using functions, *it just works*
# All taken from https://www.sfu.ca/~ssurjano/optimization.html
def sphere(parameters):
    return np.sum(parameters ** 2)


def ackley(parameters):
    n = len(parameters)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(parameters ** 2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * parameters)) / n) + 20 + np.e


def rastrigin(parameters):
    return 10 * len(parameters) + np.sum(parameters ** 2 - 10 * np.cos(2 * np.pi * parameters))


def rosenbrock(parameters):
    return np.sum(100 * (parameters[1:] - parameters[:-1] ** 2) ** 2 + (parameters[:-1] - 1) ** 2)


def griewank(parameters):
    n = len(parameters)
    return np.sum(parameters ** 2) / 4000 - np.prod(np.cos(parameters / np.sqrt(np.arange(1, n + 1)))) + 1


def schwefel(parameters):
    return 418.9829 * len(parameters) - np.sum(parameters * np.sin(np.sqrt(np.abs(parameters))))


def levy(parameters):
    w = 1 + (parameters - 1) / 4
    return np.sin(np.pi * w[0]) ** 2 + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2)) + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)


def michalewicz(parameters):
    return -np.sum(np.sin(parameters) * np.sin(np.arange(1, len(parameters) + 1) * parameters ** 2 / np.pi))


def zakharov(parameters):
    return np.sum(parameters ** 2) + np.sum(0.5 * np.arange(1, len(parameters) + 1) * parameters) ** 2 + np.sum(0.5 * np.arange(1, len(parameters) + 1) * parameters) ** 4